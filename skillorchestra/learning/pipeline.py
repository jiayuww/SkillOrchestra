"""
Full Skill Handbook Learning Pipeline.

Ties together:
  Phase 1a: Skill Discovery (discoverer.py)
  Phase 1b: Profile Construction (profiler.py)
  Phase 2:  Refinement (refiner.py + failure_refiner.py)

Also handles train/val splitting, oracle evaluation, version snapshots, and integration with HandbookStore for versioned persistence.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from ..core.handbook import SkillHandbook
from ..core.traces import ExplorationBundle
from ..llm.client import LLMClient
from .discoverer import SkillDiscoverer
from .profiler import AgentProfiler
from .refiner import HandbookRefiner, RefinementResult
from .versioner import HandbookVersioner

if TYPE_CHECKING:
    from ..selection.store import HandbookStore

logger = logging.getLogger(__name__)


@dataclass
class HandbookSnapshot:
    """A versioned snapshot of the handbook with evaluation metrics."""
    version: str
    num_skills: int
    num_agents: int
    oracle_accuracy: float = 0.0
    handbook_routing_accuracy: float = 0.0
    event: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    skill_ids: List[str] = field(default_factory=list)
    skills_added: List[str] = field(default_factory=list)
    skills_removed: List[str] = field(default_factory=list)
    skills_merged: List[str] = field(default_factory=list)
    skills_split: List[str] = field(default_factory=list)


@dataclass
class LearningConfig:
    """Configuration for the learning pipeline."""

    # Data splitting
    validation_ratio: float = 0.3
    train_samples: Optional[int] = None  # If set, use first N for learning (overrides ratio)
    val_samples: Optional[int] = None  # If set, use next N for validation (after train_samples)

    # Skill discovery
    max_discovery_bundles: Optional[int] = None
    max_pairs_per_prompt: int = 5
    discovery_prompt_type: str = "model_routing"  # "model_routing" or "agent_orchestration"

    # Profiling
    use_llm_for_skill_id: bool = True
    skill_id_traces_path: Optional[str] = None  # If set, save raw traces for skill identification (JSONL)

    # LLM for learning-critical operations (unified for model routing + agent orchestration)
    learning_llm_model: str = "gpt-5"  # Discovery, refinement, insight, summarization
    skill_id_model: Optional[str] = None  # Skill identification. If None, use learning_llm_model.

    # Refinement
    split_variance_threshold: float = 0.15
    merge_perf_threshold: float = 0.0
    min_observations_for_split: int = 3  # Split candidates: require this many obs per agent (variance + best-achievable checks)
    max_refinement_rounds: int = 3
    max_merge_credits: int = 50  # max merges per refinement phase
    max_split_credits: int = 1  # max splits per refinement phase

    # Evaluation (same setting as evaluate.py)
    use_full_router_eval: bool = True  # Full LLM router; indicator is fallback only
    router_model: str = "qwen2.5-3b-instruct"
    lambda_c: float = 0.1
    temperature: float = 0.6
    seed: Optional[int] = 42

    # Orchestration: real eval for agent orchestration, e.g., orchestration/eval_frames.py (when set, overrides use_full_router_eval)
    orchestration_eval_script: Optional[str] = None
    orchestration_model_config: str = ""
    orchestration_orchestrator: str = "Qwen/Qwen3-8B"
    orchestration_max_rounds: int = 20
    orchestration_concurrency: int = 10
    orchestration_tool_concurrency: int = 5

    # Output
    output_dir: Optional[str] = None

    # HandbookStore integration
    experiment_name: Optional[str] = None


@dataclass
class LearningResult:
    """Result of the full learning pipeline."""
    handbook: SkillHandbook
    snapshots: List[HandbookSnapshot]
    config: LearningConfig
    stats: Dict[str, Any] = field(default_factory=dict)


class LearningPipeline:
    """Orchestrates the full Skill Handbook learning process.

    1. Load exploration data
    2. Split into train / validation
    3. Phase 1a: Skill Discovery
    4. Phase 1b: Profile Construction
    5. Oracle evaluation
    6. Phase 2: Refinement (data-driven, iterative)
    7. Final evaluation
    8. Output best handbook
    """

    def __init__(
        self,
        llm: LLMClient,
        config: Optional[LearningConfig] = None,
        store: Optional[HandbookStore] = None,
    ):
        self.llm = llm
        self.config = config or LearningConfig()
        self.store = store

        skill_id_llm = llm
        if self.config.skill_id_model and self.config.skill_id_model != llm.model:
            skill_id_llm = LLMClient(model=self.config.skill_id_model)

        skill_id_traces_path = self.config.skill_id_traces_path
        if self.config.output_dir and skill_id_traces_path is None:
            skill_id_traces_path = str(Path(self.config.output_dir) / "skill_id_traces.jsonl")

        self.discoverer = SkillDiscoverer(
            llm=llm,
            max_pairs_per_prompt=self.config.max_pairs_per_prompt,
        )
        self.profiler = AgentProfiler(
            llm=llm,
            use_llm_for_skill_id=self.config.use_llm_for_skill_id,
            llm_skill_id=skill_id_llm,
            skill_id_traces_path=skill_id_traces_path,
        )
        versioner = None
        if self.config.output_dir:
            versioner = HandbookVersioner(Path(self.config.output_dir))
        self.refiner = HandbookRefiner(
            llm=llm,
            split_variance_threshold=self.config.split_variance_threshold,
            merge_perf_threshold=self.config.merge_perf_threshold,
            min_observations_for_split=self.config.min_observations_for_split,
            versioner=versioner,
            prompt_type=self.config.discovery_prompt_type,
            max_merge_credits=self.config.max_merge_credits,
            max_split_credits=self.config.max_split_credits,
        )

        self.snapshots: List[HandbookSnapshot] = []

    def run(
        self,
        bundles: List[ExplorationBundle],
        handbook: Optional[SkillHandbook] = None,
    ) -> LearningResult:
        """Run the full learning pipeline.

        Args:
            bundles: Exploration bundles
            handbook: Optional existing handbook to build on

        Returns:
            LearningResult with the learned handbook and metrics
        """
        handbook = handbook or SkillHandbook()
        stats: Dict[str, Any] = {"start_time": datetime.now().isoformat()}

        train_bundles, val_bundles = self._split_data(bundles)
        stats["train_size"] = len(train_bundles)
        stats["val_size"] = len(val_bundles)

        train_oracle = self._compute_oracle_accuracy(train_bundles)
        val_oracle = self._compute_oracle_accuracy(val_bundles)
        stats["train_oracle_accuracy"] = train_oracle
        stats["val_oracle_accuracy"] = val_oracle
        logger.info(f"Oracle accuracy: train={train_oracle:.1%}, val={val_oracle:.1%}")

        self._snapshot(handbook, "baseline", oracle_accuracy=val_oracle)
        self._persist_learning_log(stats)

        # -- Phase 1a: Skill Discovery --
        logger.info("=== Phase 1a: Skill Discovery ===")
        all_bundles = train_bundles + val_bundles
        discovery_bundles = all_bundles
        if self.config.max_discovery_bundles:
            discovery_bundles = all_bundles[: self.config.max_discovery_bundles]

        for mode in self._get_all_modes(all_bundles):
            handbook.add_mode(mode)

        logger.info(
            f"  Discovery sees {len(discovery_bundles)} bundles "
            f"({len(train_bundles)} train + {len(val_bundles)} val)"
        )
        new_skills = self.discoverer.discover_from_bundles(
            discovery_bundles,
            handbook,
            prompt_type=self.config.discovery_prompt_type,
        )
        stats["skills_discovered"] = len(new_skills)
        self._snapshot(handbook, "discovery", oracle_accuracy=val_oracle)
        self._persist_learning_log(stats)

        # -- Phase 1b: Profile Construction --
        logger.info("=== Phase 1b: Profile Construction ===")
        profile_stats = self.profiler.build_profiles(train_bundles, handbook)
        stats["profiling"] = profile_stats

        insights = self.profiler.distill_mode_insights(train_bundles, handbook)
        stats["insights_distilled"] = len(insights)

        routing_acc = self._evaluate_routing(val_bundles, handbook)
        stats["post_profiling_routing_accuracy"] = routing_acc
        self._snapshot(
            handbook, "profiling",
            oracle_accuracy=val_oracle,
            handbook_routing_accuracy=routing_acc,
        )
        self._persist_learning_log(stats)
        logger.info(f"Post-profiling routing accuracy: {routing_acc:.1%} (oracle: {val_oracle:.1%})")

        # -- Phase 2: Refinement --
        logger.info("=== Phase 2: Refinement ===")
        if self.refiner.versioner:
            self.refiner.versioner.save_initial_version(handbook)
        refinement_stats = []
        for round_idx in range(self.config.max_refinement_rounds):
            result = self.refiner.refine(handbook)
            refinement_stats.append({
                "round": round_idx,
                "splits_proposed": result.splits_proposed,
                "splits_applied": result.splits_applied,
                "merges_proposed": result.merges_proposed,
                "merges_applied": result.merges_applied,
            })

            if result.splits_applied == 0 and result.merges_applied == 0:
                logger.info(f"Refinement round {round_idx}: no changes, stopping")
                break

            if result.splits_applied > 0:
                re_profile_stats = self.profiler.build_profiles(train_bundles, handbook)
                logger.info(f"Re-profiled after refinement: {re_profile_stats['updates']} updates")

            routing_acc = self._evaluate_routing(val_bundles, handbook)
            stats["refinement_rounds"] = refinement_stats
            self._snapshot(
                handbook,
                f"refinement_round_{round_idx}",
                oracle_accuracy=val_oracle,
                handbook_routing_accuracy=routing_acc,
                details={"refinement": refinement_stats[-1]},
            )
            self._persist_learning_log(stats)
            logger.info(
                f"Refinement round {round_idx}: routing accuracy {routing_acc:.1%} "
                f"(splits={result.splits_applied}, merges={result.merges_applied})"
            )

        stats["refinement_rounds"] = refinement_stats

        # -- Profile Summarization --
        logger.info("=== Profile Summarization ===")
        self.profiler.summarize_profiles(handbook)

        # -- Final --
        final_acc = self._evaluate_routing(val_bundles, handbook)
        stats["final_routing_accuracy"] = final_acc
        stats["end_time"] = datetime.now().isoformat()
        handbook.learning_stats = stats
        self._snapshot(
            handbook, "final",
            oracle_accuracy=val_oracle,
            handbook_routing_accuracy=final_acc,
        )
        self._persist_learning_log(stats)
        logger.info(f"Final: {handbook.num_skills} skills, routing accuracy {final_acc:.1%}")

        if self.config.output_dir:
            self._save_outputs(handbook, stats)

        return LearningResult(
            handbook=handbook,
            snapshots=self.snapshots,
            config=self.config,
            stats=stats,
        )

    # ------------------------------------------------------------------
    # Data splitting
    # ------------------------------------------------------------------

    def _split_data(
        self, bundles: List[ExplorationBundle]
    ) -> tuple:
        """Split bundles into train and validation sets."""
        if self.config.train_samples is not None or self.config.val_samples is not None:
            if self.config.train_samples is not None and self.config.val_samples is not None:
                train = bundles[: self.config.train_samples]
                val = bundles[
                    self.config.train_samples : self.config.train_samples + self.config.val_samples
                ]
            elif self.config.train_samples is not None:
                train = bundles[: self.config.train_samples]
                val = bundles[self.config.train_samples :]
            else:
                n = self.config.val_samples or 0
                val = bundles[-n:] if n > 0 else []
                train = bundles[:-n] if n > 0 else bundles
            logger.info(f"Data split (explicit): {len(train)} train, {len(val)} validation")
        else:
            n_val = max(1, int(len(bundles) * self.config.validation_ratio))
            n_train = len(bundles) - n_val
            train = bundles[:n_train]
            val = bundles[n_train:]
            logger.info(f"Data split: {len(train)} train, {len(val)} validation")
        return train, val

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_oracle_accuracy(bundles: List[ExplorationBundle]) -> float:
        """Compute oracle accuracy: fraction of queries where any agent succeeded."""
        if not bundles:
            return 0.0
        return sum(b.oracle_accuracy for b in bundles) / len(bundles)

    @staticmethod
    def _identify_skills_by_indicators(
        query: str, skills: list,
    ) -> Dict[str, float]:
        """Fast per-query skill identification via keyword/indicator matching."""
        query_lower = query.lower()
        raw_scores: Dict[str, float] = {}

        for skill in skills:
            hits = 0
            for indicator in skill.indicators:
                if indicator.lower() in query_lower:
                    hits += 1
            if skill.examples:
                for ex in skill.examples:
                    overlap = set(ex.lower().split()) & set(query_lower.split())
                    if len(overlap) >= 3:
                        hits += 0.5
            if hits > 0:
                raw_scores[skill.skill_id] = hits

        if not raw_scores:
            return {s.skill_id: 1.0 / len(skills) for s in skills}

        total = sum(raw_scores.values())
        return {sid: w / total for sid, w in raw_scores.items()}

    @staticmethod
    def _evaluate_oracle_routing(
        bundles: List[ExplorationBundle],
        handbook: SkillHandbook,
    ) -> float:
        """
        Evaluate handbook routing using indicator-based skill identification. 
        Used only when use_full_router_eval=False. 
        Highly recommend live evaluation instead. Should be used very rarely.
        """
        if not bundles:
            return 0.0

        correct = 0
        total = 0

        for bundle in bundles:
            for mode in bundle.get_modes_explored():
                if mode == "reference":
                    continue
                agents = bundle.get_agents_for_mode(mode)
                if not agents:
                    continue

                mode_skills = handbook.get_skills_for_mode(mode)
                if not mode_skills:
                    continue

                skill_weights = LearningPipeline._identify_skills_by_indicators(
                    bundle.query, mode_skills,
                )

                selected = handbook.select_agent(mode, skill_weights)
                if selected and agents.get(selected, False):
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.0

    def _evaluate_orchestration_routing(
        self,
        bundles: List[ExplorationBundle],
        handbook: SkillHandbook,
    ) -> float:
        """Evaluate handbook routing by running real orchestration eval script."""
        if not bundles:
            return 0.0

        logger.info(
            f"Evaluating routing via orchestration eval script on {len(bundles)} validation bundles "
            f"(script={self.config.orchestration_eval_script})"
        )

        from ..converters.to_stage_router import save_as_stage_router

        repo_root = Path(__file__).resolve().parent.parent.parent
        # Use learning output dir when available; otherwise temp (ephemeral)
        if self.config.output_dir:
            eval_work_dir = (Path(self.config.output_dir) / "routing_eval").resolve()
            eval_work_dir.mkdir(parents=True, exist_ok=True)
            tmp = eval_work_dir
            logger.info(f"Routing eval artifacts: {tmp} (handbook.json, val_samples.jsonl, output/)")
        else:
            tmp = Path(tempfile.mkdtemp(prefix="so_routing_eval_"))

        try:
            handbook_path = tmp / "handbook.json"
            save_as_stage_router(handbook, handbook_path)

            val_jsonl = tmp / "val_samples.jsonl"
            with open(val_jsonl, "w") as f:
                for i, b in enumerate(bundles):
                    f.write(
                        json.dumps(
                            {
                                "id": b.query_id or f"val_{i}",
                                "question": b.query,
                                "answer": b.ground_truths[0] if b.ground_truths else "",
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

            output_dir = tmp / "output"
            output_dir.mkdir(parents=True, exist_ok=True)

            cmd = [
                sys.executable,
                self.config.orchestration_eval_script,
                "--model_name",
                self.config.orchestration_orchestrator,
                "--model_type",
                self.config.orchestration_orchestrator,
                "--output_dir",
                str(output_dir),
                "--example_file_path",
                str(val_jsonl),
                "--max_rounds",
                str(self.config.orchestration_max_rounds),
                "--routing_strategy",
                "weighted_avg",
                "--handbook",
                str(handbook_path),
                "--concurrency",
                str(self.config.orchestration_concurrency),
                "--tool_concurrency",
                str(self.config.orchestration_tool_concurrency),
                "--no_progress",
            ]
            if self.config.orchestration_model_config:
                cmd.extend(["--model_config", self.config.orchestration_model_config])

            env = os.environ.copy()
            env["REPO_PATH"] = str(repo_root / "orchestration")
            existing_pythonpath = env.get("PYTHONPATH", "")
            env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

            try:
                result = subprocess.run(
                    cmd,
                    cwd=str(Path(self.config.orchestration_eval_script).parent),
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=3600,
                )
            except subprocess.TimeoutExpired:
                logger.warning("eval_frames.py timed out for routing accuracy")
                return 0.0

            if result.returncode != 0:
                logger.warning(
                    f"eval_frames.py failed for routing accuracy: rc={result.returncode}\n"
                    f"{result.stderr[-500:] if result.stderr else ''}"
                )
                return 0.0

            num_correct = 0
            for jf in sorted(output_dir.rglob("*.json")):
                if jf.name.startswith("step_") or jf.name == "pipeline_config.json":
                    continue
                try:
                    with open(jf) as fh:
                        data = json.load(fh)
                    if data.get("correct"):
                        num_correct += 1
                except Exception:
                    pass

            return num_correct / len(bundles) if bundles else 0.0
        finally:
            if not self.config.output_dir and tmp.exists():
                shutil.rmtree(tmp, ignore_errors=True)

    def _evaluate_full_router_routing(
        self,
        bundles: List[ExplorationBundle],
        handbook: SkillHandbook,
    ) -> float:
        """Evaluate handbook routing using full LLM router.

        Uses router for skill identification (indicator fallback only when parse fails).
        Runs full inference: router → pool call → exact_match on answer.
        """
        if not bundles:
            return 0.0

        from ..converters.to_ar import convert_handbook

        hb_rsl = convert_handbook(handbook)

        repo_root = Path(__file__).resolve().parent.parent.parent
        sys_path = list(sys.path)
        sys.path.insert(0, str(repo_root))
        try:
            import model_routing.test_skill_routing as tsr
        finally:
            sys.path[:] = sys_path

        skill_catalog_text = tsr.extract_skill_catalog_text(hb_rsl)
        model_performance_text = tsr.extract_model_performance_text(hb_rsl)
        model_skill_scores = tsr.extract_model_skill_scores(hb_rsl)
        model_overall_rates = tsr.extract_model_overall_rates(hb_rsl)
        skill_indicators = tsr.extract_skill_indicators(hb_rsl)

        correct = 0
        for i, bundle in enumerate(bundles):
            sample = {
                "id": bundle.query_id or str(i),
                "sample_id": i,
                "question": bundle.query,
                "ground_truths": bundle.ground_truths,
            }
            result = tsr.run_inference(
                sample, hb_rsl,
                skill_catalog_text, model_performance_text,
                model_skill_scores, skill_indicators,
                model_overall_rates=model_overall_rates,
                router_model=self.config.router_model,
                routing_strategy="weighted_avg",
                lambda_c=self.config.lambda_c,
                temperature=self.config.temperature,
                seed=self.config.seed,
                verbose=False,
            )
            if result.get("exact_match", 0) >= 1.0:
                correct += 1

        return correct / len(bundles)

    def _evaluate_routing(
        self,
        bundles: List[ExplorationBundle],
        handbook: SkillHandbook,
    ) -> float:
        """Evaluate handbook routing.

        For orchestration: run real eval when orchestration_eval_script is set.
        For model routing: full LLM router (test_skill_routing) or oracle fallback.
        """
        if self.config.orchestration_eval_script and Path(self.config.orchestration_eval_script).exists():
            return self._evaluate_orchestration_routing(bundles, handbook)
        if self.config.use_full_router_eval:
            return self._evaluate_full_router_routing(bundles, handbook)
        return self._evaluate_oracle_routing(bundles, handbook)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_all_modes(bundles: List[ExplorationBundle]) -> List[str]:
        """Get all modes explored across bundles."""
        modes = set()
        for b in bundles:
            modes.update(b.get_modes_explored())
        modes.discard("reference")
        return sorted(modes)

    def _snapshot(
        self,
        handbook: SkillHandbook,
        event: str,
        oracle_accuracy: float = 0.0,
        handbook_routing_accuracy: float = 0.0,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Take a snapshot of the handbook state with skill diffs."""
        current_skills = set(handbook.skills.keys())
        prev_skills = set(self.snapshots[-1].skill_ids) if self.snapshots else set()

        added = sorted(current_skills - prev_skills)
        removed = sorted(prev_skills - current_skills)

        # Detect merges/splits from details if provided
        merged = details.get("merged_skills", []) if details else []
        split = details.get("split_skills", []) if details else []

        snap = HandbookSnapshot(
            version=handbook.version,
            num_skills=handbook.num_skills,
            num_agents=len(handbook.agent_profiles),
            oracle_accuracy=oracle_accuracy,
            handbook_routing_accuracy=handbook_routing_accuracy,
            event=event,
            details=details or {},
            skill_ids=sorted(current_skills),
            skills_added=added,
            skills_removed=removed,
            skills_merged=merged,
            skills_split=split,
        )
        self.snapshots.append(snap)

        if added or removed or merged or split:
            logger.info(
                f"  Snapshot [{event}]: {handbook.num_skills} skills "
                f"(+{len(added)} -{len(removed)} "
                f"merged:{len(merged)} split:{len(split)})"
            )

        exp = self.config.experiment_name
        if self.store and exp:
            self.store.save_snapshot(handbook, exp, event)
        elif self.config.output_dir:
            snap_dir = Path(self.config.output_dir) / "snapshots"
            snap_dir.mkdir(parents=True, exist_ok=True)
            snap_path = snap_dir / f"handbook_{event}.json"
            handbook.save(snap_path)

    def _persist_learning_log(self, stats: Dict[str, Any]) -> None:
        """Save learning_log immediately (for crash recovery). Called after each snapshot."""
        log_data = self._build_log_data(stats)
        exp = self.config.experiment_name
        if self.store and exp:
            self.store.save_learning_log(log_data, exp)
        elif self.config.output_dir:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            with open(out_dir / "learning_log.json", "w") as f:
                json.dump(log_data, f, indent=2)

    def _save_outputs(self, handbook: SkillHandbook, stats: Dict[str, Any]) -> None:
        """Save final outputs via HandbookStore (if available) or direct filesystem."""
        log_data = self._build_log_data(stats)
        exp = self.config.experiment_name

        if self.store and exp:
            self.store.save_learned(handbook, exp)
            self.store.save_learning_log(log_data, exp)
            logger.info(f"Saved outputs to store: experiment={exp}")
        elif self.config.output_dir:
            out_dir = Path(self.config.output_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            handbook.save(out_dir / "handbook_final.json")
            with open(out_dir / "learning_log.json", "w") as f:
                json.dump(log_data, f, indent=2)
            logger.info(f"Saved outputs to {out_dir}")

    def _build_log_data(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "config": {k: v for k, v in self.config.__dict__.items()},
            "stats": stats,
            "snapshots": [
                {
                    "version": s.version,
                    "num_skills": s.num_skills,
                    "num_agents": s.num_agents,
                    "oracle_accuracy": s.oracle_accuracy,
                    "handbook_routing_accuracy": s.handbook_routing_accuracy,
                    "event": s.event,
                    "timestamp": s.timestamp,
                    "details": s.details,
                    "skill_ids": s.skill_ids,
                    "skills_added": s.skills_added,
                    "skills_removed": s.skills_removed,
                    "skills_merged": s.skills_merged,
                    "skills_split": s.skills_split,
                }
                for s in self.snapshots
            ],
        }
