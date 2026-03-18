"""
Live evaluation harness for candidate handbooks.

Invokes the existing eval scripts with a converted handbook 
to evaluate orchestrator performance with real tool execution.

Output of each eval run is a directory of per-query JSON files with:
    { "id": "...", "correct": true/false, "costs": { ... } }

This module parses those results into EvaluationResult objects
for Pareto-optimal selection.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..converters.to_stage_router import save_as_stage_router
from .candidates import CandidateHandbook
from .pareto import EvaluationResult

logger = logging.getLogger(__name__)


@dataclass
class LiveEvalConfig:
    """Configuration for live evaluation runs."""

    eval_script: str = ""
    model_name: str = "Qwen/Qwen3-8B"
    model_type: str = "Qwen/Qwen3-8B"
    model_config: str = ""
    routing_strategy: str = "weighted_avg"
    max_rounds: int = 20
    concurrency: int = 10
    tool_concurrency: int = 5
    dataset: str = "frames"
    extra_args: List[str] = field(default_factory=list)


@dataclass
class LiveRunResult:
    """Detailed result from a single live evaluation run."""

    candidate_name: str
    output_dir: str
    num_queries: int = 0
    num_correct: int = 0
    accuracy: float = 0.0
    total_cost: float = 0.0
    avg_cost: float = 0.0
    # Cost for selection: total_cost_all_models_all_tokens (prioritize accuracy, then this)
    total_cost_completion_only: float = 0.0
    avg_cost_completion_only: float = 0.0
    costs_breakdown: Dict[str, float] = field(default_factory=dict)  # Aggregated cost fields
    per_query: List[Dict[str, Any]] = field(default_factory=list)
    error: str = ""
    elapsed_s: float = 0.0


class LiveEvaluator:
    """Evaluates candidate handbooks by running the real orchestrator.

    Converts each candidate handbook to StageSkillHandbook-compatible JSON,
    invokes eval_frames.py, and parses the output for accuracy + cost.
    """

    def __init__(
        self,
        config: LiveEvalConfig,
        val_queries_path: str,
        work_dir: Optional[str] = None,
    ):
        self.config = config
        self.val_queries_path = val_queries_path
        self.work_dir = Path(work_dir) if work_dir else Path(tempfile.mkdtemp(prefix="so_eval_"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

        if not self.config.eval_script:
            # Prefer repo orchestration/eval_frames.py (correct config import)
            repo_root = Path(__file__).resolve().parent.parent.parent
            default_path = repo_root / "orchestration" / "eval_frames.py"
            if default_path.exists():
                self.config.eval_script = str(default_path)
            else:
                raise FileNotFoundError(
                    "eval_frames.py not found. Set config.eval_script."
                )

    def evaluate_candidate(
        self,
        candidate: CandidateHandbook,
    ) -> LiveRunResult:
        """Run live evaluation for a single candidate handbook.

        Steps:
        1. Convert handbook to StageSkillHandbook JSON
        2. Invoke eval_frames.py with the handbook
        3. Parse output directory for results
        4. Return LiveRunResult
        """
        run_dir = self.work_dir / candidate.name
        run_dir.mkdir(parents=True, exist_ok=True)

        handbook_path = run_dir / "handbook.json"
        save_as_stage_router(candidate.handbook, handbook_path)

        output_dir = run_dir / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        cmd = self._build_command(handbook_path, output_dir)
        logger.info(f"Running live eval for {candidate.name}: {' '.join(cmd[:6])}...")

        env = os.environ.copy()
        repo_root = Path(__file__).resolve().parent.parent.parent
        orchestration_dir = repo_root / "orchestration"
        env["REPO_PATH"] = str(orchestration_dir)
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")

        start = datetime.now()
        try:
            result = subprocess.run(
                cmd,
                capture_output=False,  # Let eval_frames print progress to stdout
                text=True,
                timeout=3600,
                cwd=str(Path(self.config.eval_script).parent),
                env=env,
            )
            elapsed = (datetime.now() - start).total_seconds()

            if result.returncode != 0:
                logger.error(
                    f"eval_frames.py failed for {candidate.name}: rc={result.returncode}"
                )
                return LiveRunResult(
                    candidate_name=candidate.name,
                    output_dir=str(output_dir),
                    error=f"eval_frames exited with code {result.returncode}",
                    elapsed_s=elapsed,
                )

        except subprocess.TimeoutExpired:
            elapsed = (datetime.now() - start).total_seconds()
            logger.error(f"Timeout for {candidate.name} after {elapsed:.0f}s")
            return LiveRunResult(
                candidate_name=candidate.name,
                output_dir=str(output_dir),
                error="Timeout after 3600s",
                elapsed_s=elapsed,
            )
        except Exception as exc:
            elapsed = (datetime.now() - start).total_seconds()
            return LiveRunResult(
                candidate_name=candidate.name,
                output_dir=str(output_dir),
                error=str(exc),
                elapsed_s=elapsed,
            )

        return self._parse_results(candidate.name, output_dir, elapsed)

    def evaluate_all_candidates(
        self,
        candidates: List[CandidateHandbook],
    ) -> List[EvaluationResult]:
        """Evaluate all candidates and return EvaluationResults.

        Args:
            candidates: List of candidate handbooks

        Returns:
            List of EvaluationResult objects, one per candidate
        """
        results = []
        for i, candidate in enumerate(candidates):
            logger.info(
                f"Evaluating candidate {i + 1}/{len(candidates)}: "
                f"{candidate.name} ({candidate.handbook.num_skills} skills)"
            )
            live_result = self.evaluate_candidate(candidate)

            # Save per-candidate summary (one per evaluation run)
            run_dir = Path(self.work_dir) / candidate.name
            self._save_candidate_summary(run_dir, live_result)

            eval_result = EvaluationResult(
                name=candidate.name,
                accuracy=live_result.accuracy,
                avg_cost=live_result.avg_cost,  # total_cost_all_models_all_tokens / num_queries
                num_skills=candidate.handbook.num_skills,
                granularity=candidate.granularity,
                score=live_result.accuracy,
                details={
                    "num_queries": live_result.num_queries,
                    "num_correct": live_result.num_correct,
                    "total_cost": live_result.total_cost,
                    "total_cost_completion_only": live_result.total_cost_completion_only,
                    "elapsed_s": live_result.elapsed_s,
                    "error": live_result.error,
                    "depth_map": candidate.depth_map,
                    "per_query": live_result.per_query,
                    "costs_breakdown": live_result.costs_breakdown,
                },
            )
            results.append(eval_result)

            logger.info(
                f"  {candidate.name}: acc={live_result.accuracy:.1%}, "
                f"cost_all_tokens=${live_result.avg_cost:.4f}, "
                f"time={live_result.elapsed_s:.0f}s"
            )

        return results

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save_candidate_summary(self, run_dir: Path, result: LiveRunResult) -> None:
        """Save per-candidate summary (accuracy, costs) for this evaluation run."""
        summary = {
            "name": result.candidate_name,
            "num_queries": result.num_queries,
            "num_correct": result.num_correct,
            "accuracy": result.accuracy,
            "total_cost_completion_only": result.total_cost_completion_only,
            "avg_cost_completion_only": result.avg_cost_completion_only,
            "total_cost_all_tokens": result.total_cost,
            "elapsed_s": result.elapsed_s,
            "costs_breakdown": result.costs_breakdown,
            "error": result.error or None,
        }
        path = run_dir / "summary.json"
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        logger.debug(f"Saved candidate summary to {path}")

    def _build_command(self, handbook_path: Path, output_dir: Path) -> List[str]:
        """Build the eval_frames.py command."""
        cmd = [
            sys.executable, self.config.eval_script,
            "--model_name", self.config.model_name,
            "--model_type", self.config.model_type,
            "--output_dir", str(output_dir),
            "--example_file_path", self.val_queries_path,
            "--max_rounds", str(self.config.max_rounds),
            "--routing_strategy", self.config.routing_strategy,
            "--handbook", str(handbook_path),
            "--concurrency", str(self.config.concurrency),
            "--tool_concurrency", str(self.config.tool_concurrency),
        ]
        # Omit --no_progress so eval_frames prints progress to stdout

        if self.config.model_config:
            cmd.extend(["--model_config", str(self.config.model_config)])

        if self.config.dataset:
            cmd.extend(["--dataset", self.config.dataset])

        cmd.extend(self.config.extra_args)
        return cmd

    def _parse_results(
        self, candidate_name: str, output_dir: Path, elapsed_s: float
    ) -> LiveRunResult:
        """Parse output JSON files from eval_frames.py.

        Each file is {query_id}.json with:
            { "id": "...", "correct": bool, "costs": { ... } }

        Selection uses total_cost_all_models_all_tokens (prioritize accuracy, then this cost).
        """
        per_query = []
        num_correct = 0
        total_cost = 0.0
        total_cost_completion_only = 0.0
        costs_agg: Dict[str, float] = {}

        json_files = sorted(output_dir.rglob("*.json"))
        for jf in json_files:
            try:
                with open(jf) as f:
                    data = json.load(f)

                correct = data.get("correct", False)
                costs = data.get("costs", {})
                cost_all = costs.get("total_cost_all_models_all_tokens") or costs.get("total_cost", 0.0)
                cost_completion = costs.get("total_cost_all_models_completion_only", 0.0)

                per_query.append({
                    "id": data.get("id", jf.stem),
                    "correct": correct,
                    "cost": cost_all,
                    "costs": costs,
                })

                if correct:
                    num_correct += 1
                total_cost += cost_all
                total_cost_completion_only += cost_completion
                for k, v in costs.items():
                    if isinstance(v, (int, float)):
                        costs_agg[k] = costs_agg.get(k, 0.0) + v

            except Exception as exc:
                logger.warning(f"Failed to parse result {jf}: {exc}")

        num_queries = len(per_query)
        accuracy = num_correct / num_queries if num_queries > 0 else 0.0
        avg_cost = total_cost / num_queries if num_queries > 0 else 0.0
        avg_cost_completion_only = total_cost_completion_only / num_queries if num_queries > 0 else 0.0

        logger.info(
            f"Parsed {num_queries} results for {candidate_name}: "
            f"{num_correct}/{num_queries} correct ({accuracy:.1%}), "
            f"cost_all_tokens=${total_cost:.4f}"
        )

        return LiveRunResult(
            candidate_name=candidate_name,
            output_dir=str(output_dir),
            num_queries=num_queries,
            num_correct=num_correct,
            accuracy=accuracy,
            total_cost=total_cost,
            avg_cost=avg_cost,
            total_cost_completion_only=total_cost_completion_only,
            avg_cost_completion_only=avg_cost_completion_only,
            costs_breakdown=costs_agg,
            per_query=per_query,
            elapsed_s=elapsed_s,
        )
