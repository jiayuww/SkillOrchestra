"""
Phase 1a: Skill Discovery via hierarchical taxonomy.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..core.handbook import SkillHandbook
from ..core.traces import ExplorationBundle, ExecutionTrace
from ..core.types import Skill, SkillProvenance
from ..llm.client import LLMClient
from skillorchestra.prompts.learning import (
    SKILL_DISCOVERY_PROMPT,
    AGENT_ORCHESTRATION_DISCOVERY_PROMPT,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class DiscoveredSkill(BaseModel):
    """Model routing: skill_id, mode. Agent orchestration: id (stage/mode in category)."""
    skill_id: str = ""
    id: str = ""
    name: str = ""
    description: str = ""
    indicators: List[str] = Field(default_factory=list)
    examples: List[str] = Field(default_factory=list)
    mode: str = ""


class DiscoveredCategory(BaseModel):
    """Model routing: name. Agent orchestration: stage/mode, name."""
    stage: str = ""
    name: str = ""
    description: str = ""
    skills: List[DiscoveredSkill] = Field(default_factory=list)


class SkillDiscoveryOutput(BaseModel):
    categories: List[DiscoveredCategory] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Discoverer
# ---------------------------------------------------------------------------

class SkillDiscoverer:
    """Discovers skills from exploration bundles using a taxonomy approach."""

    def __init__(
        self,
        llm: LLMClient,
        max_pairs_per_prompt: int = 5,
        max_trajectory_chars: int = 1500,
    ):
        self.llm = llm
        self.max_pairs_per_prompt = max_pairs_per_prompt
        self.max_trajectory_chars = max_trajectory_chars

    def discover_from_bundles(
        self,
        bundles: List[ExplorationBundle],
        handbook: SkillHandbook,
        modes: Optional[List[str]] = None,
        prompt_type: str = "model_routing",
    ) -> List[Skill]:
        """Discover skills from exploration bundles.

        prompt_type: "model_routing" (default) or "agent_orchestration"
        - model_routing: Uses SKILL_DISCOVERY_PROMPT with contrastive evidence
        - agent_orchestration: Uses AGENT_ORCHESTRATION_DISCOVERY_PROMPT

        Returns:
            List of newly discovered Skill objects
        """
        if not bundles:
            return []

        target_modes = modes
        if not target_modes:
            all_modes: set = set()
            for b in bundles:
                all_modes.update(b.get_modes_explored())
            all_modes.discard("reference")
            target_modes = sorted(all_modes)

        new_skills: List[Skill] = []
        use_agent_orchestration = prompt_type == "agent_orchestration"

        for mode in target_modes:
            logger.info(
                f"  Mode '{mode}': {len(bundles)} queries, "
                f"generating taxonomy ({prompt_type})..."
            )

            self.llm.set_role("skill_discoverer")
            output = None
            bundle_sizes = self._progressive_bundle_sizes(len(bundles))
            for size_idx, bundle_count in enumerate(bundle_sizes):
                cur_bundles = bundles[:bundle_count]
                prompt = self._build_discovery_prompt(
                    cur_bundles, mode, handbook, use_agent_orchestration
                )
                logger.info(
                    f"  Mode '{mode}': discovery prompt uses {bundle_count}/{len(bundles)} queries"
                )
                for attempt in range(3):
                    try:
                        output = self.llm.complete_structured(
                            prompt=prompt,
                            response_model=SkillDiscoveryOutput,
                        )
                        break
                    except Exception as exc:
                        logger.warning(
                            f"Skill discovery attempt {attempt + 1}/3 failed for mode {mode} "
                            f"(queries={bundle_count}): {exc}"
                        )
                        # If prompt is too long, immediately retry with fewer bundles.
                        if self._is_context_length_error(exc) and size_idx < len(bundle_sizes) - 1:
                            next_count = bundle_sizes[size_idx + 1]
                            logger.warning(
                                f"Context limit hit for mode {mode}; reducing discovery samples "
                                f"from {bundle_count} to {next_count} and retrying."
                            )
                            break
                        if attempt == 2 and size_idx == len(bundle_sizes) - 1:
                            logger.error(f"Skill discovery failed for mode {mode}: {exc}")
                if output is not None:
                    break
            if output is None:
                continue

            total_cat = len(output.categories)
            total_sk = sum(len(c.skills) for c in output.categories)
            logger.info(f"  Mode '{mode}': LLM proposed {total_cat} categories, {total_sk} skills")

            for cat in output.categories:
                skill_mode = (cat.stage or mode) if use_agent_orchestration else mode

                logger.info(f"    Category: {cat.name} ({skill_mode}) -- {cat.description[:80]}")
                for ds in cat.skills:
                    sid = (ds.id or ds.name) if use_agent_orchestration else (ds.skill_id or ds.name)
                    if sid in handbook.skills:
                        logger.debug(f"    Skipping duplicate: {sid}")
                        continue

                    query_ids = [b.query_id for b in bundles[:15]]
                    provenance = SkillProvenance(
                        discovered_from_queries=query_ids,
                        discovery_round=0,
                    )

                    skill = Skill(
                        skill_id=sid,
                        name=ds.name,
                        description=ds.description,
                        indicators=ds.indicators,
                        examples=ds.examples or [b.query[:200] for b in bundles[:3]],
                        mode=skill_mode or ds.mode,
                        provenance=provenance,
                    )

                    handbook.add_mode(skill_mode)
                    handbook.add_skill(skill)
                    new_skills.append(skill)
                    logger.info(f"      + {skill.skill_id}: {skill.name}")

        logger.info(f"Discovered {len(new_skills)} skills across {len(target_modes)} modes")
        return new_skills

    def _build_discovery_prompt(
        self,
        bundles: List[ExplorationBundle],
        mode: str,
        handbook: SkillHandbook,
        use_agent_orchestration: bool,
    ) -> str:
        if use_agent_orchestration:
            problems_text = self._format_problems_agent_orchestration(bundles, mode)
            return AGENT_ORCHESTRATION_DISCOVERY_PROMPT.format(sample_problems=problems_text)

        problems_text = self._format_problems(bundles, mode)
        contrastive_text = self._format_contrastive_evidence(bundles, mode)
        existing_skills = self._format_existing_skills(handbook, mode)
        return SKILL_DISCOVERY_PROMPT.format(
            sample_problems=problems_text,
            contrastive_evidence=contrastive_text or "(no contrastive pairs -- all models agreed)",
            existing_skills=existing_skills or "(none yet)",
        )

    @staticmethod
    def _is_context_length_error(exc: Exception) -> bool:
        msg = str(exc).lower()
        return (
            "maximum context length" in msg
            or "requested token count exceeds" in msg
            or "context length" in msg
            or "too many tokens" in msg
        )

    @staticmethod
    def _progressive_bundle_sizes(total: int) -> List[int]:
        """Return decreasing unique bundle counts for adaptive prompt shrink."""
        if total <= 1:
            return [total]
        candidates = [
            total,
            max(1, int(total * 0.75)),
            max(1, int(total * 0.5)),
            max(1, int(total * 0.35)),
            max(1, int(total * 0.2)),
            1,
        ]
        dedup: List[int] = []
        for c in candidates:
            if c not in dedup:
                dedup.append(c)
        return dedup

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def _format_problems(self, bundles: List[ExplorationBundle], mode: str) -> str:
        """Format ALL queries with every model's answer (model routing)."""
        parts = []
        for i, bundle in enumerate(bundles):
            traces = bundle.get_trajectories_for_mode(mode)
            if not traces:
                continue

            gt = ", ".join(bundle.ground_truths[:3]) if bundle.ground_truths else "(no ground truth)"

            parts.append(f"### Problem {i + 1}: {bundle.query[:300]}")
            parts.append(f"Ground truth: {gt[:200]}")

            for trace in traces:
                label = "CORRECT" if trace.task_success else "WRONG"
                answer = trace.final_answer or "(no answer)"
                parts.append(f"  {trace.varied_agent_id} [{label}]: {answer}")

            parts.append("")

        return "\n".join(parts) if parts else "(no data for this mode)"

    def _format_problems_agent_orchestration(self, bundles: List[ExplorationBundle], mode: str) -> str:
        """Format problems for agent orchestration."""
        parts = []
        for i, bundle in enumerate(bundles):
            traces = bundle.get_trajectories_for_mode(mode)
            gt = ", ".join(bundle.ground_truths[:3]) if bundle.ground_truths else "N/A"

            parts.append(f"**Problem {i + 1}:**")
            parts.append(bundle.query[:500])
            parts.append(f"**Answer:** {gt[:300]}")

            if traces:
                outcomes = ", ".join(
                    f"{t.varied_agent_id} {'✓' if t.task_success else '✗'}"
                    for t in traces
                )
                parts.append(f"**Model outcomes:** {outcomes}")

            parts.append("")

        return "\n".join(parts) if parts else "(no data for this mode)"

    def _format_contrastive_evidence(
        self, bundles: List[ExplorationBundle], mode: str
    ) -> str:
        """Format contrastive pairs as supplementary evidence."""
        pairs = []
        for bundle in bundles:
            for pos, neg in bundle.get_contrastive_pairs(mode):
                pairs.append({
                    "query": bundle.query,
                    "positive_agent": pos.varied_agent_id,
                    "negative_agent": neg.varied_agent_id,
                    "pos_answer": pos.final_answer or "(no answer)",
                    "neg_answer": neg.final_answer or "(no answer)",
                })

        if not pairs:
            return ""

        shown = pairs[:self.max_pairs_per_prompt]
        parts = [f"({len(pairs)} total contrastive pairs, showing {len(shown)})\n"]
        for i, p in enumerate(shown):
            parts.append(f"Pair {i+1}: \"{p['query'][:200]}\"")
            parts.append(f"  SUCCESS ({p['positive_agent']}): {p['pos_answer']}")
            parts.append(f"  FAILURE ({p['negative_agent']}): {p['neg_answer']}")
            parts.append("")

        return "\n".join(parts)

    def _format_existing_skills(self, handbook: SkillHandbook, mode: str) -> str:
        """Format existing skills for duplicate avoidance."""
        skills = handbook.get_skills_for_mode(mode)
        if not skills:
            return ""
        return "\n".join(f"- {s.skill_id}: {s.description}" for s in skills)
