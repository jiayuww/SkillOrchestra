"""
Failure-driven skill refinement.

Triggered when skill-based routing fails to achieve oracle accuracy on the
training set. Presents current skills and failed queries to an LLM, which
reflects on why routing failed and proposes new skills or splits.

This complements the variance-based refinement in refiner.py by using
concrete routing failures as evidence.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from ..llm.client import LLMClient
from .prompts import FAILURE_DRIVEN_REFINEMENT_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for LLM output
# ---------------------------------------------------------------------------

class ProposedNewSkill(BaseModel):
    skill_id: str
    name: str
    description: str
    indicators: List[str] = Field(default_factory=list)
    example_queries: List[str] = Field(default_factory=list)


class ProposedSubSkill(BaseModel):
    skill_id: str
    name: str
    description: str
    indicators: List[str] = Field(default_factory=list)
    distinguishing_feature: str = ""


class ProposedSplit(BaseModel):
    parent_skill_id: str
    rationale: str
    proposed_sub_skills: List[ProposedSubSkill] = Field(default_factory=list)


class FailureRefinementOutput(BaseModel):
    rationale: str = ""
    proposed_new_skills: List[ProposedNewSkill] = Field(default_factory=list)
    proposed_splits: List[ProposedSplit] = Field(default_factory=list)


try:
    from config import POOL_MODEL_DISPLAY_NAMES
except ImportError:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from config import POOL_MODEL_DISPLAY_NAMES


# ---------------------------------------------------------------------------
# FailureDrivenRefiner
# ---------------------------------------------------------------------------

@dataclass
class FailedQuery:
    """A query where oracle would have been correct but skill routing failed."""
    sample_id: int
    question: str
    ground_truths: List[str]
    oracle_models: List[str]
    routed_model: Optional[str]
    routed_correct: bool


@dataclass
class FailureRefinementResult:
    """Result of failure-driven refinement."""
    triggered: bool
    num_failed: int
    rationale: str = ""
    proposed_new_skills: List[Dict] = field(default_factory=list)
    proposed_splits: List[Dict] = field(default_factory=list)
    raw_output: Optional[Dict] = None


class FailureDrivenRefiner:
    """Refines skills based on routing failures vs oracle."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def refine(
        self,
        exploration_records: List[Dict],
        skill_routing_results: List[Dict],
        handbook: Dict,
        oracle_accuracy: float,
        skill_accuracy: float,
    ) -> FailureRefinementResult:
        """Run failure-driven refinement.

        Args:
            exploration_records: From run_exploration_for_set (oracle data)
            skill_routing_results: From run_skill_single (our routing)
            handbook: SkillHandbook JSON
            oracle_accuracy: Oracle accuracy on train
            skill_accuracy: Skill-based accuracy on train

        Returns:
            FailureRefinementResult with LLM's reflection and proposals
        """
        explore_by_id = {r["sample_id"]: r for r in exploration_records}
        skill_by_id = {r["sample_id"]: r for r in skill_routing_results}

        failed = []
        for sid, skill_rec in skill_by_id.items():
            explore_rec = explore_by_id.get(sid)
            if not explore_rec:
                continue

            oracle_em = explore_rec.get("oracle_em", 0)
            skill_em = skill_rec.get("exact_match", 0)

            if oracle_em >= 1.0 and skill_em < 1.0:
                oracle_models = explore_rec.get("oracle_models", [])
                models_called = skill_rec.get("models_called", [])
                routed = models_called[0] if models_called else None

                # Check if routed model was correct (in case we routed to right model but answer parsing failed)
                routed_correct = False
                if routed:
                    # Resolve display name to key for lookup
                    key_map = {v: k for k, v in POOL_MODEL_DISPLAY_NAMES.items()}
                    routed_key = key_map.get(routed, routed)
                    if routed_key in explore_rec.get("model_results", {}):
                        routed_correct = (
                            explore_rec["model_results"][routed_key].get("exact_match", 0) >= 1.0
                        )

                failed.append(FailedQuery(
                    sample_id=sid,
                    question=skill_rec.get("question", ""),
                    ground_truths=skill_rec.get("ground_truths", []),
                    oracle_models=[
                        POOL_MODEL_DISPLAY_NAMES.get(m, m) for m in oracle_models
                    ],
                    routed_model=routed,
                    routed_correct=routed_correct,
                ))

        if not failed:
            logger.info("No failed queries (skill routing matched oracle on train)")
            return FailureRefinementResult(
                triggered=False,
                num_failed=0,
            )

        # Don't trigger if we already match oracle
        if skill_accuracy >= oracle_accuracy:
            logger.info("Skill accuracy >= oracle, skipping failure refinement")
            return FailureRefinementResult(
                triggered=False,
                num_failed=len(failed),
            )

        logger.info(f"Failure-driven refinement: {len(failed)} failed queries")

        skill_catalog_text = self._format_skill_catalog(handbook)
        failed_queries_text = self._format_failed_queries(failed)

        prompt = FAILURE_DRIVEN_REFINEMENT_PROMPT.format(
            oracle_accuracy=oracle_accuracy,
            skill_accuracy=skill_accuracy,
            skill_catalog=skill_catalog_text,
            failed_queries=failed_queries_text,
        )

        self.llm.set_role("failure_refiner")
        try:
            output = self.llm.complete_structured(
                prompt=prompt,
                response_model=FailureRefinementOutput,
            )
        except Exception as exc:
            logger.warning(f"Failure refinement LLM call failed: {exc}")
            return FailureRefinementResult(
                triggered=True,
                num_failed=len(failed),
                rationale=f"LLM call failed: {exc}",
            )

        new_skills = [s.model_dump() for s in output.proposed_new_skills]
        splits = []
        for ps in output.proposed_splits:
            splits.append({
                "parent_skill_id": ps.parent_skill_id,
                "rationale": ps.rationale,
                "proposed_sub_skills": [s.model_dump() for s in ps.proposed_sub_skills],
            })

        return FailureRefinementResult(
            triggered=True,
            num_failed=len(failed),
            rationale=output.rationale,
            proposed_new_skills=new_skills,
            proposed_splits=splits,
            raw_output=output.model_dump(),
        )

    def _format_skill_catalog(self, handbook: Dict) -> str:
        """Format skill catalog for the prompt."""
        catalog = handbook.get("skill_catalog", {})
        categories = catalog.get("categories", {})

        lines = []
        for cat_id, cat in sorted(categories.items()):
            lines.append(f"\n### {cat.get('name', cat_id)}")
            lines.append(cat.get("description", ""))
            for sk_id, sk in cat.get("skills", {}).items():
                lines.append(f"  - **{sk_id}**: {sk.get('name', '')}")
                lines.append(f"    Description: {sk.get('description', '')}")
                ind = sk.get("indicators", [])
                if ind:
                    lines.append(f"    Indicators: {', '.join(ind)}")
                ex = sk.get("examples", [])
                if ex:
                    lines.append(f"    Examples: {ex[0][:80]}..." if len(str(ex[0])) > 80 else f"    Examples: {ex}")

        return "\n".join(lines) if lines else "(No skills in catalog)"

    def _format_failed_queries(self, failed: List[FailedQuery]) -> str:
        """Format failed queries for the prompt."""
        lines = []
        for i, fq in enumerate(failed[:20], 1):
            oracle_str = ", ".join(fq.oracle_models) if fq.oracle_models else "(none)"
            routed_str = fq.routed_model or "(no model called)"
            lines.append(f"\n{i}. Question: {fq.question[:200]}{'...' if len(fq.question) > 200 else ''}")
            lines.append(f"   Ground truth: {fq.ground_truths[:2]}")
            lines.append(f"   Oracle would use: {oracle_str}")
            lines.append(f"   We routed to: {routed_str} (correct={fq.routed_correct})")
        if len(failed) > 20:
            lines.append(f"\n... and {len(failed) - 20} more failed queries")
        return "\n".join(lines) if lines else "(No failed queries)"
