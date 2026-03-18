"""
Phase 2: Handbook Refinement (split/merge).

Data-driven refinement triggered by Beta distribution statistics:
- Split: We use variance-based refinement (best achievable < oracle) 
  or failure-driven refinement (failure_refiner.py) to trigger splits.
- Merge: two skills under the same parent where all models 
  have the same performance on both → propose merge candidate. 
  LLM decides based on skill description whether to actually merge.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..core.handbook import SkillHandbook
from ..core.types import BetaCompetence, Skill, SkillProvenance
from ..llm.client import LLMClient
from skillorchestra.prompts.learning import (
    AGENT_ORCHESTRATION_MERGE_PROMPT,
    AGENT_ORCHESTRATION_SPLIT_PROMPT,
    SKILL_MERGE_PROMPT,
    SKILL_SPLIT_PROMPT,
)
from .versioner import HandbookVersioner, _increment_patch

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for LLM output
# ---------------------------------------------------------------------------

class ProposedSplit(BaseModel):
    skill_id: str
    name: str
    description: str
    indicators: List[str] = Field(default_factory=list)
    distinguishing_feature: str = ""


class SplitOutput(BaseModel):
    should_split: bool = False
    rationale: str = ""
    proposed_splits: List[ProposedSplit] = Field(default_factory=list)


class MergedSkillDef(BaseModel):
    skill_id: str
    name: str
    description: str
    indicators: List[str] = Field(default_factory=list)


class MergeOutput(BaseModel):
    should_merge: bool = False
    rationale: str = ""
    merged_skill: Optional[MergedSkillDef] = None
    alternative_explanation: str = ""  # When not merging: why they should remain separate


# ---------------------------------------------------------------------------
# Refinement candidates
# ---------------------------------------------------------------------------

@dataclass
class SplitCandidate:
    """A skill that might need splitting."""
    skill_id: str
    variance: float
    agent_scores: Dict[str, float]
    best_achievable: float
    current_best: float


@dataclass
class MergeCandidate:
    """A pair of skills that might be redundant."""
    skill_id_1: str
    skill_id_2: str
    max_perf_diff: float


@dataclass
class RefinementResult:
    """Result of a refinement round."""
    splits_proposed: int = 0
    splits_applied: int = 0
    merges_proposed: int = 0
    merges_applied: int = 0
    skills_added: List[str] = field(default_factory=list)
    skills_removed: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Refiner
# ---------------------------------------------------------------------------

class HandbookRefiner:
    """Refines the skill taxonomy based on competence statistics.

    Split trigger: skill where best achievable << oracle (high variance)
    Merge trigger: skill pair where all agents have |diff| <= threshold
    """

    def __init__(
        self,
        llm: LLMClient,
        split_variance_threshold: float = 0.15,
        merge_perf_threshold: float = 0.0,
        min_observations_for_split: int = 3,
        versioner: Optional[HandbookVersioner] = None,
        prompt_type: str = "model_routing",
        max_merge_credits: int = 50,
        max_split_credits: int = 1,
    ):
        self.llm = llm
        self.split_variance_threshold = split_variance_threshold
        self.merge_perf_threshold = merge_perf_threshold
        self.min_observations_for_split = min_observations_for_split
        self.versioner = versioner
        self.prompt_type = prompt_type
        self.max_merge_credits = max_merge_credits
        self.max_split_credits = max_split_credits

    def refine(self, handbook: SkillHandbook) -> RefinementResult:
        """Run a full refinement round: identify candidates, then apply.

        Stage_router-style: same-prefix merge pairs first, max_merge_credits,
        recalculate candidates after each merge. Splits limited by max_split_credits.
        """
        result = RefinementResult()

        # Merges: loop until no candidates or credits exhausted; recalc after each merge
        merges_applied_this_round = 0
        while merges_applied_this_round < self.max_merge_credits:
            merge_candidates = self._order_merge_candidates(
                self.find_merge_candidates(handbook)
            )
            if not merge_candidates:
                break
            logger.info(
                f"Found {len(merge_candidates)} merge candidates "
                f"({merges_applied_this_round}/{self.max_merge_credits} credits used)"
            )
            applied_this_batch = False
            for mc in merge_candidates:
                if merges_applied_this_round >= self.max_merge_credits:
                    break
                result.merges_proposed += 1
                applied = self._apply_merge(handbook, mc)
                if applied:
                    result.merges_applied += 1
                    result.skills_added.append(applied)
                    result.skills_removed.extend([mc.skill_id_1, mc.skill_id_2])
                    merges_applied_this_round += 1
                    applied_this_batch = True
                    handbook.version = _increment_patch(handbook.version)
                    if self.versioner:
                        self.versioner.save_version(
                            handbook,
                            f"Merged {mc.skill_id_1} + {mc.skill_id_2} -> {applied}",
                        )
                    break  # Recalculate candidates after each merge
            if not applied_this_batch:
                break

        # Splits: limit to max_split_credits, highest variance first
        split_candidates = self.find_split_candidates(handbook)
        splits_applied_this_round = 0
        if split_candidates:
            logger.info(f"Found {len(split_candidates)} split candidates")
            for sc in split_candidates:
                if splits_applied_this_round >= self.max_split_credits:
                    break
                result.splits_proposed += 1
                added = self._apply_split(handbook, sc)
                if added:
                    result.splits_applied += len(added)
                    result.skills_added.extend(added)
                    splits_applied_this_round += 1
                    handbook.version = _increment_patch(handbook.version)
                    if self.versioner:
                        self.versioner.save_version(
                            handbook,
                            f"Split {sc.skill_id} -> {', '.join(added)}",
                        )

        if (result.splits_applied > 0 or result.merges_applied > 0) and not self.versioner:
            handbook.version = self._increment_version(handbook.version)
        logger.info(
            f"Refinement complete: {result.merges_applied} merges, "
            f"{result.splits_applied} splits"
        )
        return result

    def _order_merge_candidates(
        self, candidates: List[MergeCandidate]
    ) -> List[MergeCandidate]:
        """Order merge candidates: same-prefix pairs first, random within each group."""
        if not candidates:
            return []

        def get_prefix(skill_id: str) -> str:
            if "." in skill_id:
                return ".".join(skill_id.split(".")[:-1])
            return ""

        same_prefix: List[MergeCandidate] = []
        other: List[MergeCandidate] = []
        for mc in candidates:
            p1 = get_prefix(mc.skill_id_1)
            p2 = get_prefix(mc.skill_id_2)
            if p1 and p1 == p2:
                same_prefix.append(mc)
            else:
                other.append(mc)
        random.shuffle(same_prefix)
        random.shuffle(other)
        return same_prefix + other

    # ------------------------------------------------------------------
    # Find candidates
    # ------------------------------------------------------------------

    def find_split_candidates(self, handbook: SkillHandbook) -> List[SplitCandidate]:
        """Find skills with high variance that may need splitting."""
        candidates = []

        for skill_id, skill in handbook.skills.items():
            agents = handbook.get_agents_for_mode(skill.mode)
            scores = {}
            for agent in agents:
                bc = agent.skill_competence.get(skill_id)
                if bc and bc.total_observations >= self.min_observations_for_split:
                    scores[agent.agent_id] = bc.empirical_rate

            if len(scores) < 2:
                continue

            values = list(scores.values())
            mean_score = sum(values) / len(values)
            variance = sum((v - mean_score) ** 2 for v in values) / len(values)

            if variance >= self.split_variance_threshold:
                candidates.append(SplitCandidate(
                    skill_id=skill_id,
                    variance=variance,
                    agent_scores=scores,
                    best_achievable=max(values),
                    current_best=max(values),
                ))

        candidates.sort(key=lambda c: c.variance, reverse=True)
        return candidates

    def find_merge_candidates(self, handbook: SkillHandbook) -> List[MergeCandidate]:
        """Find skill pairs under the same parent that are statistically indistinguishable.

        Merge candidates: two skills under the same parent where all models have
        the same performance on both. LLM decides whether to merge based on
        skill description.
        """
        candidates = []
        skills_list = list(handbook.skills.values())

        for i, s1 in enumerate(skills_list):
            for s2 in skills_list[i + 1:]:
                if s1.mode != s2.mode:
                    continue
                # Only consider skills under the same parent
                if s1.parent_skill_id != s2.parent_skill_id:
                    continue

                agents = handbook.get_agents_for_mode(s1.mode)
                max_diff = 0.0
                has_enough_data = True

                for agent in agents:
                    bc1 = agent.skill_competence.get(s1.skill_id)
                    bc2 = agent.skill_competence.get(s2.skill_id)
                    if not bc1 or not bc2:
                        has_enough_data = False
                        break
                    diff = abs(bc1.empirical_rate - bc2.empirical_rate)
                    max_diff = max(max_diff, diff)

                if has_enough_data and max_diff <= self.merge_perf_threshold:
                    candidates.append(MergeCandidate(
                        skill_id_1=s1.skill_id,
                        skill_id_2=s2.skill_id,
                        max_perf_diff=max_diff,
                    ))

        return candidates

    # ------------------------------------------------------------------
    # Apply refinements
    # ------------------------------------------------------------------

    def _apply_merge(
        self, handbook: SkillHandbook, candidate: MergeCandidate
    ) -> Optional[str]:
        """Ask LLM to merge two skills, apply if approved."""
        s1 = handbook.get_skill(candidate.skill_id_1)
        s2 = handbook.get_skill(candidate.skill_id_2)
        if not s1 or not s2:
            return None

        perf_evidence = self._format_merge_evidence(handbook, candidate)

        if self.prompt_type == "agent_orchestration":
            skills_definitions = f"""### Skill 1: {s1.skill_id}
Name: {s1.name}
Description: {s1.description}
Examples: {', '.join(s1.examples[:3])}

### Skill 2: {s2.skill_id}
Name: {s2.name}
Description: {s2.description}
Examples: {', '.join(s2.examples[:3])}"""
            skill1_queries = "\n".join(f"- {q[:150]}" for q in s1.examples[:5]) or "(no samples)"
            skill2_queries = "\n".join(f"- {q[:150]}" for q in s2.examples[:5]) or "(no samples)"
            prompt = AGENT_ORCHESTRATION_MERGE_PROMPT.format(
                skills_definitions=skills_definitions,
                performance_correlation=perf_evidence,
                skill1_queries=skill1_queries,
                skill2_queries=skill2_queries,
            )
        else:
            prompt = SKILL_MERGE_PROMPT.format(
                skill_1_id=s1.skill_id,
                skill_1_name=s1.name,
                skill_1_description=s1.description,
                skill_2_id=s2.skill_id,
                skill_2_name=s2.name,
                skill_2_description=s2.description,
                performance_evidence=perf_evidence,
            )

        self.llm.set_role("skill_refiner")
        try:
            output = self.llm.complete_structured(prompt=prompt, response_model=MergeOutput)
        except Exception as exc:
            logger.warning(f"Merge analysis failed: {exc}")
            return None

        # Log and save decision (merge or not)
        reason = output.rationale if output.should_merge else (output.alternative_explanation or output.rationale)
        if output.should_merge:
            logger.info(f"  ✓ Merge {s1.skill_id} + {s2.skill_id}: {output.rationale}")
        else:
            logger.info(f"  ✗ No merge {s1.skill_id} + {s2.skill_id}: {reason}")
        if self.versioner:
            self.versioner.log_merge_decision(
                skill_id_1=s1.skill_id,
                skill_id_2=s2.skill_id,
                applied=output.should_merge,
                rationale=output.rationale,
                alternative_explanation=output.alternative_explanation,
                merged_skill_id=output.merged_skill.skill_id if output.merged_skill else None,
            )

        if not output.should_merge or not output.merged_skill:
            return None

        ms = output.merged_skill
        merged_skill = Skill(
            skill_id=ms.skill_id,
            name=ms.name,
            description=ms.description,
            indicators=ms.indicators,
            mode=s1.mode,
            provenance=SkillProvenance(
                refinement_history=[{
                    "action": "merge",
                    "sources": [s1.skill_id, s2.skill_id],
                    "rationale": output.rationale,
                }]
            ),
        )

        self._transfer_competence_for_merge(handbook, s1, s2, merged_skill)

        handbook.add_skill(merged_skill)
        handbook.remove_skill(s1.skill_id)
        handbook.remove_skill(s2.skill_id)

        logger.info(f"Merged {s1.skill_id} + {s2.skill_id} -> {merged_skill.skill_id}")
        return merged_skill.skill_id

    def _apply_split(
        self, handbook: SkillHandbook, candidate: SplitCandidate
    ) -> List[str]:
        """Ask LLM to split a skill, apply if approved."""
        skill = handbook.get_skill(candidate.skill_id)
        if not skill:
            return []

        perf_evidence = self._format_split_evidence(handbook, candidate)

        if self.prompt_type == "agent_orchestration":
            skill_definition = f"""ID: {skill.skill_id}
Name: {skill.name}
Description: {skill.description}
Examples: {'; '.join(skill.examples[:3])}"""
            performance_data = perf_evidence
            all_queries = skill.examples
            high_perf = "\n".join(f"- {q[:150]}" for q in all_queries[:5]) or "(none)"
            low_perf = "\n".join(f"- {q[:150]}" for q in all_queries[5:10]) or "(none)"
            divergent = "\n".join(f"- {q[:150]}" for q in all_queries[10:15]) or "(none)"
            prompt = AGENT_ORCHESTRATION_SPLIT_PROMPT.format(
                skill_definition=skill_definition,
                performance_data=performance_data,
                high_perf_queries=high_perf,
                low_perf_queries=low_perf,
                divergent_queries=divergent,
                success_trajectories="(not provided)",
                failure_trajectories="(not provided)",
            )
        else:
            prompt = SKILL_SPLIT_PROMPT.format(
                skill_id=skill.skill_id,
                skill_name=skill.name,
                skill_description=skill.description,
                mode=skill.mode,
                performance_evidence=perf_evidence,
                sample_queries="\n".join(skill.examples[:5]),
            )

        self.llm.set_role("skill_refiner")
        try:
            output = self.llm.complete_structured(prompt=prompt, response_model=SplitOutput)
        except Exception as exc:
            logger.warning(f"Split analysis failed: {exc}")
            return []

        # Log and save decision (split or not)
        if output.should_split:
            logger.info(f"  ✓ Split {skill.skill_id}: {output.rationale}")
        else:
            logger.info(f"  ✗ No split {skill.skill_id}: {output.rationale}")
        if self.versioner:
            self.versioner.log_split_decision(
                skill_id=skill.skill_id,
                applied=output.should_split,
                rationale=output.rationale,
                proposed_splits=[ps.skill_id for ps in output.proposed_splits] if output.proposed_splits else None,
            )

        if not output.should_split or not output.proposed_splits:
            return []

        added = []
        for ps in output.proposed_splits:
            new_skill = Skill(
                skill_id=ps.skill_id,
                name=ps.name,
                description=ps.description,
                indicators=ps.indicators,
                mode=skill.mode,
                parent_skill_id=skill.skill_id,
                provenance=SkillProvenance(
                    refinement_history=[{
                        "action": "split_from",
                        "parent": skill.skill_id,
                        "rationale": output.rationale,
                    }]
                ),
            )
            handbook.add_skill(new_skill)
            added.append(new_skill.skill_id)
            logger.info(f"Split {skill.skill_id} -> {new_skill.skill_id}")

        return added

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _transfer_competence_for_merge(
        self,
        handbook: SkillHandbook,
        s1: Skill,
        s2: Skill,
        merged: Skill,
    ) -> None:
        """Transfer competence statistics to the merged skill."""
        for profile in handbook.get_agents_for_mode(s1.mode):
            bc1 = profile.skill_competence.get(s1.skill_id, BetaCompetence())
            bc2 = profile.skill_competence.get(s2.skill_id, BetaCompetence())
            merged_bc = BetaCompetence(
                alpha=(bc1.alpha + bc2.alpha) / 2,
                beta=(bc1.beta + bc2.beta) / 2,
            )
            profile.skill_competence[merged.skill_id] = merged_bc
            profile.skill_competence.pop(s1.skill_id, None)
            profile.skill_competence.pop(s2.skill_id, None)

    def _format_split_evidence(
        self, handbook: SkillHandbook, candidate: SplitCandidate
    ) -> str:
        parts = [f"Variance: {candidate.variance:.3f}"]
        for agent_id, score in sorted(candidate.agent_scores.items()):
            bc = handbook.agent_profiles[agent_id].skill_competence.get(candidate.skill_id)
            obs = bc.total_observations if bc else 0
            parts.append(f"  {agent_id}: {score:.1%} ({obs} observations)")
        return "\n".join(parts)

    def _format_merge_evidence(
        self, handbook: SkillHandbook, candidate: MergeCandidate
    ) -> str:
        parts = [f"Max performance difference: {candidate.max_perf_diff:.3f}"]
        s1 = handbook.get_skill(candidate.skill_id_1)
        if s1:
            for agent in handbook.get_agents_for_mode(s1.mode):
                bc1 = agent.skill_competence.get(candidate.skill_id_1)
                bc2 = agent.skill_competence.get(candidate.skill_id_2)
                if bc1 and bc2:
                    parts.append(
                        f"  {agent.agent_id}: skill1={bc1.empirical_rate:.1%}, skill2={bc2.empirical_rate:.1%}"
                    )
        return "\n".join(parts)

    @staticmethod
    def _increment_version(version: str) -> str:
        if version.startswith("v") and version[1:].isdigit():
            return f"v{int(version[1:]) + 1}"
        return version + ".1"
