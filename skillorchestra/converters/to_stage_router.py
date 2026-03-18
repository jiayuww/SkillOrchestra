"""
Convert SkillOrchestra SkillHandbook to StageSkillHandbook-compatible JSON (for orchestration eval script).

This adapter produces a JSON file that the orchestration eval script
scripts can consume via StageSkillHandbook.load().

StageSkillHandbook JSON structure:
{
  "version": "...",
  "created_at": "...",
  "updated_at": "...",
  "skills": {
    "search": { skill_id: {...}, ... },
    "code": { ... },
    "answer": { ... }
  },
  "model_profiles": {
    "search-1": { ... },
    "reasoner-1": { ... },
    "answer-1": { ... }
  },
  "usage_patterns": { "stages": {...}, "guidelines": {...} },
  "routing_insights": [ "..." ],
  "learning_history": []
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from ..core.handbook import SkillHandbook
from ..core.types import AgentProfile, ModeMetadata, RoutingInsight, Skill

logger = logging.getLogger(__name__)

# Our modes -> StageSkillHandbook stage names
MODE_TO_STAGE = {
    "search": "search",
    "code": "code",
    "answer": "answer",
}

# Reverse: stage_router stage names -> our mode names
STAGE_TO_MODE = {
    "search": "search",
    "code": "code",
    "reasoning": "code",
    "answer": "answer",
}


def _convert_skill(skill: Skill) -> Dict[str, Any]:
    """Convert a Skill to StageSkillHandbook skill format."""
    discovered_from = []
    if skill.provenance and skill.provenance.discovered_from_queries:
        for qid in skill.provenance.discovered_from_queries[:10]:
            discovered_from.append({"problem_id": qid, "problem": "", "answer": ""})

    return {
        "skill_id": skill.skill_id,
        "name": skill.name,
        "description": skill.description,
        "stage": MODE_TO_STAGE.get(skill.mode, skill.mode),
        "examples": skill.examples[:5],
        "discovered_from_problems": discovered_from,
    }


def _convert_agent_profile(
    profile: AgentProfile,
    allowed_skill_ids: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Convert an AgentProfile to StageSkillHandbook model_profile format.

    When allowed_skill_ids is provided (e.g. for candidate handbooks), only
    include skill_competence for skills in that set. This ensures the
    converted handbook's format_skills() shows only the candidate's skills,
    not the full handbook's.
    """
    skill_scores = {}
    skill_attempts = {}
    skill_successes = {}

    for skill_id, bc in profile.skill_competence.items():
        if allowed_skill_ids is not None and skill_id not in allowed_skill_ids:
            continue
        attempts = bc.total_observations
        successes = max(0, int(bc.alpha - 1))
        skill_scores[skill_id] = round(bc.empirical_rate, 4)
        skill_attempts[skill_id] = attempts
        skill_successes[skill_id] = successes

    # For candidate handbooks (filtered skills): use skill-level sums so overall_rate
    # reflects performance on the candidate's skill set. Otherwise use trajectory-level.
    skill_total_attempts = sum(skill_attempts.values())
    skill_total_successes = sum(skill_successes.values())
    if allowed_skill_ids is not None and skill_total_attempts > 0:
        # Filtered: overall_rate = success rate on candidate's skills only
        total_attempts = skill_total_attempts
        total_successes = skill_total_successes
    elif profile.total_attempts > 0:
        total_attempts = profile.total_attempts
        total_successes = profile.total_successes
    else:
        total_attempts = skill_total_attempts
        total_successes = skill_total_successes
    overall_rate = total_successes / total_attempts if total_attempts > 0 else 0.0

    stage = MODE_TO_STAGE.get(profile.mode, profile.mode)

    return {
        "model_alias": profile.agent_id,
        "actual_model": profile.model_name,
        "stage": stage,
        "skill_scores": skill_scores,
        "skill_attempts": skill_attempts,
        "skill_successes": skill_successes,
        "overall_success_rate": round(overall_rate, 4),
        "total_attempts": total_attempts,
        "total_successes": total_successes,
        "avg_prompt_tokens": profile.cost_stats.avg_prompt_tokens,
        "avg_completion_tokens": profile.cost_stats.avg_completion_tokens,
        "avg_cost_usd": profile.cost_stats.avg_cost_usd,
        "strengths": profile.strengths[:3],
        "weaknesses": profile.weaknesses[:3],
    }


def _convert_usage_patterns(handbook: SkillHandbook) -> Dict[str, Any]:
    """Convert mode metadata + insights into usage_patterns."""
    stages = {}
    guidelines = {}

    for mode, meta in handbook.modes.items():
        stage = MODE_TO_STAGE.get(mode, mode)

        stages[stage] = {
            "stage_id": stage,
            "tool_name": "enhance_reasoning" if stage == "code" else stage,
            "avg_calls_per_query": 0.0,
            "avg_cost_per_call": 0.0,
            "avg_cost_per_query": 0.0,
            "avg_tokens_per_call": 0.0,
            "avg_turns_per_query": 0.0,
            "success_rate_when_used": 0.0,
            "success_rate_when_skipped": 0.0,
            "necessity_score": 0.8,
        }

        use_when = []
        skip_when = []
        learned_insights = []

        for ins in meta.insights:
            if ins.insight_type == "usage":
                use_when.append(ins.content)
            elif ins.insight_type == "constraint":
                skip_when.append(ins.content)
            else:
                learned_insights.append(ins.content)

        guidelines[stage] = {
            "stage_id": stage,
            "use_when": use_when or [meta.description] if meta.description else [],
            "skip_when": skip_when,
            "learned_insights": learned_insights,
        }

    return {
        "stages": stages,
        "guidelines": guidelines,
        "models": {},
        "raw": {},
    }


def _convert_routing_insights(handbook: SkillHandbook) -> List[str]:
    """Convert all routing insights to a flat list of strings."""
    insights = []
    for mode, meta in handbook.modes.items():
        for ins in meta.insights:
            prefix = ""
            if ins.insight_type == "transition":
                prefix = "[ToolSelection] "
            elif ins.insight_type == "usage":
                prefix = "[Usage] "
            elif ins.insight_type == "constraint":
                prefix = "[Cost] "
            insights.append(f"{prefix}{ins.content}")

    for profile in handbook.agent_profiles.values():
        for signal in profile.routing_signals:
            insights.append(signal)

    return insights


def convert_handbook(handbook: SkillHandbook) -> Dict[str, Any]:
    """Convert a SkillHandbook to StageSkillHandbook-compatible dict.

    For candidate handbooks (subgraphs), only skills in the handbook are
    included in model_profiles' skill_scores. This ensures the prompt shows
    the candidate's skill set, not the full handbook's.

    Returns:
        Dict that can be serialized as JSON and loaded by
        StageSkillHandbook.load().
    """
    skills_by_stage: Dict[str, Dict[str, Any]] = {}
    allowed_skill_ids: Set[str] = set()
    for skill_id, skill in handbook.skills.items():
        allowed_skill_ids.add(skill_id)
        stage = MODE_TO_STAGE.get(skill.mode, skill.mode)
        if stage not in skills_by_stage:
            skills_by_stage[stage] = {}
        skills_by_stage[stage][skill_id] = _convert_skill(skill)

    model_profiles = {}
    for agent_id, profile in handbook.agent_profiles.items():
        model_profiles[agent_id] = _convert_agent_profile(
            profile, allowed_skill_ids=allowed_skill_ids
        )

    return {
        "version": handbook.version,
        "created_at": handbook.created_at,
        "updated_at": handbook.updated_at,
        "skills": skills_by_stage,
        "model_profiles": model_profiles,
        "usage_patterns": _convert_usage_patterns(handbook),
        "routing_insights": _convert_routing_insights(handbook),
        "learning_history": [],
    }


def save_as_stage_router(
    handbook: SkillHandbook, path: str | Path
) -> Path:
    """Convert and save a SkillHandbook as StageSkillHandbook JSON.

    Args:
        handbook: Our SkillHandbook
        path: Output file path

    Returns:
        Path where the file was saved
    """
    data = convert_handbook(handbook)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

    n_skills = sum(len(s) for s in data["skills"].values())
    n_models = len(data["model_profiles"])
    logger.info(
        f"Saved StageSkillHandbook-compatible JSON "
        f"({n_skills} skills, {n_models} models) to {path}"
    )
    return path
