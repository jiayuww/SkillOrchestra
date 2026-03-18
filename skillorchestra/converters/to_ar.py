"""
Convert SkillOrchestra SkillHandbook to RSL handbook format (for model routing).

The RSL (Router Skill Learner) handbook is consumed by:
  - model_routing/test_skill_routing.py  (testing with real router)

RSL handbook JSON structure:
{
  "version": 0,
  "total_experiences": N,
  "skill_catalog": {
    "categories": {
      "category_name": {
        "name": "...",
        "description": "...",
        "skills": {
          "category.skill_name": {
            "id": "category.skill_name",
            "name": "...",
            "description": "...",
            "category": "category_name",
            "indicators": [...],
            "examples": [...]
          }
        }
      }
    },
    "version": 1,
    "created_at": "...",
    "updated_at": "..."
  },
  "model_profiles": {
    "model_key": {
      "model_name": "model_key",
      "skill_scores": {skill_id: float},
      "skill_attempts": {skill_id: int},
      "skill_successes": {skill_id: int},
      "strengths": [...],
      "weaknesses": [...],
      "routing_rules": [...],
      "total_attempts": int,
      "total_successes": int,
      "updated_at": "...",
      "recent_success_summaries": [],
      "recent_failure_summaries": []
    }
  },
  "routing_insights": [...]
}
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.handbook import SkillHandbook
from ..core.types import AgentProfile, Skill

logger = logging.getLogger(__name__)


def _group_skills_by_category(skills: List[Skill]) -> Dict[str, List[Skill]]:
    """Group skills into categories based on the first segment of skill_id."""
    categories: Dict[str, List[Skill]] = {}
    for skill in skills:
        parts = skill.skill_id.split(".", 1)
        if len(parts) >= 2:
            cat = parts[0]
        else:
            cat = skill.mode or "general"
        categories.setdefault(cat, []).append(skill)
    return categories


def _convert_skill_catalog(handbook: SkillHandbook) -> Dict[str, Any]:
    """Convert all skills to RSL skill_catalog format."""
    all_skills = list(handbook.skills.values())
    grouped = _group_skills_by_category(all_skills)

    categories = {}
    for cat_name, skills in sorted(grouped.items()):
        skill_entries = {}
        for skill in skills:
            skill_entries[skill.skill_id] = {
                "id": skill.skill_id,
                "name": skill.name,
                "description": skill.description,
                "category": cat_name,
                "indicators": skill.indicators,
                "examples": skill.examples[:5],
            }
        categories[cat_name] = {
            "name": cat_name,
            "description": skills[0].description if len(skills) == 1 else f"Skills related to {cat_name}",
            "skills": skill_entries,
        }

    now = datetime.now().isoformat()
    return {
        "categories": categories,
        "version": 1,
        "created_at": handbook.created_at or now,
        "updated_at": handbook.updated_at or now,
    }


def _convert_model_profile(
    profile: AgentProfile,
    all_skill_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Convert an AgentProfile to RSL model_profile format.

    Includes ALL skills in skill_scores (0.0 when no competence data),
    so the router can use skills even when all models have 0 for that skill.
    """
    skill_scores: Dict[str, float] = {}
    skill_attempts: Dict[str, int] = {}
    skill_successes: Dict[str, int] = {}

    for skill_id, bc in profile.skill_competence.items():
        attempts = bc.total_observations
        successes = max(0, int(bc.alpha - 1))
        skill_attempts[skill_id] = attempts
        skill_successes[skill_id] = successes
        # Use empirical success rate (0 successes → 0 score), not Beta posterior mean
        skill_scores[skill_id] = round(successes / attempts, 4) if attempts > 0 else 0.0

    # Include skills with no observations as 0.0 (all models 0 still matter)
    if all_skill_ids:
        for skill_id in all_skill_ids:
            if skill_id not in skill_scores:
                skill_scores[skill_id] = 0.0
                skill_attempts[skill_id] = 0
                skill_successes[skill_id] = 0

    # Use trajectory-level when available, else skill-level sums
    if profile.total_attempts > 0:
        total_attempts = profile.total_attempts
        total_successes = profile.total_successes
    else:
        total_attempts = sum(skill_attempts.values())
        total_successes = sum(skill_successes.values())

    cs = profile.cost_stats
    n = cs.total_executions or 1
    return {
        "model_name": profile.agent_id,
        "skill_scores": skill_scores,
        "skill_attempts": skill_attempts,
        "skill_successes": skill_successes,
        "strengths": profile.strengths[:5],
        "weaknesses": profile.weaknesses[:5],
        "routing_rules": profile.routing_signals[:5],
        "total_attempts": total_attempts,
        "total_successes": total_successes,
        "total_executions": n,  # actual model calls (for correct avg cost)
        # Cost fields matching AR's ModelSkillProfile schema
        "total_completion_tokens": int(cs.avg_completion_tokens * n),
        "total_prompt_tokens": int(cs.avg_prompt_tokens * n),
        "total_cost_usd": cs.avg_cost_usd * n,
        "total_completion_cost_usd": cs.avg_completion_cost_usd * n,
        "updated_at": datetime.now().isoformat(),
        "recent_success_summaries": [],
        "recent_failure_summaries": [],
    }


def convert_handbook(handbook: SkillHandbook) -> Dict[str, Any]:
    """Convert a SkillHandbook to RSL handbook-compatible dict."""
    total_exp = sum(
        p.cost_stats.total_executions for p in handbook.agent_profiles.values()
    )
    all_skill_ids = list(handbook.skills.keys())

    model_profiles = {}
    for agent_id, profile in handbook.agent_profiles.items():
        model_profiles[agent_id] = _convert_model_profile(profile, all_skill_ids)

    routing_insights = []
    insight_idx = 0
    for mode, meta in handbook.modes.items():
        for ins in meta.insights:
            routing_insights.append({
                "insight_id": f"ins_{insight_idx}",
                "content": ins.content,
                "applies_to_skills": [],
                "applies_to_models": [],
                "confidence": getattr(ins, "confidence", 0.5),
                "supporting_evidence": 1,
            })
            insight_idx += 1
    for agent_id, profile in handbook.agent_profiles.items():
        for signal in profile.routing_signals:
            routing_insights.append({
                "insight_id": f"ins_{insight_idx}",
                "content": signal,
                "applies_to_skills": [],
                "applies_to_models": [agent_id],
                "confidence": 0.5,
                "supporting_evidence": 1,
            })
            insight_idx += 1

    return {
        "version": 0,
        "total_experiences": total_exp,
        "created_at": handbook.created_at,
        "updated_at": handbook.updated_at,
        "skill_catalog": _convert_skill_catalog(handbook),
        "model_profiles": model_profiles,
        "routing_insights": routing_insights,
    }


def save_as_rsl(handbook: SkillHandbook, path: str | Path) -> Path:
    """Convert and save a SkillHandbook as RSL handbook JSON.

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

    n_categories = len(data["skill_catalog"].get("categories", {}))
    n_skills = sum(
        len(cat["skills"])
        for cat in data["skill_catalog"]["categories"].values()
    )
    n_models = len(data["model_profiles"])
    logger.info(
        f"Saved RSL handbook ({n_categories} categories, {n_skills} skills, "
        f"{n_models} models) to {path}"
    )
    return path
