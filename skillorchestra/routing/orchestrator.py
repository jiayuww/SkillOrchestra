"""
Deployment-time orchestrator using the Skill Handbook.

At each timestep t:
  1. Mode selection
  2. Identify active skills
  3. Agent selection
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..core.handbook import SkillHandbook
from ..core.types import AgentProfile
from ..llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class RoutingDecision:
    """A single routing decision at timestep t."""
    mode: str
    agent_id: str
    skill_weights: Dict[str, float]
    competence_score: float
    cost_penalty: float
    final_score: float
    reasoning: str = ""


@dataclass
class RoutingContext:
    """Accumulated state during multi-turn orchestration."""
    query: str
    history: List[Dict[str, Any]] = field(default_factory=list)
    decisions: List[RoutingDecision] = field(default_factory=list)
    total_cost: float = 0.0


class SkillOrchestrator:
    """Skill-based agent orchestrator."""

    def __init__(
        self,
        handbook: SkillHandbook,
        llm: Optional[LLMClient] = None,
        lambda_c: float = 0.0,
    ):
        self.handbook = handbook
        self.llm = llm
        self.lambda_c = lambda_c

    def select_agent(
        self,
        mode: str,
        query: str,
        skill_weights: Optional[Dict[str, float]] = None,
    ) -> RoutingDecision:
        if skill_weights is None:
            mode_skills = self.handbook.get_skills_for_mode(mode)
            if mode_skills:
                skill_weights = {s.skill_id: 1.0 / len(mode_skills) for s in mode_skills}
            else:
                skill_weights = {}

        agents = self.handbook.get_agents_for_mode(mode)
        if not agents:
            return RoutingDecision(
                mode=mode,
                agent_id="",
                skill_weights=skill_weights,
                competence_score=0.0,
                cost_penalty=0.0,
                final_score=0.0,
                reasoning="No agents available for this mode",
            )

        selected_id = self.handbook.select_agent(mode, skill_weights, self.lambda_c)
        best_agent = None
        best_competence = 0.0
        best_cost_penalty = 0.0
        best_score = 0.0

        if selected_id:
            best_agent = self.handbook.get_agent_profile(selected_id)
            if best_agent:
                best_competence = best_agent.weighted_competence(skill_weights)
                best_cost_penalty = self.lambda_c * best_agent.cost_stats.avg_cost_usd
                best_score = best_competence - best_cost_penalty

        agent_id = best_agent.agent_id if best_agent else ""

        return RoutingDecision(
            mode=mode,
            agent_id=agent_id,
            skill_weights=skill_weights,
            competence_score=best_competence,
            cost_penalty=best_cost_penalty,
            final_score=best_score,
            reasoning=self._build_reasoning(best_agent, skill_weights),
        )

    def get_handbook_context(self, mode: Optional[str] = None) -> str:
        return self.handbook.as_prompt(mode)

    def _build_reasoning(
        self,
        agent: Optional[AgentProfile],
        skill_weights: Dict[str, float],
    ) -> str:
        if not agent or not skill_weights:
            return ""
        parts = [f"Selected {agent.agent_id} (model: {agent.model_name})"]
        for sid, w in sorted(skill_weights.items(), key=lambda x: -x[1]):
            score = agent.get_competence(sid)
            parts.append(f"  {sid}: weight={w:.2f}, competence={score:.2f}")
        return "\n".join(parts)
