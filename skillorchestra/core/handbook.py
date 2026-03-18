"""
Skill Handbook: stores mode-level routing metadata, skill registry, and agent profiles.
"""

from __future__ import annotations

import copy
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from .types import (
    AgentProfile,
    BetaCompetence,
    CostStats,
    ModeMetadata,
    RoutingInsight,
    Skill,
)

logger = logging.getLogger(__name__)


class SkillHandbook:
    """Skill Handbook: stores mode-level metadata, skill registry, and agent profiles."""
    def __init__(self) -> None:
        self.modes: Dict[str, ModeMetadata] = {}

        self.skills: Dict[str, Skill] = {}

        self.agent_profiles: Dict[str, AgentProfile] = {}

        self.mode_skill_index: Dict[str, Set[str]] = {}

        self.version: str = "v0"
        self.created_at: str = datetime.now().isoformat()
        self.updated_at: str = datetime.now().isoformat()
        self.learning_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Mode operations
    # ------------------------------------------------------------------

    def add_mode(self, mode: str, description: str = "") -> ModeMetadata:
        """Register an operational mode."""
        if mode not in self.modes:
            self.modes[mode] = ModeMetadata(mode=mode, description=description)
        if mode not in self.mode_skill_index:
            self.mode_skill_index[mode] = set()
        return self.modes[mode]

    def get_mode_metadata(self, mode: str) -> Optional[ModeMetadata]:
        return self.modes.get(mode)

    def add_mode_insight(self, mode: str, insight: RoutingInsight) -> None:
        """Add a routing insight to a mode."""
        meta = self.modes.get(mode)
        if meta is None:
            meta = self.add_mode(mode)
        meta.add_insight(insight)
        self._touch()

    # ------------------------------------------------------------------
    # Skill operations
    # ------------------------------------------------------------------

    def add_skill(self, skill: Skill) -> None:
        """Add a skill to the registry and link it to its mode."""
        if not skill.skill_id:
            raise ValueError("Skill must have a skill_id")
        if not skill.mode:
            raise ValueError(f"Skill {skill.skill_id} must have a mode")

        self.skills[skill.skill_id] = skill

        if skill.mode not in self.mode_skill_index:
            self.mode_skill_index[skill.mode] = set()
        self.mode_skill_index[skill.mode].add(skill.skill_id)
        self._touch()

    def remove_skill(self, skill_id: str) -> Optional[Skill]:
        """Remove a skill and its mode-skill edge."""
        skill = self.skills.pop(skill_id, None)
        if skill:
            mode_skills = self.mode_skill_index.get(skill.mode, set())
            mode_skills.discard(skill_id)
            self._touch()
        return skill

    def get_skill(self, skill_id: str) -> Optional[Skill]:
        return self.skills.get(skill_id)

    def get_skills_for_mode(self, mode: str) -> List[Skill]:
        """Get all skills associated with a mode (follows edges E)."""
        skill_ids = self.mode_skill_index.get(mode, set())
        return [self.skills[sid] for sid in skill_ids if sid in self.skills]

    def get_leaf_skills_for_mode(self, mode: str) -> List[Skill]:
        """Get only leaf skills (no children) for a mode."""
        return [s for s in self.get_skills_for_mode(mode) if s.is_leaf(self.skills)]

    def get_skills_at_depth(self, mode: str, max_depth: int) -> List[Skill]:
        """Get skills up to a certain depth in the hierarchy.

        depth 0 = root skills only, depth 1 = roots + their children, etc.
        Useful for generating candidate handbooks at different granularity.
        """
        mode_skills = self.get_skills_for_mode(mode)
        roots = [s for s in mode_skills if s.parent_skill_id is None]

        if max_depth == 0:
            return roots

        result = list(roots)
        frontier = list(roots)
        for _ in range(max_depth):
            next_frontier = []
            for parent in frontier:
                children = parent.get_children(self.skills)
                result.extend(children)
                next_frontier.extend(children)
            frontier = next_frontier

        return result

    def get_skills_at_category_depth(self, mode: str, max_depth: int) -> List[Skill]:
        """Get skills by category granularity when hierarchy is flat.

        DEPRECATED: Use get_skills_at_path_depth for tree-based depth from dots.
        Kept for backward compatibility.
        """
        return self.get_skills_at_path_depth(mode, max_depth)

    def _path_within_mode(self, skill_id: str, mode: str) -> str:
        """Path within mode subtree (strip mode prefix if present). Mode is root."""
        prefix = mode + "."
        if skill_id.startswith(prefix):
            return skill_id[len(prefix) :]
        return skill_id

    def _path_depth(self, skill_id: str, mode: str) -> int:
        """Depth relative to mode root = segments in path within mode."""
        path = self._path_within_mode(skill_id, mode)
        if not path:
            return 0
        return path.count(".") + 1

    def get_skills_at_path_depth(self, mode: str, max_depth: int) -> List[Skill]:
        """Get skills by tree depth. Mode is the root.

        Tree: mode (root) -> category (depth 1) -> category.skill (depth 2) -> ...
        Depth = path segments within mode (strip mode prefix from skill_id first).

        depth 0 = no skills, use overall mode-level performance
        depth 1 = category level (1 skill per category as representative)
        depth 2 = category.skill (all leaf skills)
        depth 3 = category.sub.skill, etc.
        """
        mode_skills = self.get_skills_for_mode(mode)
        if not mode_skills:
            return []

        if max_depth == 0:
            # Depth 0: mode level
            return []

        if max_depth == 1:
            by_cat: Dict[str, List[Skill]] = {}
            for s in mode_skills:
                path = self._path_within_mode(s.skill_id, mode)
                cat = path.split(".", 1)[0] if "." in path else path
                by_cat.setdefault(cat, []).append(s)
            for cat in by_cat:
                by_cat[cat] = sorted(by_cat[cat], key=lambda x: x.skill_id)
            return [by_cat[cat][0] for cat in sorted(by_cat.keys())]

        # Depth >= 2: include all skills with path_depth <= max_depth
        return [s for s in mode_skills if self._path_depth(s.skill_id, mode) <= max_depth]

    def max_path_depth(self, mode: str) -> int:
        """Max tree depth for a mode (path segments within mode)."""
        mode_skills = self.get_skills_for_mode(mode)
        if not mode_skills:
            return 0
        return max(self._path_depth(s.skill_id, mode) for s in mode_skills)

    @property
    def num_skills(self) -> int:
        return len(self.skills)

    @property
    def all_modes(self) -> List[str]:
        return sorted(self.modes.keys())

    # ------------------------------------------------------------------
    # Agent profile operations
    # ------------------------------------------------------------------

    def add_agent_profile(self, profile: AgentProfile) -> None:
        """Add or replace an agent profile."""
        self.agent_profiles[profile.agent_id] = profile
        self._touch()

    def get_agent_profile(self, agent_id: str) -> Optional[AgentProfile]:
        return self.agent_profiles.get(agent_id)

    def get_or_create_agent_profile(
        self, agent_id: str, mode: str, model_name: str = "", tools: Optional[List[str]] = None
    ) -> AgentProfile:
        """Get an existing profile or create a new one."""
        if agent_id not in self.agent_profiles:
            self.agent_profiles[agent_id] = AgentProfile(
                agent_id=agent_id,
                mode=mode,
                model_name=model_name,
                tools=tools or [],
            )
        else:
            if model_name:
                self.agent_profiles[agent_id].model_name = model_name
        return self.agent_profiles[agent_id]

    def get_agents_for_mode(self, mode: str) -> List[AgentProfile]:
        """Get all agent profiles for a specific mode."""
        return [p for p in self.agent_profiles.values() if p.mode == mode]

    def harmonize_agent_skill_sets(self) -> None:
        """Ensure all agents in a mode have the same skill set."""
        for mode in self.all_modes:
            mode_skill_ids = self.mode_skill_index.get(mode, set())
            if not mode_skill_ids:
                continue
            for profile in self.get_agents_for_mode(mode):
                for skill_id in mode_skill_ids:
                    if skill_id not in profile.skill_competence:
                        profile.skill_competence[skill_id] = BetaCompetence()
        self._touch()

    def update_competence(self, agent_id: str, skill_id: str, success: bool) -> None:
        """Update an agent's competence on a skill."""
        profile = self.agent_profiles.get(agent_id)
        if profile:
            profile.update_competence(skill_id, success)
            self._touch()

    def get_competence(self, agent_id: str, skill_id: str) -> float:
        """Get an agent's competence on a skill (posterior mean)."""
        profile = self.agent_profiles.get(agent_id)
        if profile:
            return profile.get_competence(skill_id)
        return 0.5

    # ------------------------------------------------------------------
    # Agent selection
    # ------------------------------------------------------------------

    def select_agent(
        self,
        mode: str,
        skill_weights: Dict[str, float],
        lambda_c: float = 0.0,
    ) -> Optional[str]:
        """Select agent — performance first, then category, then mode, then cost.

        1. Compute weighted competence for each agent.
        2. Find agents tied at the top competence score.
        3. If single winner, return directly.
        4. Among tied agents: go up to category level (e.g. entertainment_knowledge).
        5. If still tied: go up to mode level (overall success rate).
        6. If still tied: cost-penalize, normalize to probs, sample.

        Args:
            mode: operational mode
            skill_weights: {skill_id: weight} for active skills
            lambda_c: cost penalty coefficient

        Returns:
            agent_id of the selected agent, or None if no agents available
        """
        import random

        agents = self.get_agents_for_mode(mode)
        if not agents:
            return None

        # Step 1: compute competence scores
        scored: Dict[str, float] = {}
        cost_map: Dict[str, float] = {}
        for profile in agents:
            scored[profile.agent_id] = profile.weighted_competence(skill_weights)
            cost_map[profile.agent_id] = profile.cost_stats.avg_cost_usd

        # Step 2: find top performers
        best_score = max(scored.values())
        eps = 1e-6
        tied = [aid for aid, s in scored.items() if abs(s - best_score) < eps]

        # Step 3: single winner
        if len(tied) == 1:
            return tied[0]

        profile_map = {p.agent_id: p for p in agents}
        active_skill_ids = list(skill_weights.keys())

        # Step 4: among tied agents, go up to category level
        still_tied = tied
        if active_skill_ids:
            by_category = sorted(
                tied,
                key=lambda aid: profile_map[aid].category_competence_for_skills(active_skill_ids),
                reverse=True,
            )
            best_cat = profile_map[by_category[0]].category_competence_for_skills(active_skill_ids) if by_category else 0.0
            still_tied = [aid for aid in by_category
                          if profile_map[aid].category_competence_for_skills(active_skill_ids) >= best_cat - 1e-9]

        # Step 5: if still tied, go up to mode level (overall success rate)
        if len(still_tied) > 1:
            by_overall = sorted(
                still_tied,
                key=lambda aid: profile_map[aid].overall_success_rate,
                reverse=True,
            )
            best_overall = profile_map[by_overall[0]].overall_success_rate if by_overall else 0.0
            still_tied = [aid for aid in by_overall
                          if profile_map[aid].overall_success_rate >= best_overall - 1e-9]

        # Step 6: if still tied, cost-penalize → normalize → sample
        adjusted: Dict[str, float] = {}
        for aid in still_tied:
            adjusted[aid] = max(0.0, 1.0 - lambda_c * cost_map[aid])

        total = sum(adjusted.values())
        if total <= 0:
            probs = {aid: 1.0 / len(still_tied) for aid in still_tied}
        else:
            probs = {aid: v / total for aid, v in adjusted.items()}

        agent_ids = list(probs.keys())
        weights = [probs[aid] for aid in agent_ids]
        return random.choices(agent_ids, weights=weights, k=1)[0]

    # ------------------------------------------------------------------
    # Oracle check (for learning-phase validation)
    # ------------------------------------------------------------------

    def oracle_check_query(
        self,
        mode: str,
        skill_weights: Dict[str, float],
        agent_outcomes: Dict[str, bool],
        lambda_c: float = 0.0,
    ) -> Dict[str, Any]:
        """Check if the handbook would pick a successful agent.

        Args:
            mode: operational mode
            skill_weights: {skill_id: weight} for active skills
            agent_outcomes: {agent_id: True/False} -- actual outcomes from traces
            lambda_c: cost penalty

        Returns:
            Dict with 'selected_agent', 'selected_success', 'oracle_agent', 'oracle_success'
        """
        selected = self.select_agent(mode, skill_weights, lambda_c)
        oracle_agents = [a for a, s in agent_outcomes.items() if s]
        oracle_agent = oracle_agents[0] if oracle_agents else None

        return {
            "selected_agent": selected,
            "selected_success": agent_outcomes.get(selected, False) if selected else False,
            "oracle_agent": oracle_agent,
            "oracle_success": bool(oracle_agents),
        }

    # ------------------------------------------------------------------
    # Subgraph extraction (for handbook selection)
    # ------------------------------------------------------------------

    def subgraph(
        self,
        skill_ids: Optional[Set[str]] = None,
        modes: Optional[Set[str]] = None,
    ) -> SkillHandbook:
        """Create an induced subgraph (for Pareto-optimal selection).

        Args:
            skill_ids: if provided, only include these skills
            modes: if provided, only include these modes

        Returns:
            A new SkillHandbook containing only the specified subset
        """
        sub = SkillHandbook()
        sub.version = self.version + "-sub"
        sub.created_at = self.created_at
        sub.learning_stats = copy.deepcopy(self.learning_stats)

        target_modes = modes or set(self.modes.keys())
        for m in target_modes:
            if m in self.modes:
                sub.modes[m] = copy.deepcopy(self.modes[m])
                sub.mode_skill_index[m] = set()

        for sid, skill in self.skills.items():
            if skill.mode not in target_modes:
                continue
            if skill_ids is not None and sid not in skill_ids:
                continue
            sub.skills[sid] = copy.deepcopy(skill)
            sub.mode_skill_index[skill.mode].add(sid)

        for aid, profile in self.agent_profiles.items():
            if profile.mode not in target_modes:
                continue
            sub_profile = copy.deepcopy(profile)
            if skill_ids is not None:
                sub_profile.skill_competence = {
                    sid: bc
                    for sid, bc in sub_profile.skill_competence.items()
                    if sid in skill_ids
                }
            sub.agent_profiles[aid] = sub_profile

        sub._touch()
        return sub

    # ------------------------------------------------------------------
    # Prompt rendering (for LLM consumption)
    # ------------------------------------------------------------------

    def as_prompt(self, mode: Optional[str] = None) -> str:
        """Render the handbook as natural language for LLM prompts."""
        parts = []

        target_modes = [mode] if mode else self.all_modes

        for m in target_modes:
            meta = self.modes.get(m)
            if meta:
                parts.append(f"## Mode: {m}")
                if meta.description:
                    parts.append(f"Description: {meta.description}")
                if meta.insights:
                    parts.append("Mode Insights:")
                    for ins in meta.insights:
                        parts.append(f"  - [{ins.insight_type}] {ins.content} (confidence: {ins.confidence:.1f})")
                parts.append("")

            skills = self.get_skills_for_mode(m)
            if skills:
                parts.append(f"### Skills for {m}")
                for skill in skills:
                    parts.append(f"- **{skill.name}** ({skill.skill_id})")
                    parts.append(f"  Description: {skill.description}")
                    if skill.indicators:
                        parts.append(f"  Indicators: {', '.join(skill.indicators)}")
                parts.append("")

            agents = self.get_agents_for_mode(m)
            if agents:
                parts.append(f"### Agent Profiles for {m}")
                for agent in agents:
                    parts.append(f"- **{agent.agent_id}** (model: {agent.model_name})")
                    if agent.skill_competence:
                        competences = []
                        for sid, bc in agent.skill_competence.items():
                            competences.append(f"{sid}: {bc.empirical_rate:.2f}")
                        parts.append(f"  Skill scores: {', '.join(competences)}")
                    if agent.cost_stats.total_executions > 0:
                        parts.append(f"  Avg cost: ${agent.cost_stats.avg_cost_usd:.4f}")
                    if agent.strengths:
                        parts.append(f"  Strengths: {', '.join(agent.strengths)}")
                    if agent.weaknesses:
                        parts.append(f"  Weaknesses: {', '.join(agent.weaknesses)}")
                parts.append("")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Integrity checks
    # ------------------------------------------------------------------

    def validate(self) -> List[str]:
        """Check referential integrity. Returns list of issues."""
        issues = []

        for mode, skill_ids in self.mode_skill_index.items():
            if mode not in self.modes:
                issues.append(f"Mode '{mode}' in index but not in modes registry")
            for sid in skill_ids:
                if sid not in self.skills:
                    issues.append(f"Skill '{sid}' in mode-skill index but not in skill registry")

        for sid, skill in self.skills.items():
            if skill.mode not in self.mode_skill_index:
                issues.append(f"Skill '{sid}' references mode '{skill.mode}' not in index")
            elif sid not in self.mode_skill_index.get(skill.mode, set()):
                issues.append(f"Skill '{sid}' not in mode-skill index for mode '{skill.mode}'")
            if skill.parent_skill_id and skill.parent_skill_id not in self.skills:
                issues.append(f"Skill '{sid}' references parent '{skill.parent_skill_id}' not found")

        for aid, profile in self.agent_profiles.items():
            if profile.mode and profile.mode not in self.modes:
                issues.append(f"Agent '{aid}' references mode '{profile.mode}' not in modes")

        return issues

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save handbook as a single JSON file."""
        self._touch()
        issues = self.validate()
        if issues:
            logger.warning(f"Handbook has {len(issues)} integrity issues: {issues[:3]}")

        data = {
            "metadata": {
                "version": self.version,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
                "num_modes": len(self.modes),
                "num_skills": self.num_skills,
                "num_agents": len(self.agent_profiles),
                "learning_stats": self.learning_stats,
            },
            "modes": {m: meta.to_dict() for m, meta in self.modes.items()},
            "skills": {sid: s.to_dict() for sid, s in self.skills.items()},
            "agent_profiles": {aid: p.to_dict() for aid, p in self.agent_profiles.items()},
            "mode_skill_index": {
                m: sorted(sids) for m, sids in self.mode_skill_index.items()
            },
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved handbook ({self.num_skills} skills, {len(self.agent_profiles)} agents) to {path}")

    @classmethod
    def load(cls, path: str | Path) -> SkillHandbook:
        """Load handbook from a JSON file."""
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        handbook = cls()

        meta = data.get("metadata", {})
        handbook.version = meta.get("version", "v0")
        handbook.created_at = meta.get("created_at", "")
        handbook.updated_at = meta.get("updated_at", "")
        handbook.learning_stats = meta.get("learning_stats", {})

        for m, mdict in data.get("modes", {}).items():
            handbook.modes[m] = ModeMetadata.from_dict(mdict)

        for sid, sdict in data.get("skills", {}).items():
            handbook.skills[sid] = Skill.from_dict(sdict)

        for aid, pdict in data.get("agent_profiles", {}).items():
            handbook.agent_profiles[aid] = AgentProfile.from_dict(pdict)

        for m, sids in data.get("mode_skill_index", {}).items():
            handbook.mode_skill_index[m] = set(sids)

        issues = handbook.validate()
        if issues:
            logger.warning(f"Loaded handbook has {len(issues)} integrity issues")

        logger.info(f"Loaded handbook v{handbook.version} ({handbook.num_skills} skills) from {path}")
        return handbook

    # ------------------------------------------------------------------
    # Summary / inspection
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable summary of the handbook."""
        lines = [
            f"SkillHandbook v{self.version}",
            f"  Modes: {', '.join(self.all_modes)}",
            f"  Total skills: {self.num_skills}",
        ]
        for m in self.all_modes:
            skills = self.get_skills_for_mode(m)
            agents = self.get_agents_for_mode(m)
            lines.append(f"  [{m}] {len(skills)} skills, {len(agents)} agents")
        lines.append(f"  Total agents: {len(self.agent_profiles)}")
        issues = self.validate()
        if issues:
            lines.append(f"  WARNING: {len(issues)} integrity issues")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"SkillHandbook(v={self.version}, skills={self.num_skills}, agents={len(self.agent_profiles)})"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _touch(self) -> None:
        self.updated_at = datetime.now().isoformat()
