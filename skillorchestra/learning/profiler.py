"""
Phase 1b: Agent Profile Construction.

Updates Beta(alpha, beta) competence estimates from execution traces
and distills mode-level routing insights.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from config import resolve_model
from ..core.handbook import SkillHandbook
from ..core.traces import ExplorationBundle, ExecutionTrace
from ..core.types import AgentProfile, CostStats, ModeMetadata, RoutingInsight
from ..llm.client import LLMClient
from .prompts import SKILL_IDENTIFICATION_PROMPT, MODE_INSIGHT_PROMPT, PROFILE_SUMMARY_PROMPT

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------

class ActiveSkill(BaseModel):
    skill_id: str
    weight: float = 1.0
    reasoning: str = ""


class SkillIdentificationOutput(BaseModel):
    active_skills: List[ActiveSkill] = Field(default_factory=list)


class InsightItem(BaseModel):
    mode: str
    content: str
    insight_type: str = "usage"
    confidence: float = 0.5


class ModeInsightOutput(BaseModel):
    insights: List[InsightItem] = Field(default_factory=list)


class ProfileSummaryOutput(BaseModel):
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    routing_signals: List[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Profiler
# ---------------------------------------------------------------------------

class AgentProfiler:
    """Constructs agent profiles from exploration traces.

    For each (agent, mode, skill) triple, updates:
        alpha += I[agent succeeds on skill]
        beta  += I[agent fails on skill]

    Also distills mode-level routing insights and agent summaries.
    """

    def __init__(
        self,
        llm: LLMClient,
        use_llm_for_skill_id: bool = True,
        llm_skill_id: Optional[LLMClient] = None,
        skill_id_traces_path: Optional[str] = None,
    ):
        self.llm = llm
        self.llm_skill_id = llm_skill_id or llm  # Skill identification uses dedicated LLM when provided
        self.use_llm_for_skill_id = use_llm_for_skill_id
        self.skill_id_traces_path = skill_id_traces_path

    def build_profiles(
        self,
        bundles: List[ExplorationBundle],
        handbook: SkillHandbook,
    ) -> Dict[str, Any]:
        """Build agent profiles from exploration bundles.

        For each bundle (query):
        1. Identify active skills
        2. For each trajectory: update ALL identified skills with success/fail
        3. Update cost statistics
        """
        stats = {"queries_processed": 0, "updates": 0, "skill_id_calls": 0}
        total_bundles = len(bundles)

        for i, bundle in enumerate(bundles):
            modes_seen: dict = {}
            for trace in bundle.trajectories:
                if trace.varied_mode and trace.varied_mode != "reference":
                    modes_seen[trace.varied_mode] = True

            for mode in modes_seen:
                active_skills = self._identify_active_skills(
                    bundle, mode, handbook
                )
                if active_skills:
                    stats["skill_id_calls"] += 1

                for trace in bundle.trajectories:
                    if trace.varied_mode != mode:
                        continue

                    agent_id = trace.varied_agent_id
                    model_name = resolve_model(agent_id)
                    profile = handbook.get_or_create_agent_profile(
                        agent_id=agent_id,
                        mode=mode,
                        model_name=model_name,
                    )

                    # Trajectory-level tracking
                    profile.total_attempts += 1
                    if trace.task_success:
                        profile.total_successes += 1

                    for skill_id, weight in active_skills.items():
                        profile.update_competence(skill_id, trace.task_success)
                        stats["updates"] += 1

                    self._update_cost_stats(profile, trace)

            stats["queries_processed"] += 1
            if (i + 1) % 5 == 0 or (i + 1) == total_bundles:
                logger.info(
                    f"  Profiling progress: {i+1}/{total_bundles} bundles, "
                    f"{stats['updates']} updates, {stats['skill_id_calls']} LLM calls"
                )

        handbook.harmonize_agent_skill_sets()
        logger.info(
            f"Profiling complete: {stats['queries_processed']} queries, "
            f"{stats['updates']} competence updates"
        )
        return stats

    def _identify_active_skills(
        self,
        bundle: ExplorationBundle,
        mode: str,
        handbook: SkillHandbook,
    ) -> Dict[str, float]:
        mode_skills = handbook.get_skills_for_mode(mode)
        if not mode_skills:
            return {}

        if self.use_llm_for_skill_id and len(mode_skills) > 1:
            return self._identify_skills_with_llm(bundle, mode, mode_skills)

        return self._identify_skills_uniform(mode_skills)

    def _format_model_results(
        self, bundle: ExplorationBundle, mode: str, max_output_chars: int = 800
    ) -> str:
        traces = bundle.get_trajectories_for_mode(mode)
        if not traces:
            return "(no traces for this mode)"

        parts = []
        for trace in traces:
            label = "CORRECT" if trace.task_success else "WRONG"
            output = ""
            relevant_steps = trace.get_steps_for_mode(mode)
            if relevant_steps:
                output = relevant_steps[0].output_text
            if not output and trace.final_answer:
                output = trace.final_answer
            output = (output or "(no output)")[:max_output_chars]
            parts.append(f"- {trace.varied_agent_id} [{label}]: {output}")

        return "\n".join(parts)

    def _identify_skills_uniform(
        self, mode_skills: list
    ) -> Dict[str, float]:
        """Fallback: assign uniform weight to all skills in the mode. This should be used rarely."""
        weight = 1.0 / len(mode_skills) if mode_skills else 0.0
        return {s.skill_id: weight for s in mode_skills}

    def _identify_skills_with_llm(
        self,
        bundle: ExplorationBundle,
        mode: str,
        mode_skills: list,
    ) -> Dict[str, float]:
        """Use LLM to identify which skills are active."""
        skills_text = "\n".join(
            f"- {s.skill_id}: {s.description}" for s in mode_skills
        )

        model_results_text = self._format_model_results(bundle, mode)
        ground_truth = ", ".join(bundle.ground_truths[:3]) if bundle.ground_truths else "(none)"

        prompt = SKILL_IDENTIFICATION_PROMPT.format(
            query=bundle.query[:500],
            ground_truth=ground_truth[:200],
            mode=mode,
            mode_skills=skills_text,
            model_results=model_results_text,
        )

        self.llm_skill_id.set_role("skill_identifier")
        try:
            output = self.llm_skill_id.complete_structured(
                prompt=prompt,
                response_model=SkillIdentificationOutput,
            )
            result = {}
            for item in output.active_skills:
                if item.skill_id in {s.skill_id for s in mode_skills}:
                    result[item.skill_id] = item.weight
            if result:
                total = sum(result.values())
                if total > 0:
                    result = {k: v / total for k, v in result.items()}
            else:
                result = self._identify_skills_uniform(mode_skills)

            if self.skill_id_traces_path:
                llm_raw = [{"skill_id": s.skill_id, "weight": s.weight} for s in output.active_skills]
                self._store_skill_id_trace(
                    bundle=bundle,
                    mode=mode,
                    active_skills=result,
                    llm_raw=llm_raw,
                )
            return result
        except Exception as exc:
            logger.warning(f"LLM skill identification failed: {exc}")

        return self._identify_skills_uniform(mode_skills)

    def _store_skill_id_trace(
        self,
        bundle: ExplorationBundle,
        mode: str,
        active_skills: Dict[str, float],
        llm_raw: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Append raw trace for skill identification to JSONL for labeling audit."""
        traces = bundle.get_trajectories_for_mode(mode)
        model_results = []
        trace_outcomes = {}
        for trace in traces:
            output = ""
            relevant_steps = trace.get_steps_for_mode(mode)
            if relevant_steps:
                output = relevant_steps[0].output_text
            if not output and trace.final_answer:
                output = trace.final_answer
            model_results.append({
                "agent_id": trace.varied_agent_id,
                "success": trace.task_success,
                "output": (output or "(no output)")[:800],
            })
            trace_outcomes[trace.varied_agent_id] = trace.task_success

        record = {
            "query_id": bundle.query_id,
            "query": bundle.query[:500],
            "ground_truths": bundle.ground_truths[:5],
            "mode": mode,
            "model_results": model_results,
            "trace_outcomes": trace_outcomes,
            "active_skills": active_skills,
        }
        if llm_raw is not None:
            record["llm_raw"] = llm_raw
        path = Path(self.skill_id_traces_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def _update_cost_stats(
        self, profile: AgentProfile, trace: ExecutionTrace
    ) -> None:
        """Update cost statistics from a trace. Includes prompt and completion costs."""
        from ..converters.from_ar import API_PRICE_1M_TOKENS

        total_prompt = sum(s.prompt_tokens for s in trace.steps)
        total_completion = sum(s.completion_tokens for s in trace.steps)
        total_latency = sum(s.latency_s for s in trace.steps)
        total_cost = trace.total_cost_usd

        model = trace.varied_agent_id or profile.agent_id
        prices = API_PRICE_1M_TOKENS.get(model)
        if prices and (total_prompt > 0 or total_completion > 0):
            prompt_cost = total_prompt * prices["input"] / 1_000_000
            completion_cost = total_completion * prices["output"] / 1_000_000
        else:
            prompt_cost = 0.0
            completion_cost = total_cost

        profile.cost_stats.update(
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            latency_s=total_latency,
            cost_usd=total_cost,
            completion_cost_usd=completion_cost,
            prompt_cost_usd=prompt_cost,
        )

    # ------------------------------------------------------------------
    # Mode-level insight distillation
    # ------------------------------------------------------------------

    def distill_mode_insights(
        self,
        bundles: List[ExplorationBundle],
        handbook: SkillHandbook,
    ) -> List[RoutingInsight]:
        """Distill mode-level routing insights from execution patterns."""
        logger.info(f"  Distilling insights from {len(bundles)} bundles...")
        patterns = self._collect_execution_patterns(bundles)
        if not patterns:
            logger.info("  No execution patterns found, skipping insight distillation")
            return []

        logger.info(f"  Patterns:\n    {patterns.replace(chr(10), chr(10) + '    ')}")
        modes_text = ", ".join(handbook.all_modes)
        prompt = MODE_INSIGHT_PROMPT.format(
            execution_patterns=patterns,
            modes=modes_text,
        )

        self.llm.set_role("insight_distiller")
        try:
            output = self.llm.complete_structured(
                prompt=prompt,
                response_model=ModeInsightOutput,
            )
        except Exception as exc:
            logger.error(f"Mode insight distillation failed: {exc}")
            return []

        new_insights = []
        for item in output.insights:
            insight = RoutingInsight(
                content=item.content,
                insight_type=item.insight_type,
                confidence=item.confidence,
            )
            handbook.add_mode_insight(item.mode, insight)
            new_insights.append(insight)

        logger.info(f"Distilled {len(new_insights)} mode-level insights")
        return new_insights

    def _collect_execution_patterns(
        self, bundles: List[ExplorationBundle]
    ) -> str:
        """Collect execution patterns for insight distillation."""
        mode_success_rates: Dict[str, Dict[str, int]] = {}
        mode_costs: Dict[str, List[float]] = {}

        for bundle in bundles:
            for trace in bundle.trajectories:
                if trace.varied_mode == "reference":
                    continue
                mode = trace.varied_mode
                if mode not in mode_success_rates:
                    mode_success_rates[mode] = {"success": 0, "total": 0}
                mode_success_rates[mode]["total"] += 1
                if trace.task_success:
                    mode_success_rates[mode]["success"] += 1
                if mode not in mode_costs:
                    mode_costs[mode] = []
                mode_costs[mode].append(trace.total_cost_usd)

        parts = []
        for mode, rates in sorted(mode_success_rates.items()):
            rate = rates["success"] / rates["total"] if rates["total"] else 0
            avg_cost = sum(mode_costs.get(mode, [0])) / max(len(mode_costs.get(mode, [1])), 1)
            parts.append(
                f"Mode {mode}: {rate:.1%} success rate, "
                f"${avg_cost:.4f} avg cost ({rates['total']} samples)"
            )

        return "\n".join(parts) if parts else ""

    # ------------------------------------------------------------------
    # Agent profile summarization
    # ------------------------------------------------------------------

    def summarize_profiles(self, handbook: SkillHandbook) -> None:
        """Generate natural-language summaries for agent profiles."""
        total = len(handbook.agent_profiles)
        logger.info(f"  Summarizing {total} agent profiles...")
        for idx, (agent_id, profile) in enumerate(handbook.agent_profiles.items()):
            if profile.strengths:
                logger.info(f"    [{idx+1}/{total}] {agent_id}: already summarized, skip")
                continue
            logger.info(f"    [{idx+1}/{total}] {agent_id} ({profile.mode})...")

            perf_data = self._format_performance_data(profile, handbook)
            if not perf_data:
                continue

            prompt = PROFILE_SUMMARY_PROMPT.format(
                agent_id=agent_id,
                model_name=profile.model_name,
                mode=profile.mode,
                performance_data=perf_data,
            )

            self.llm.set_role("profile_summarizer")
            try:
                output = self.llm.complete_structured(
                    prompt=prompt,
                    response_model=ProfileSummaryOutput,
                )
                profile.strengths = output.strengths
                profile.weaknesses = output.weaknesses
                profile.routing_signals = output.routing_signals
            except Exception as exc:
                logger.warning(f"Profile summary failed for {agent_id}: {exc}")

    def _format_performance_data(
        self, profile: AgentProfile, handbook: SkillHandbook
    ) -> str:
        """Format performance data for the summarization prompt."""
        if not profile.skill_competence:
            return ""

        parts = []
        for skill_id, bc in profile.skill_competence.items():
            skill = handbook.get_skill(skill_id)
            skill_name = skill.name if skill else skill_id
            parts.append(
                f"- {skill_name}: {bc.empirical_rate:.1%} success "
                f"({int(bc.alpha - 1)} wins, {int(bc.beta - 1)} losses)"
            )

        if profile.cost_stats.total_executions > 0:
            parts.append(f"- Avg cost: ${profile.cost_stats.avg_cost_usd:.4f}")
            parts.append(f"- Total executions: {profile.cost_stats.total_executions}")

        return "\n".join(parts)
