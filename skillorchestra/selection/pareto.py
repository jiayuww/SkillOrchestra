"""
Pareto-optimal handbook selection.

For a target orchestrator O, evaluate candidate handbooks on a validation
set and select the one on the Pareto frontier of (accuracy, cost).

Supports two evaluation modes:
1. Oracle (offline): check if handbook routing would pick a successful agent
   from the exploration traces. Fast and free.
2. Live: run the real orchestrator with eval scripts and measure accuracy
   + cost. Expensive but realistic.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..core.handbook import SkillHandbook
from ..core.traces import ExplorationBundle
from ..core.types import Skill
from .candidates import CandidateHandbook

logger = logging.getLogger(__name__)


def _identify_skills_by_indicators(
    query: str, skills: List[Skill],
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


@dataclass
class EvaluationResult:
    """Evaluation of one candidate handbook on the validation set."""

    name: str
    accuracy: float
    avg_cost: float
    num_skills: int
    granularity: str
    score: float = 0.0  # accuracy - lambda * cost
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "accuracy": self.accuracy,
            "avg_cost": self.avg_cost,
            "num_skills": self.num_skills,
            "granularity": self.granularity,
            "score": self.score,
            "details": self.details,
        }


# ---------------------------------------------------------------------------
# Oracle evaluation (offline, on exploration traces)
# ---------------------------------------------------------------------------

def evaluate_candidate_oracle(
    candidate: CandidateHandbook,
    val_bundles: List[ExplorationBundle],
    lambda_cost: float = 0.0,
) -> EvaluationResult:

    handbook = candidate.handbook
    correct = 0
    total = 0
    costs: List[float] = []

    for bundle in val_bundles:
        query_correct = False

        for mode in bundle.get_modes_explored():
            if mode == "reference":
                continue
            agents = bundle.get_agents_for_mode(mode)
            if not agents:
                continue

            mode_skills = handbook.get_skills_for_mode(mode)
            if not mode_skills:
                continue
            skill_weights = _identify_skills_by_indicators(
                bundle.query, mode_skills,
            )

            selected = handbook.select_agent(mode, skill_weights, lambda_c=lambda_cost)
            if selected and agents.get(selected, False):
                query_correct = True

        if query_correct:
            correct += 1

        for trace in bundle.trajectories:
            if trace.varied_mode != "reference":
                costs.append(trace.total_cost_usd)

        total += 1

    accuracy = correct / total if total > 0 else 0.0
    avg_cost = sum(costs) / len(costs) if costs else 0.0
    score = accuracy - lambda_cost * avg_cost

    return EvaluationResult(
        name=candidate.name,
        accuracy=accuracy,
        avg_cost=avg_cost,
        num_skills=handbook.num_skills,
        granularity=candidate.granularity,
        score=score,
    )


def select_pareto_optimal(
    candidates: List[CandidateHandbook],
    val_bundles: List[ExplorationBundle],
    lambda_cost: float = 0.0,
) -> Tuple[CandidateHandbook, List[EvaluationResult]]:

    results = []
    for candidate in candidates:
        result = evaluate_candidate_oracle(candidate, val_bundles, lambda_cost)
        results.append(result)
        logger.info(
            f"  {result.name}: acc={result.accuracy:.1%}, "
            f"cost=${result.avg_cost:.4f}, skills={result.num_skills}, "
            f"score={result.score:.4f}"
        )

    results.sort(key=lambda r: r.score, reverse=True)
    best = results[0]

    best_candidate = next(c for c in candidates if c.name == best.name)
    logger.info(
        f"Selected: {best.name} (acc={best.accuracy:.1%}, "
        f"score={best.score:.4f})"
    )

    return best_candidate, results


# ---------------------------------------------------------------------------
# Live evaluation (real orchestrator runs)
# ---------------------------------------------------------------------------

def select_pareto_optimal_live(
    candidates: List[CandidateHandbook],
    live_results: List[EvaluationResult],
    lambda_cost: float = 0.0,
) -> Tuple[CandidateHandbook, List[EvaluationResult]]:
    """Select the Pareto-optimal handbook from live evaluation results."""
    
    for r in live_results:
        r.score = r.accuracy - lambda_cost * r.avg_cost

    scored = sorted(live_results, key=lambda r: r.score, reverse=True)
    best_name = scored[0].name

    candidate_map = {c.name: c for c in candidates}
    if best_name not in candidate_map:
        raise ValueError(f"Best candidate '{best_name}' not found in candidates list")

    best_candidate = candidate_map[best_name]
    logger.info(
        f"Selected (live): {best_name} (acc={scored[0].accuracy:.1%}, "
        f"cost=${scored[0].avg_cost:.4f}, score={scored[0].score:.4f})"
    )

    return best_candidate, scored


# ---------------------------------------------------------------------------
# Pareto frontier (works for both oracle and live results)
# ---------------------------------------------------------------------------

def find_pareto_frontier(
    results: List[EvaluationResult],
) -> List[EvaluationResult]:
    """Find points on the Pareto frontier (accuracy vs cost).

    A result is Pareto-optimal if no other result has both higher
    accuracy AND lower cost.
    """
    sorted_results = sorted(results, key=lambda r: r.avg_cost)
    frontier = []
    best_accuracy = -1.0

    for r in sorted_results:
        if r.accuracy > best_accuracy:
            frontier.append(r)
            best_accuracy = r.accuracy

    return frontier


def compare_results(
    oracle_results: List[EvaluationResult],
    live_results: List[EvaluationResult],
) -> List[Dict[str, Any]]:
    """Compare oracle vs live results for the same candidates."""
    oracle_map = {r.name: r for r in oracle_results}
    live_map = {r.name: r for r in live_results}

    comparison = []
    for name in sorted(set(oracle_map.keys()) | set(live_map.keys())):
        oracle = oracle_map.get(name)
        live = live_map.get(name)
        comparison.append({
            "name": name,
            "oracle_accuracy": oracle.accuracy if oracle else None,
            "live_accuracy": live.accuracy if live else None,
            "oracle_cost": oracle.avg_cost if oracle else None,
            "live_cost": live.avg_cost if live else None,
            "accuracy_diff": (
                (live.accuracy - oracle.accuracy)
                if oracle and live
                else None
            ),
        })

    return comparison
