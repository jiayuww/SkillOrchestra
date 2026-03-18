"""
Convert RSL (Router Skill Learner) inference results into SkillOrchestra format.

The RSL produces inference_results.jsonl with one line per query, containing
all pool models' responses and success flags.
  - There is a single mode: "search" (router searches for answer by calling pool models)
  - Each model is an "agent"
  - Each query produces one trajectory per model

RSL inference_results.jsonl fields:
    sample_id, question, ground_truths,
    model_succeeded: {model_key: bool},
    model_exact_match: {model_key: float},
    model_f1: {model_key: float},
    model_responses: {model_key: parsed_answer},
    model_raw_responses: {model_key: full_response},
    pool_cost, pool_prompt_tokens, pool_completion_tokens,
    best_model, best_score, best_answer
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.traces import ExecutionStep, ExecutionTrace, ExplorationBundle

logger = logging.getLogger(__name__)

ROUTING_MODE = "search"

from config.pool import API_PRICE_1M_TOKENS


def _calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate cost in USD given token counts and model pricing."""
    prices = API_PRICE_1M_TOKENS.get(model)
    if not prices:
        return 0.0
    return (
        prompt_tokens * prices["input"] / 1_000_000
        + completion_tokens * prices["output"] / 1_000_000
    )


def _parse_rsl_line(data: Dict[str, Any]) -> Optional[ExplorationBundle]:
    """Parse one line of inference_results.jsonl into an ExplorationBundle."""
    sample_id = data.get("sample_id", "")
    question = data.get("question", "")
    ground_truths = data.get("ground_truths", [])

    # Support evaluate.py format: model_results with nested {response, exact_match, ...}
    model_results = data.get("model_results", {})
    if model_results:
        model_succeeded = {mk: mr.get("exact_match", 0) >= 0.7 for mk, mr in model_results.items()}
        model_responses = {mk: mr.get("response", "") for mk, mr in model_results.items()}
        model_em = {mk: mr.get("exact_match", 0.0) for mk, mr in model_results.items()}
        model_costs_dict = {
            mk: {
                "prompt_tokens": mr.get("prompt_tokens", 0),
                "completion_tokens": mr.get("completion_tokens", 0),
                "total_cost": mr.get("total_cost", 0.0),
            }
            for mk, mr in model_results.items()
        }
        model_raw = {}
        pool_prompt = sum(mr.get("prompt_tokens", 0) for mr in model_results.values())
        pool_completion = sum(mr.get("completion_tokens", 0) for mr in model_results.values())
    else:
        model_succeeded = data.get("model_succeeded", {})
        model_responses = data.get("model_responses", {})
        model_raw = data.get("model_raw_responses", {})
        model_em = data.get("model_exact_match", {})
        model_costs_dict = data.get("model_costs", {})
        pool_prompt = data.get("pool_prompt_tokens", 0)
        pool_completion = data.get("pool_completion_tokens", 0)

    if not model_succeeded:
        return None

    n_models = len(model_succeeded)

    # If per-model costs are available (from our explore.py or evaluate.py), use them directly.
    # Otherwise estimate from aggregate totals (AR RSL format).
    has_per_model_costs = bool(model_costs_dict)

    if not has_per_model_costs:
        per_model_prompt = pool_prompt // max(n_models, 1)
        model_comp_tokens: Dict[str, int] = {}
        for model_key in model_succeeded:
            raw_text = model_raw.get(model_key, "")
            model_comp_tokens[model_key] = len(str(raw_text)) // 4 if raw_text else 0
        est_total_comp = sum(model_comp_tokens.values()) or 1
        for mk in model_comp_tokens:
            model_comp_tokens[mk] = int(
                model_comp_tokens[mk] * pool_completion / est_total_comp
            ) if est_total_comp > 0 else 0

    trajectories = []
    for model_key, success in model_succeeded.items():
        answer_text = model_responses.get(model_key, "")
        if isinstance(answer_text, list):
            answer_text = str(answer_text)

        if has_per_model_costs and model_key in model_costs_dict:
            mc = model_costs_dict[model_key]
            est_prompt = mc.get("prompt_tokens", 0)
            est_completion = mc.get("completion_tokens", 0)
            model_cost = mc.get("total_cost", _calculate_cost(model_key, est_prompt, est_completion))
        else:
            est_prompt = per_model_prompt
            est_completion = model_comp_tokens.get(model_key, 0)
            model_cost = _calculate_cost(model_key, est_prompt, est_completion)

        step = ExecutionStep(
            step_idx=0,
            mode=ROUTING_MODE,
            agent_id=model_key,
            model_name=model_key,
            output_text=str(answer_text)[:2000],
            cost_usd=model_cost,
            prompt_tokens=est_prompt,
            completion_tokens=est_completion,
        )

        trace = ExecutionTrace(
            query_id=sample_id,
            query=question,
            ground_truths=ground_truths,
            steps=[step],
            final_answer=str(answer_text),
            task_success=bool(success),
            total_cost_usd=model_cost,
            varied_mode=ROUTING_MODE,
            varied_agent_id=model_key,
            metadata={
                "exact_match": model_em.get(model_key, 0.0),
            },
        )
        trajectories.append(trace)

    return ExplorationBundle(
        query_id=sample_id,
        query=question,
        ground_truths=ground_truths,
        trajectories=trajectories,
    )


def load_rsl_results(
    results_path: str | Path,
    max_samples: Optional[int] = None,
) -> List[ExplorationBundle]:
    """Load RSL inference_results.jsonl as ExplorationBundles.

    Each line in the JSONL contains results for one query across all pool models.
    This converts to one ExplorationBundle per query.

    Note: RSL results may contain multiple epochs (3 lines per sample for
    epochs=3). We deduplicate by sample_id, keeping the last occurrence.

    Args:
        results_path: Path to inference_results.jsonl
        max_samples: Optional limit on number of unique queries

    Returns:
        List of ExplorationBundle, one per unique query
    """
    results_path = Path(results_path)
    bundles_by_id: Dict[str, ExplorationBundle] = {}

    with open(results_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            bundle = _parse_rsl_line(data)
            if bundle:
                bundles_by_id[bundle.query_id] = bundle

    bundles = list(bundles_by_id.values())
    if max_samples:
        bundles = bundles[:max_samples]

    total_traces = sum(b.num_trajectories for b in bundles)
    oracle_correct = sum(1 for b in bundles if b.any_successful)
    logger.info(
        f"Loaded {len(bundles)} queries ({total_traces} traces) from {results_path}"
    )
    logger.info(
        f"Oracle accuracy: {oracle_correct}/{len(bundles)} = "
        f"{oracle_correct / len(bundles):.1%}" if bundles else "N/A"
    )

    return bundles


def rsl_stats(bundles: List[ExplorationBundle]) -> Dict[str, Any]:
    """Compute stats for RSL-converted bundles."""
    if not bundles:
        return {}

    oracle_correct = sum(1 for b in bundles if b.any_successful)
    model_stats: Dict[str, Dict[str, int]] = {}

    for bundle in bundles:
        agents = bundle.get_agents_for_mode(ROUTING_MODE)
        for model_key, success in agents.items():
            if model_key not in model_stats:
                model_stats[model_key] = {"correct": 0, "total": 0}
            model_stats[model_key]["total"] += 1
            if success:
                model_stats[model_key]["correct"] += 1

    return {
        "num_queries": len(bundles),
        "total_traces": sum(b.num_trajectories for b in bundles),
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / len(bundles),
        "mode": ROUTING_MODE,
        "model_stats": {
            k: {**v, "accuracy": v["correct"] / v["total"] if v["total"] else 0}
            for k, v in sorted(model_stats.items())
        },
    }


def find_rsl_results(
    rsl_dir: str | Path,
    dataset_prefix: str,
) -> Optional[Path]:
    """Find the latest inference_results.jsonl for a dataset.

    Searches rsl_results/<dataset_prefix>*/ directories for the
    most recent inference_results.jsonl file.
    """
    rsl_dir = Path(rsl_dir)
    candidates = []
    for d in rsl_dir.iterdir():
        if d.is_dir() and d.name.startswith(dataset_prefix):
            for jsonl in d.rglob("inference_results.jsonl"):
                candidates.append(jsonl)

    if not candidates:
        return None

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]
