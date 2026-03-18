"""
Convert orchestration exploration data into SkillOrchestra format.

The orchestration exploration data is structured as:
    <exploration_dir>/
      samples_<num_samples>.jsonl           # task set (query + answer)
      <stage>/<agent_id>/<query_id>.json    # trajectory with <agent_id> forced for <stage> stage
      <stage>/<agent_id>/<query_id>.json    # trajectory with <agent_id> forced for <stage> stage
      ...
      reference/<query_id>.json             # reference/default trajectory

Each trajectory JSON has:
    id, all_tool_calls, all_tool_responses, all_message_responses, correct, costs

This converter reads these files and produces ExplorationBundle objects.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..core.traces import ExecutionStep, ExecutionTrace, ExplorationBundle

logger = logging.getLogger(__name__)

STAGE_TO_MODE = {
    "search": "search",
    "code": "code",
    "answer": "answer",
}

MODE_DIR_TO_MODE = {
    "search": "search",
    "code": "code",
    "answer": "answer",
}

TOOL_NAME_TO_MODE = {
    "search": "search",
    "enhance_reasoning": "code",
    "answer": "answer",
}


def load_tasks(samples_path: str | Path) -> Dict[str, Dict[str, Any]]:
    """Load task set from JSONL file.

    Returns:
        Dict mapping query_id -> {"question": ..., "answer": ...}
    """
    tasks = {}
    with open(samples_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            tasks[item["id"]] = {
                "question": item["question"],
                "answer": item["answer"],
            }
    logger.info(f"Loaded {len(tasks)} tasks from {samples_path}")
    return tasks


def _parse_trajectory(
    trace_json: Dict[str, Any],
    query_id: str,
    query: str,
    ground_truths: List[str],
    varied_mode: str,
    varied_agent_id: str,
) -> ExecutionTrace:
    """Parse a single exploration trajectory JSON into an ExecutionTrace."""
    steps = []
    tool_calls = trace_json.get("all_tool_calls", [])
    tool_responses = trace_json.get("all_tool_responses", {})
    message_responses = trace_json.get("all_message_responses", {})

    for i, turn_calls in enumerate(tool_calls):
        if not isinstance(turn_calls, list):
            continue  # skip malformed entries (e.g. "342 invalid tool calls None")
        for call in turn_calls:
            if not isinstance(call, dict):
                continue
            tool_name = call.get("name", "")
            args = call.get("arguments", {})
            agent_id = args.get("model", "")
            mode = TOOL_NAME_TO_MODE.get(tool_name, tool_name)

            response_key = f"turn_{i}_response"
            message_key = f"turn_{i}_message"
            output_text = ""
            observation = ""

            if response_key in tool_responses:
                resp = tool_responses[response_key]
                if isinstance(resp, str):
                    output_text = resp
                elif isinstance(resp, dict):
                    output_text = resp.get("content", str(resp))

            if message_key in message_responses:
                msg = message_responses[message_key]
                if isinstance(msg, str):
                    observation = msg
                elif isinstance(msg, dict):
                    observation = msg.get("content", str(msg))

            steps.append(ExecutionStep(
                step_idx=i,
                mode=mode,
                agent_id=agent_id,
                model_name=agent_id,
                tools_used=[tool_name],
                input_text=args.get("query", ""),
                output_text=output_text[:2000],
                observation=observation[:2000],
            ))

    costs = trace_json.get("costs", {})
    total_cost = costs.get("total_cost_routed_all_tokens", 0.0)

    return ExecutionTrace(
        query_id=query_id,
        query=query,
        ground_truths=ground_truths,
        steps=steps,
        task_success=trace_json.get("correct", False),
        total_cost_usd=total_cost,
        varied_mode=varied_mode,
        varied_agent_id=varied_agent_id,
        metadata={"costs": costs},
    )


def load_exploration_bundle(
    exploration_dir: str | Path,
    query_id: str,
    query: str,
    ground_truths: List[str],
) -> ExplorationBundle:
    """Load all trajectories for a single query into an ExplorationBundle.

    Args:
        exploration_dir: Root dir containing search/, code/, answer/, reference/
        query_id: The query identifier (e.g., "wiki____367")
        query: The question text
        ground_truths: List of acceptable answers

    Returns:
        ExplorationBundle with all available trajectories
    """
    exploration_dir = Path(exploration_dir)
    trajectories = []

    for mode_dir in ["search", "code", "answer"]:
        mode = MODE_DIR_TO_MODE[mode_dir]
        stage_dir = exploration_dir / mode_dir
        if not stage_dir.is_dir():
            continue

        for agent_dir in sorted(stage_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            agent_id = agent_dir.name
            trace_file = agent_dir / f"{query_id}.json"
            if not trace_file.exists():
                continue

            try:
                with open(trace_file) as f:
                    trace_json = json.load(f)
                trace = _parse_trajectory(
                    trace_json, query_id, query, ground_truths,
                    varied_mode=mode, varied_agent_id=agent_id,
                )
                trajectories.append(trace)
            except Exception as exc:
                logger.warning(f"Failed to parse {trace_file}: {exc}")

    ref_file = exploration_dir / "reference" / f"{query_id}.json"
    if ref_file.exists():
        try:
            with open(ref_file) as f:
                ref_json = json.load(f)
            ref_trace = _parse_trajectory(
                ref_json, query_id, query, ground_truths,
                varied_mode="reference", varied_agent_id="reference",
            )
            trajectories.append(ref_trace)
        except Exception as exc:
            logger.warning(f"Failed to parse reference {ref_file}: {exc}")

    return ExplorationBundle(
        query_id=query_id,
        query=query,
        ground_truths=ground_truths,
        trajectories=trajectories,
    )


def load_exploration_dataset(
    exploration_dir: str | Path,
    samples_path: str | Path,
) -> List[ExplorationBundle]:
    """Load the full exploration dataset.

    Args:
        exploration_dir: Root dir with search/, code/, answer/, reference/
        samples_path: Path to samples JSONL file

    Returns:
        List of ExplorationBundle, one per query
    """
    tasks = load_tasks(samples_path)
    exploration_dir = Path(exploration_dir)
    bundles = []

    for query_id, task in sorted(tasks.items()):
        bundle = load_exploration_bundle(
            exploration_dir,
            query_id=query_id,
            query=task["question"],
            ground_truths=[task["answer"]],
        )
        if bundle.trajectories:
            bundles.append(bundle)
        else:
            logger.warning(f"No trajectories found for {query_id}")

    logger.info(
        f"Loaded {len(bundles)} exploration bundles "
        f"({sum(b.num_trajectories for b in bundles)} total trajectories)"
    )
    return bundles


def exploration_stats(bundles: List[ExplorationBundle]) -> Dict[str, Any]:
    """Compute summary statistics for an exploration dataset."""
    total_traces = sum(b.num_trajectories for b in bundles)
    oracle_correct = sum(1 for b in bundles if b.any_successful)

    mode_agent_stats: Dict[str, Dict[str, Dict[str, int]]] = {}
    for bundle in bundles:
        for trace in bundle.trajectories:
            if trace.varied_mode == "reference":
                continue
            mode = trace.varied_mode
            agent = trace.varied_agent_id
            if mode not in mode_agent_stats:
                mode_agent_stats[mode] = {}
            if agent not in mode_agent_stats[mode]:
                mode_agent_stats[mode][agent] = {"correct": 0, "total": 0}
            mode_agent_stats[mode][agent]["total"] += 1
            if trace.task_success:
                mode_agent_stats[mode][agent]["correct"] += 1

    return {
        "num_queries": len(bundles),
        "total_traces": total_traces,
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / len(bundles) if bundles else 0.0,
        "mode_agent_stats": mode_agent_stats,
    }
