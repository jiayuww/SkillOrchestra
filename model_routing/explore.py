#!/usr/bin/env python3
"""Generate exploration data by running all pool models on a dataset.

Usage:
    python model_routing/explore.py --dataset nq_validation_qwen --output-dir output/nq/exploration --max-samples 20
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillorchestra.routing.pool_service import (
    call_pool_models_parallel,
    check_all_servers,
    load_distributed_config,
)
from skillorchestra.eval import compute_exact_match, compute_f1

from model_routing.config import POOL_MODELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("explore")


def load_dataset_samples(dataset_name: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load dataset from HuggingFace."""
    from datasets import load_dataset

    is_validation = "validation" in dataset_name
    source = "MilaWang/qa_validation_qwen" if is_validation else "MilaWang/qa_test_qwen"

    logger.info(f"Loading {source}/{dataset_name} from HuggingFace...")
    ds = load_dataset(source, dataset_name, split="test")
    samples = list(ds)
    if max_samples:
        samples = samples[:max_samples]
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def process_sample(
    sample_idx: int,
    sample: Dict,
    pool_models: List[str],
    max_tokens: int = 600,
) -> Dict[str, Any]:
    """Process a single sample: call all pool models and evaluate."""
    question = sample.get("question", sample.get("query", ""))
    ground_truths = sample.get("ground_truths", sample.get("golden_answers", sample.get("answer", [])))
    if isinstance(ground_truths, str):
        ground_truths = [ground_truths]

    record = {
        "sample_id": sample_idx,
        "question": question,
        "ground_truths": ground_truths,
        "model_responses": {},
        "model_exact_match": {},
        "model_f1": {},
        "model_succeeded": {},
        "model_costs": {},
        "best_model": None,
        "best_score": 0.0,
        "timestamp": datetime.now().isoformat(),
    }

    # Call all pool models in parallel
    parallel_results = call_pool_models_parallel(
        pool_models, question,
        max_tokens=max_tokens,
        max_workers=15,
    )

    for model_key, result in parallel_results.items():
        response = result.response
        cost = result.cost
        em = compute_exact_match(response, ground_truths)
        f1 = compute_f1(response, ground_truths)

        record["model_responses"][model_key] = response
        record["model_exact_match"][model_key] = em
        record["model_f1"][model_key] = f1
        record["model_succeeded"][model_key] = em >= 1.0
        record["model_costs"][model_key] = {
            "prompt_tokens": cost.prompt_tokens,
            "completion_tokens": cost.completion_tokens,
            "input_cost": cost.input_cost,
            "output_cost": cost.output_cost,
            "total_cost": cost.total,
        }

        if em > record["best_score"]:
            record["best_score"] = em
            record["best_model"] = model_key

    return record


def run_exploration(
    dataset_name: str,
    output_dir: Path,
    max_samples: Optional[int] = None,
    pool_models: Optional[List[str]] = None,
    max_tokens: int = 600,
    num_workers: int = 1,
) -> Tuple[List[Dict], Dict]:
    """Run exploration: call all pool models on all queries."""
    if pool_models is None:
        pool_models = POOL_MODELS

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = load_dataset_samples(dataset_name, max_samples)

    # Health check
    logger.info("Checking pool model servers...")
    health = check_all_servers()
    for model, ok in health.items():
        status = "OK" if ok else "FAILED"
        logger.info(f"  {model}: {status}")
    unhealthy = [m for m, ok in health.items() if not ok and m in pool_models]
    if unhealthy:
        logger.warning(f"Unhealthy models: {unhealthy}")

    results = []
    results_path = output_dir / "inference_results.jsonl"
    start_time = time.time()

    logger.info(f"Processing {len(samples)} samples with {len(pool_models)} models...")

    with open(results_path, "w") as f:
        for i, sample in enumerate(samples):
            logger.info(f"[{i+1}/{len(samples)}] {sample.get('question', '')[:80]}...")
            record = process_sample(i, sample, pool_models, max_tokens)
            results.append(record)
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            correct = [m for m, s in record["model_succeeded"].items() if s]
            logger.info(
                f"  Correct: {correct or '(none)'} | "
                f"Best: {record['best_model']} ({record['best_score']:.2f})"
            )

    elapsed = time.time() - start_time

    # Compute summary
    model_correct = {m: 0 for m in pool_models}
    model_total = {m: 0 for m in pool_models}
    oracle_correct = 0
    total_cost = 0.0

    for r in results:
        any_correct = False
        for m in pool_models:
            model_total[m] += 1
            if r["model_succeeded"].get(m, False):
                model_correct[m] += 1
                any_correct = True
            for cost_key in ["total_cost"]:
                total_cost += r["model_costs"].get(m, {}).get(cost_key, 0.0)
        if any_correct:
            oracle_correct += 1

    summary = {
        "dataset": dataset_name,
        "num_samples": len(results),
        "oracle_correct": oracle_correct,
        "oracle_accuracy": oracle_correct / len(results) if results else 0,
        "model_accuracies": {
            m: {
                "correct": model_correct[m],
                "total": model_total[m],
                "accuracy": round(model_correct[m] / model_total[m], 4) if model_total[m] else 0,
            }
            for m in pool_models
        },
        "total_cost_usd": round(total_cost, 6),
        "elapsed_seconds": round(elapsed, 1),
        "pool_models": pool_models,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = output_dir / "exploration_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nExploration complete in {elapsed:.1f}s")
    logger.info(f"  Oracle accuracy: {oracle_correct}/{len(results)} = {summary['oracle_accuracy']:.1%}")
    for m in pool_models:
        acc = model_correct[m] / model_total[m] if model_total[m] else 0
        logger.info(f"  {m}: {model_correct[m]}/{model_total[m]} = {acc:.1%}")
    logger.info(f"  Total cost: ${total_cost:.4f}")
    logger.info(f"  Results: {results_path}")

    return results, summary


def main():
    parser = argparse.ArgumentParser(
        description="Generate exploration data by running all pool models on a dataset",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="HuggingFace dataset name (e.g., nq_validation_qwen)")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples")
    parser.add_argument("--pool-models", type=str, default=None,
                        help="Comma-separated pool model keys (default: all 6)")
    parser.add_argument("--max-tokens", type=int, default=600,
                        help="Max tokens per pool model response")
    parser.add_argument("--distributed-config", type=str, default=None,
                        help="Path to distributed config JSON")

    args = parser.parse_args()

    if args.distributed_config:
        load_distributed_config(args.distributed_config)

    pool_models = args.pool_models.split(",") if args.pool_models else None
    run_exploration(
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        max_samples=args.max_samples,
        pool_models=pool_models,
        max_tokens=args.max_tokens,
    )


if __name__ == "__main__":
    main()
