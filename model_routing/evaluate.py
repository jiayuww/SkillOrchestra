#!/usr/bin/env python3
"""Comprehensive evaluation: Oracle vs Baseline vs Skill-based routing.

Runs three evaluation strategies on both training and test sets, producing a
comparison table of accuracy and cost.

Strategies:
  1. Oracle  — best single model per query (from exploration data)
  2. Baseline — plain ROUTER_PROMPT_TEMPLATE (no skill info)
  3. Skill   — handbook + routing strategy (our skill-based routing, usually weighted_avg performs best)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillorchestra.routing.pool_service import (
    MODEL_CONFIGS,
    POOL_MODEL_KEYS,
    API_PRICE_1M_TOKENS,
    call_pool_model,
    call_pool_models_parallel,
    call_router,
    resolve_model_key,
    display_name,
    load_distributed_config,
    check_all_servers,
)

from skillorchestra.eval import compute_exact_match, compute_f1

from skillorchestra.prompts.model_routing import BASELINE_PROMPT
from model_routing.config import MAX_TURNS, MODEL_KEY_TO_DISPLAY

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")

# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(name: str, max_samples: Optional[int] = None) -> List[Dict]:
    """Load QA dataset from HuggingFace. Uses shared loader with pyarrow fallback."""
    import ast
    from model_routing.load_qa_dataset import load_qa_dataset_raw

    rows = load_qa_dataset_raw(name, max_samples)
    samples = []
    for i, row in enumerate(rows):
        gt = row.get("golden_answers", [])
        if isinstance(gt, str):
            try:
                gt = ast.literal_eval(gt)
            except (ValueError, SyntaxError):
                gt = [gt]
        if isinstance(gt, str):
            gt = [gt]
        samples.append({
            "id": row.get("id", f"q_{i}"),
            "sample_id": i,
            "question": row["question"],
            "ground_truths": gt,
        })
    return samples


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

import re

def parse_answer(text: str) -> Optional[str]:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_search(text: str, original_question: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"<search>\s*(.*?)\s*:\s*(.*?)\s*</search>", text, re.DOTALL)
    if not m:
        m = re.search(r"<search>\s*(.*?)\s*</search>", text, re.DOTALL)
        if m:
            content = m.group(1).strip()
            if ":" in content:
                parts = content.split(":", 1)
                return parts[0].strip(), original_question
    if m:
        return m.group(1).strip(), original_question
    return None


# ---------------------------------------------------------------------------
# 1. Oracle — compute from exploration data
# ---------------------------------------------------------------------------

def run_exploration_for_set(
    samples: List[Dict],
    output_path: Path,
    max_tokens: int = 1024,
    temperature: float = 0.6,
    seed: Optional[int] = None,
) -> List[Dict]:
    """Call all 6 pool models on every query. Return per-query records."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = []

    with open(output_path, "w") as f:
        for i, sample in enumerate(samples):
            q = sample["question"]
            gt = sample["ground_truths"]
            logger.info(f"  [explore {i+1}/{len(samples)}] {q[:80]}")

            # Call all pool models in parallel
            parallel_results = call_pool_models_parallel(
                POOL_MODEL_KEYS, q,
                max_tokens=max_tokens,
                temperature=temperature,
                seed=seed,
                max_workers=15,
            )

            model_results = {}
            for mk, result in parallel_results.items():
                em = compute_exact_match(result.response, gt)
                f1 = compute_f1(result.response, gt)
                model_results[mk] = {
                    "response": result.response[:500],
                    "exact_match": em,
                    "f1": f1,
                    "prompt_tokens": result.cost.prompt_tokens,
                    "completion_tokens": result.cost.completion_tokens,
                    "total_cost": result.cost.total,
                }

            best_em = max(mr["exact_match"] for mr in model_results.values())
            best_f1 = max(mr["f1"] for mr in model_results.values())
            best_models = [mk for mk, mr in model_results.items() if mr["exact_match"] == best_em and best_em > 0]
            if best_models:
                oracle_cost = min(model_results[mk]["total_cost"] for mk in best_models)
            else:
                oracle_cost = min(mr["total_cost"] for mr in model_results.values())

            rec = {
                "sample_id": sample["sample_id"],
                "question": q,
                "ground_truths": gt,
                "model_results": model_results,
                "oracle_em": best_em,
                "oracle_f1": best_f1,
                "oracle_models": best_models,
                "oracle_cost": oracle_cost,
            }
            records.append(rec)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()

            correct = [mk for mk, mr in model_results.items() if mr["exact_match"] >= 1.0]
            logger.info(f"    correct: {correct or '(none)'}")

    return records


def compute_oracle_from_file(path: Path) -> Dict:
    """Compute oracle stats from existing exploration JSONL."""
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning(f"Skipping invalid JSON line in {path}")
                continue

    n = len(records)
    oracle_em = sum(r["oracle_em"] for r in records) / n
    oracle_f1 = sum(r["oracle_f1"] for r in records) / n
    oracle_cost = sum(r["oracle_cost"] for r in records) / n

    per_model = {}
    for mk in POOL_MODEL_KEYS:
        ems = [r["model_results"][mk]["exact_match"] for r in records if mk in r["model_results"]]
        f1s = [r["model_results"][mk]["f1"] for r in records if mk in r["model_results"]]
        costs = [r["model_results"][mk]["total_cost"] for r in records if mk in r["model_results"]]
        per_model[mk] = {
            "accuracy": sum(ems) / len(ems) if ems else 0,
            "f1": sum(f1s) / len(f1s) if f1s else 0,
            "avg_cost": sum(costs) / len(costs) if costs else 0,
        }

    return {
        "n": n,
        "oracle_accuracy": oracle_em,
        "oracle_f1": oracle_f1,
        "oracle_avg_cost": oracle_cost,
        "per_model": per_model,
    }


# ---------------------------------------------------------------------------
# 2. Baseline — plain router prompt, no skill info
# ---------------------------------------------------------------------------

def run_baseline_single(
    sample: Dict,
    *,
    router_model: str = "qwen2.5-3b-instruct",
    temperature: float = 0.6,
    seed: Optional[int] = None,
    max_pool_tokens: int = 1024,
) -> Dict[str, Any]:
    """Run baseline (plain routing prompt) on a single query."""
    question = sample["question"]
    gt = sample["ground_truths"]

    conversation = BASELINE_PROMPT.format(question=question)
    total_cost = 0.0
    answer = None
    models_called = []

    for turn_idx in range(MAX_TURNS):
        router_output, r_pt, r_ct = call_router(
            conversation, router_model,
            max_tokens=8192, temperature=temperature, seed=seed,
            stop=["</search>", "</answer>"],
        )

        rp = API_PRICE_1M_TOKENS.get(router_model, {"input": 0, "output": 0})
        total_cost += r_pt * rp["input"] / 1e6 + r_ct * rp["output"] / 1e6

        if not router_output:
            break

        if "<search>" in router_output and "</search>" not in router_output:
            router_output += "</search>"
        if "<answer>" in router_output and "</answer>" not in router_output:
            router_output += "</answer>"

        ans = parse_answer(router_output)
        if ans:
            answer = ans
            break

        search = parse_search(router_output, question)
        if search:
            model_name, query = search
            mk = resolve_model_key(model_name)
            if mk:
                result = call_pool_model(mk, question, max_tokens=max_pool_tokens,
                                         temperature=temperature, seed=seed)
                total_cost += result.cost.total
                models_called.append(display_name(mk))

                conversation += router_output
                resp_text = result.response[:2000] if result.response else ""
                conversation += f"\n\n<information>{resp_text}</information>\n\n"
                continue

        break

    if answer is None:
        answer = ""

    em = compute_exact_match(answer, gt)
    f1 = compute_f1(answer, gt)

    return {
        "sample_id": sample["sample_id"],
        "question": question,
        "ground_truths": gt,
        "answer": answer,
        "exact_match": em,
        "f1": f1,
        "total_cost": total_cost,
        "models_called": models_called,
    }


# ---------------------------------------------------------------------------
# 3. Skill-based routing — reuse test_skill_routing infrastructure
# ---------------------------------------------------------------------------

def _import_skill_routing():
    """Import functions from test_skill_routing.py."""
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import test_skill_routing as tsr
    return tsr


def run_skill_single(
    sample: Dict,
    tsr,
    handbook: Dict,
    skill_catalog_text: str,
    model_performance_text: str,
    model_skill_scores: Dict,
    skill_indicators: Dict,
    model_overall_rates: Optional[Dict[str, float]] = None,
    *,
    lambda_c: float = 0.1,
    temperature: float = 0.6,
    seed: Optional[int] = None,
    max_pool_tokens: int = 1024,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run skill-based routing on a single query."""
    return tsr.run_inference(
        sample, handbook,
        skill_catalog_text, model_performance_text,
        model_skill_scores, skill_indicators,
        model_overall_rates=model_overall_rates,
        routing_strategy="weighted_avg",
        lambda_c=lambda_c,
        temperature=temperature,
        seed=seed,
        max_pool_tokens=max_pool_tokens,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def format_table(results: Dict[str, Dict[str, Dict]]) -> str:
    """Format results into a comparison table.

    results: {strategy_name: {split_name: {accuracy, f1, avg_cost, n}}}
    """
    strategies = list(results.keys())
    splits = []
    for s in strategies:
        for sp in results[s]:
            if sp not in splits:
                splits.append(sp)

    lines = []
    header = f"{'Strategy':<25}"
    for sp in splits:
        header += f" | {'EM ('+sp+')':>12} {'F1 ('+sp+')':>10} {'Cost ('+sp+')':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    for strat in strategies:
        row = f"{strat:<25}"
        for sp in splits:
            d = results[strat].get(sp, {})
            em = d.get("accuracy", 0)
            f1 = d.get("f1", 0)
            cost = d.get("avg_cost", 0)
            n = d.get("n", 0)
            row += f" | {em:>10.1%}  {f1:>8.3f}  ${cost:>10.6f}"
        lines.append(row)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Oracle vs Baseline vs Skill evaluation")
    parser.add_argument("--handbook", required=True, help="Path to learned handbook JSON")
    parser.add_argument("--train-dataset", required=True, help="HF dataset name for training set")
    parser.add_argument("--test-dataset", required=True, help="HF dataset name for test set")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--train-exploration", type=str, default=None,
                        help="Path to existing train exploration JSONL (skip re-running)")
    parser.add_argument("--test-exploration", type=str, default=None,
                        help="Path to existing test exploration JSONL (skip re-running)")
    parser.add_argument("--router-model", default="qwen2.5-3b-instruct")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lambda-c", type=float, default=0.1)
    parser.add_argument("--max-pool-tokens", type=int, default=1024)
    parser.add_argument("--distributed-config", type=str, default=None)
    parser.add_argument("--skip-explore", action="store_true",
                        help="Skip exploration phase entirely (requires --*-exploration)")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-skill", action="store_true")
    parser.add_argument("--skip-refinement", action="store_true",
                        help="Skip failure-driven skill refinement")
    parser.add_argument("--llm-model", type=str, default=None,
                        help="LLM for failure refinement (default: from OPENAI_MODEL or gpt-4o)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.distributed_config:
        load_distributed_config(args.distributed_config)

    random.seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check servers
    status = check_all_servers()
    alive = [k for k, v in status.items() if v]
    logger.info(f"Servers alive: {alive} ({len(alive)}/{len(status)})")

    # Load datasets
    train_samples = load_dataset(args.train_dataset, args.max_train)
    test_samples = load_dataset(args.test_dataset, args.max_test)

    comparison = {}  # {strategy: {split: {accuracy, f1, avg_cost, n}}}

    # =====================================================================
    # Phase 1: Oracle (from exploration)
    # =====================================================================
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: ORACLE (exploration — all models on all queries)")
    logger.info("=" * 60)

    train_explore_path = Path(args.train_exploration) if args.train_exploration else out_dir / "explore_train" / "inference_results.jsonl"
    test_explore_path = Path(args.test_exploration) if args.test_exploration else out_dir / "explore_test" / "inference_results.jsonl"

    if not args.skip_explore:
        if not (args.train_exploration and Path(args.train_exploration).exists()):
            logger.info(f"\nRunning exploration on TRAIN set ({len(train_samples)} samples)...")
            run_exploration_for_set(train_samples, train_explore_path,
                                   max_tokens=args.max_pool_tokens,
                                   temperature=args.temperature, seed=args.seed)
        else:
            logger.info(f"Using existing train exploration: {train_explore_path}")

        if not (args.test_exploration and Path(args.test_exploration).exists()):
            logger.info(f"\nRunning exploration on TEST set ({len(test_samples)} samples)...")
            run_exploration_for_set(test_samples, test_explore_path,
                                   max_tokens=args.max_pool_tokens,
                                   temperature=args.temperature, seed=args.seed)
        else:
            logger.info(f"Using existing test exploration: {test_explore_path}")

    train_oracle = compute_oracle_from_file(train_explore_path)
    test_oracle = compute_oracle_from_file(test_explore_path)

    comparison["Oracle"] = {
        "train": {
            "accuracy": train_oracle["oracle_accuracy"],
            "f1": train_oracle["oracle_f1"],
            "avg_cost": train_oracle["oracle_avg_cost"],
            "n": train_oracle["n"],
        },
        "test": {
            "accuracy": test_oracle["oracle_accuracy"],
            "f1": test_oracle["oracle_f1"],
            "avg_cost": test_oracle["oracle_avg_cost"],
            "n": test_oracle["n"],
        },
    }

    logger.info(f"\nOracle TRAIN: EM={train_oracle['oracle_accuracy']:.1%} F1={train_oracle['oracle_f1']:.3f} cost=${train_oracle['oracle_avg_cost']:.6f}")
    logger.info(f"Oracle TEST:  EM={test_oracle['oracle_accuracy']:.1%} F1={test_oracle['oracle_f1']:.3f} cost=${test_oracle['oracle_avg_cost']:.6f}")

    for split_name, oracle_data in [("train", train_oracle), ("test", test_oracle)]:
        logger.info(f"\n  Per-model {split_name}:")
        for mk in POOL_MODEL_KEYS:
            pm = oracle_data["per_model"].get(mk, {})
            logger.info(f"    {MODEL_KEY_TO_DISPLAY.get(mk, mk):30s} EM={pm.get('accuracy',0):.1%} F1={pm.get('f1',0):.3f} cost=${pm.get('avg_cost',0):.6f}")

    # Save oracle detail
    with open(out_dir / "oracle_results.json", "w") as f:
        json.dump({"train": train_oracle, "test": test_oracle}, f, indent=2)

    # =====================================================================
    # Phase 2: Baseline (plain routing prompt)
    # =====================================================================
    if not args.skip_baseline:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: BASELINE (plain routing prompt, no skills)")
        logger.info("=" * 60)

        for split_name, samples in [("train", train_samples), ("test", test_samples)]:
            logger.info(f"\nRunning baseline on {split_name.upper()} ({len(samples)} samples)...")
            baseline_dir = out_dir / f"baseline_{split_name}"
            baseline_dir.mkdir(parents=True, exist_ok=True)

            results = []
            with open(baseline_dir / "inference_results.jsonl", "w") as f:
                for i, sample in enumerate(samples):
                    logger.info(f"  [{i+1}/{len(samples)}] {sample['question'][:80]}")
                    rec = run_baseline_single(
                        sample,
                        router_model=args.router_model,
                        temperature=args.temperature,
                        seed=args.seed,
                        max_pool_tokens=args.max_pool_tokens,
                    )
                    results.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    logger.info(f"    → EM={rec['exact_match']:.1f} model={rec['models_called']}")

            avg_em = sum(r["exact_match"] for r in results) / len(results)
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            avg_cost = sum(r["total_cost"] for r in results) / len(results)

            summary = {
                "strategy": "baseline",
                "split": split_name,
                "n": len(results),
                "accuracy": avg_em,
                "f1": avg_f1,
                "avg_cost": avg_cost,
                "total_cost": sum(r["total_cost"] for r in results),
                "model_distribution": {},
            }
            for r in results:
                for m in r["models_called"]:
                    summary["model_distribution"][m] = summary["model_distribution"].get(m, 0) + 1

            with open(baseline_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            if "Baseline" not in comparison:
                comparison["Baseline"] = {}
            comparison["Baseline"][split_name] = {
                "accuracy": avg_em, "f1": avg_f1, "avg_cost": avg_cost, "n": len(results),
            }

            logger.info(f"  Baseline {split_name}: EM={avg_em:.1%} F1={avg_f1:.3f} cost=${avg_cost:.6f}")
            logger.info(f"  Model distribution: {summary['model_distribution']}")

    # =====================================================================
    # Phase 3: Skill-based routing (handbook + weighted_avg)
    # =====================================================================
    if not args.skip_skill:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: SKILL-BASED (handbook + weighted_avg)")
        logger.info("=" * 60)

        tsr = _import_skill_routing()
        hb = tsr.load_handbook(args.handbook)
        skill_catalog_text = tsr.extract_skill_catalog_text(hb)
        model_perf_text = tsr.extract_model_performance_text(hb)
        model_skill_scores = tsr.extract_model_skill_scores(hb)
        model_overall_rates = tsr.extract_model_overall_rates(hb)
        skill_indicators = tsr.extract_skill_indicators(hb)

        logger.info(f"Handbook: {len(model_skill_scores)} models, "
                    f"{sum(len(v) for v in model_skill_scores.values())} total skill scores")

        train_skill_results = []
        for split_name, samples in [("train", train_samples), ("test", test_samples)]:
            logger.info(f"\nRunning skill routing on {split_name.upper()} ({len(samples)} samples)...")
            skill_dir = out_dir / f"skill_{split_name}"
            skill_dir.mkdir(parents=True, exist_ok=True)

            results = []
            with open(skill_dir / "inference_results.jsonl", "w") as f:
                for i, sample in enumerate(samples):
                    logger.info(f"  [{i+1}/{len(samples)}] {sample['question'][:80]}")
                    rec = run_skill_single(
                        sample, tsr, hb,
                        skill_catalog_text, model_perf_text,
                        model_skill_scores, skill_indicators,
                        model_overall_rates=model_overall_rates,
                        lambda_c=args.lambda_c,
                        temperature=args.temperature,
                        seed=args.seed,
                        max_pool_tokens=args.max_pool_tokens,
                        verbose=args.verbose,
                    )
                    results.append(rec)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    logic = rec["turns"][0]["routing_decision_logic"] if rec.get("turns") else "?"
                    logger.info(f"    → EM={rec['exact_match']:.1f} model={rec.get('models_called', [])} logic={logic}")

            avg_em = sum(r["exact_match"] for r in results) / len(results)
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            costs = []
            model_dist = {}
            for r in results:
                c = r.get("costs", {}).get("all_total", 0)
                if c == 0:
                    c = sum(
                        t.get("cost", 0) for t in r.get("turns", [])
                    )
                    rp = API_PRICE_1M_TOKENS.get(args.router_model, {"input": 0, "output": 0})
                    c += r.get("tokens", {}).get("router_prompt", 0) * rp["input"] / 1e6
                    c += r.get("tokens", {}).get("router_completion", 0) * rp["output"] / 1e6
                costs.append(c)
                for m in r.get("models_called", []):
                    model_dist[m] = model_dist.get(m, 0) + 1

            avg_cost = sum(costs) / len(costs) if costs else 0

            summary = {
                "strategy": "skill_routing",
                "split": split_name,
                "n": len(results),
                "accuracy": avg_em,
                "f1": avg_f1,
                "avg_cost": avg_cost,
                "total_cost": sum(costs),
                "model_distribution": model_dist,
                "lambda_c": args.lambda_c,
            }

            with open(skill_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)

            if "Skill-based" not in comparison:
                comparison["Skill-based"] = {}
            comparison["Skill-based"][split_name] = {
                "accuracy": avg_em, "f1": avg_f1, "avg_cost": avg_cost, "n": len(results),
            }

            logger.info(f"  Skill {split_name}: EM={avg_em:.1%} F1={avg_f1:.3f} cost=${avg_cost:.6f}")
            logger.info(f"  Model distribution: {model_dist}")

            if split_name == "train":
                train_skill_results = results

        # =====================================================================
        # Phase 4: Failure-driven refinement (when skill < oracle on train)
        # =====================================================================
        if (
            not args.skip_refinement
            and not args.skip_skill
            and "Skill-based" in comparison
            and "Oracle" in comparison
        ):
            skill_train_em = comparison["Skill-based"]["train"]["accuracy"]
            oracle_train_em = comparison["Oracle"]["train"]["accuracy"]

            if skill_train_em < oracle_train_em and train_skill_results:
                logger.info("\n" + "=" * 60)
                logger.info("PHASE 4: FAILURE-DRIVEN SKILL REFINEMENT")
                logger.info("=" * 60)
                logger.info(f"Skill train EM ({skill_train_em:.1%}) < Oracle ({oracle_train_em:.1%})")
                logger.info("Asking LLM to reflect on failed queries and propose skill refinements...")

                # Load exploration records
                explore_records = []
                with open(train_explore_path) as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            explore_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

                from skillorchestra.llm.client import LLMClient
                from skillorchestra.learning.failure_refiner import FailureDrivenRefiner

                llm_model = args.llm_model or os.environ.get("OPENAI_MODEL", "gpt-5")
                llm = LLMClient(model=llm_model)
                refiner = FailureDrivenRefiner(llm=llm)

                refinement = refiner.refine(
                    exploration_records=explore_records,
                    skill_routing_results=train_skill_results,
                    handbook=hb,
                    oracle_accuracy=oracle_train_em,
                    skill_accuracy=skill_train_em,
                )

                refinement_dir = out_dir / "refinement"
                refinement_dir.mkdir(parents=True, exist_ok=True)

                refinement_out = {
                    "triggered": refinement.triggered,
                    "num_failed": refinement.num_failed,
                    "rationale": refinement.rationale,
                    "proposed_new_skills": refinement.proposed_new_skills,
                    "proposed_splits": refinement.proposed_splits,
                    "oracle_train_em": oracle_train_em,
                    "skill_train_em": skill_train_em,
                }
                with open(refinement_dir / "failure_refinement.json", "w") as f:
                    json.dump(refinement_out, f, indent=2)

                logger.info(f"\nRefinement rationale:\n{refinement.rationale[:500]}...")
                logger.info(f"  Proposed new skills: {len(refinement.proposed_new_skills)}")
                logger.info(f"  Proposed splits: {len(refinement.proposed_splits)}")
                logger.info(f"  Saved to {refinement_dir / 'failure_refinement.json'}")
            else:
                logger.info("\nSkipping failure refinement: skill >= oracle on train or no train results")

    # =====================================================================
    # Phase 5: Comparison table
    # =====================================================================
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON TABLE")
    logger.info("=" * 60)

    table = format_table(comparison)
    logger.info("\n" + table)

    with open(out_dir / "comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    with open(out_dir / "comparison_table.txt", "w") as f:
        f.write(table + "\n")

    logger.info(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
