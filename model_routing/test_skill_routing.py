#!/usr/bin/env python3
"""
Test skill-based model routing using a learned handbook.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from skillorchestra.routing.pool_service import (
    MODEL_CONFIGS,
    POOL_MODEL_KEYS,
    API_PRICE_1M_TOKENS,
    call_pool_model,
    call_router,
    resolve_model_key,
    display_name,
    load_distributed_config,
    check_all_servers,
)
from skillorchestra.eval import compute_exact_match, compute_f1

from skillorchestra.prompts.model_routing import SKILL_ANALYSIS_PROMPT
from model_routing.config import (
    MAX_TURNS,
    MODEL_KEY_TO_DISPLAY,
    MODEL_RELATIVE_COST_FALLBACK,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_skill_routing")

MAX_RETRIES = 10


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class SkillWeight:
    skill_id: str
    percentage: float

@dataclass
class SkillAnalysis:
    required_skills: List[SkillWeight]
    reasoning: str = ""

@dataclass
class TurnRecord:
    turn_idx: int
    action: str
    router_output: str
    routed_model: Optional[str] = None
    routed_query: Optional[str] = None
    routed_response: Optional[str] = None
    cost: float = 0.0
    router_prompt_tokens: int = 0
    router_completion_tokens: int = 0
    routing_decision_logic: Optional[str] = None
    skill_analysis: Optional[SkillAnalysis] = None
    indicator_fallback_analysis: Optional[SkillAnalysis] = None


# ---------------------------------------------------------------------------
# Handbook loader
# ---------------------------------------------------------------------------

def load_handbook(path: str) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def extract_skill_catalog_text(hb: Dict) -> str:
    """Build the skill catalog text from handbook JSON (same format as ar).

    Includes skill_id so the router knows the exact IDs to use in <skill_analysis>.
    """
    catalog = hb.get("skill_catalog", {})
    categories = catalog.get("categories", {})
    lines: List[str] = []
    for cat_id, cat in categories.items():
        lines.append(f"\n### {cat_id}")
        lines.append(f"Skills related to {cat_id}")
        for sk_id, sk in cat.get("skills", {}).items():
            desc = sk.get("description", "")
            name = sk.get("name", sk_id)
            lines.append(f"  - {sk_id}: {name} - {desc}")
            examples = sk.get("examples", [])
            if examples:
                lines.append(f"    Examples: {', '.join(examples[:3])}")
    return "\n".join(lines)


def extract_model_performance_text(hb: Dict) -> str:
    """Build the model performance text from handbook JSON (same format as ar)."""
    profiles = hb.get("model_profiles", hb.get("agent_profiles", {}))
    lines: List[str] = []
    for model_id, prof in profiles.items():
        dname = MODEL_KEY_TO_DISPLAY.get(model_id, model_id)
        total = prof.get("total_attempts", 0)
        correct = prof.get("total_successes", 0)
        overall = correct / total if total else 0
        lines.append(f"\n### {dname}")
        lines.append(f"Overall: {int(overall*100)}% success ({correct}/{total})")

        skill_scores = prof.get("skill_scores", {})
        if skill_scores:
            lines.append("Skill scores:")
            for sk, score in sorted(skill_scores.items(), key=lambda x: -x[1])[:5]:
                lines.append(f"  - {sk}: {int(score*100)}%")

        strengths = prof.get("strengths", "")
        weaknesses = prof.get("weaknesses", "")
        if strengths:
            lines.append(f"Strengths: {strengths}")
        if weaknesses:
            lines.append(f"Weaknesses: {weaknesses}")

    return "\n".join(lines)


def extract_model_skill_scores(hb: Dict) -> Dict[str, Dict[str, float]]:
    """Return {display_name: {skill_id: score}} for weighted_avg routing.

    Keys use display names (matching the router prompt) so that the
    weighted_avg selection returns a name resolvable by pool_service.
    """
    profiles = hb.get("model_profiles", hb.get("agent_profiles", {}))
    result: Dict[str, Dict[str, float]] = {}
    for model_id, prof in profiles.items():
        dname = MODEL_KEY_TO_DISPLAY.get(model_id, model_id)
        result[dname] = prof.get("skill_scores", {})
    return result


def extract_model_overall_rates(hb: Dict) -> Dict[str, float]:
    """Return {display_name: overall_success_rate} for tie-breaking.

    Uses handbook total_successes/total_attempts (skill-level aggregate).
    When models tie on skill-weighted score, use this higher-level rate.
    """
    profiles = hb.get("model_profiles", hb.get("agent_profiles", {}))
    result: Dict[str, float] = {}
    for model_id, prof in profiles.items():
        dname = MODEL_KEY_TO_DISPLAY.get(model_id, model_id)
        total = prof.get("total_attempts", 0)
        correct = prof.get("total_successes", 0)
        result[dname] = correct / total if total > 0 else 0.0
    return result


def extract_skill_indicators(hb: Dict) -> Dict[str, Dict]:
    """Return {skill_id: {indicators: [...], examples: [...]}} from catalog."""
    catalog = hb.get("skill_catalog", {})
    result: Dict[str, Dict] = {}
    for cat_id, cat in catalog.get("categories", {}).items():
        for sk_id, sk in cat.get("skills", {}).items():
            result[sk_id] = {
                "indicators": sk.get("indicators", []),
                "examples": sk.get("examples", []),
            }
    return result


def build_skill_id_normalizer(hb: Dict) -> Dict[str, str]:
    """Return {alias: canonical_skill_id} so router output can be normalized."""
    catalog = hb.get("skill_catalog", {})
    mapping: Dict[str, str] = {}
    for cat_id, cat in catalog.get("categories", {}).items():
        for sk_id, sk in cat.get("skills", {}).items():
            mapping[sk_id] = sk_id
            name = sk.get("name", "")
            if name:
                mapping[name] = sk_id
                mapping[name.lower()] = sk_id
                mapping[name.lower().replace(" ", "_").replace("-", "_")] = sk_id
    return mapping


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def parse_answer(text: str) -> Optional[str]:
    m = re.search(r"<answer>\s*(.+?)\s*</answer>", text, re.DOTALL)
    return m.group(1).strip() if m else None


def parse_search(text: str, original_question: str) -> Optional[Tuple[str, str]]:
    m = re.search(r"<search>\s*(.+?)\s*</search>", text, re.DOTALL)
    if not m:
        return None
    content = m.group(1).strip()
    if ":" not in content:
        return None
    parts = content.split(":", 1)
    model_name = parts[0].strip()
    query = parts[1].strip() if len(parts) > 1 else original_question
    if not query or len(query) < 5:
        query = original_question
    return model_name, query


def parse_skill_analysis(text: str) -> Optional[SkillAnalysis]:
    m = re.search(r"<skill_analysis>\s*(.+?)\s*</skill_analysis>", text, re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip()
    if not raw or raw == "{}":
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        raw_fixed = re.sub(r",\s*]", "]", re.sub(r",\s*}", "}", raw))
        try:
            data = json.loads(raw_fixed)
        except json.JSONDecodeError:
            return None
    skills_raw = data.get("required_skills", [])
    if not skills_raw:
        return None
    skills = [SkillWeight(skill_id=s["skill_id"], percentage=s.get("percentage", 50))
              for s in skills_raw if "skill_id" in s]
    return SkillAnalysis(required_skills=skills,
                         reasoning=data.get("reasoning", "")) if skills else None


def identify_skills_by_indicators(
    query: str,
    skill_indicators: Dict[str, Dict],
) -> Optional[SkillAnalysis]:
    """Keyword/indicator-based skill identification fallback."""
    q = query.lower()
    raw: Dict[str, float] = {}
    for sk_id, info in skill_indicators.items():
        hits = 0.0
        for ind in info.get("indicators", []):
            if ind.lower() in q:
                hits += 1.0
        for ex in info.get("examples", []):
            overlap = set(ex.lower().split()) & set(q.split())
            if len(overlap) >= 3:
                hits += 0.5
        if hits > 0:
            raw[sk_id] = hits
    if not raw:
        return None
    total = sum(raw.values())
    skills = [SkillWeight(skill_id=sid, percentage=round(w / total * 100, 1))
              for sid, w in sorted(raw.items(), key=lambda x: -x[1])]
    return SkillAnalysis(required_skills=skills,
                         reasoning="indicator-based fallback")


# ---------------------------------------------------------------------------
# Weighted-average model selection (cost-penalized, probabilistic)
# ---------------------------------------------------------------------------

def extract_model_costs(hb: Dict) -> Dict[str, float]:
    """Extract avg cost per run from handbook."""
    profiles = hb.get("model_profiles", hb.get("agent_profiles", {}))
    result: Dict[str, float] = {}
    for model_id, prof in profiles.items():
        dname = MODEL_KEY_TO_DISPLAY.get(model_id, model_id)
        total_cost = prof.get("total_cost_usd", 0)
        n = prof.get("total_executions", 0) or prof.get("total_attempts", 0)
        avg_cost = total_cost / n if n > 0 else 0
        result[dname] = avg_cost
    return result


def _category_competence_for_skills(
    model_skill_scores: Dict[str, Dict[str, float]],
    required_skills: List[SkillWeight],
    skill_id_normalizer: Optional[Dict[str, str]] = None,
) -> Dict[str, float]:
    """Compute category-level competence per model for hierarchical tie-breaking."""
    if not required_skills:
        return {}

    def _canonical(sid: str) -> str:
        if skill_id_normalizer and sid in skill_id_normalizer:
            return skill_id_normalizer[sid]
        if skill_id_normalizer:
            norm = sid.lower().strip().replace(" ", "_").replace("-", "_")
            return skill_id_normalizer.get(norm, sid)
        return sid

    categories = {
        sw.skill_id.rsplit(".", 1)[0] if "." in sw.skill_id else sw.skill_id
        for sw in required_skills
    }
    if not categories:
        return {}

    result: Dict[str, float] = {}
    for model_name, skill_map in model_skill_scores.items():
        cat_scores: List[float] = []
        cat_weights: List[float] = []
        for cat in categories:
            prefix = cat + "."
            under_cat = [
                (sw.skill_id, sw.percentage / 100.0)
                for sw in required_skills
                if sw.skill_id.startswith(prefix) or (sw.skill_id == cat and "." not in sw.skill_id)
            ]
            if not under_cat:
                continue
            wsum = 0.0
            wtot = 0.0
            for sid, pct in under_cat:
                canonical_id = _canonical(sid)
                score = skill_map.get(canonical_id, skill_map.get(sid, 0.0))
                wsum += score * pct
                wtot += pct
            if wtot > 0:
                cat_scores.append(wsum / wtot)
                cat_weights.append(wtot)
        if cat_scores and cat_weights:
            total_w = sum(cat_weights)
            result[model_name] = (
                sum(s * w for s, w in zip(cat_scores, cat_weights)) / total_w
                if total_w > 0 else 0.0
            )
        else:
            result[model_name] = 0.0
    return result


def route_by_weighted_avg(
    skill_analysis: SkillAnalysis,
    model_skill_scores: Dict[str, Dict[str, float]],
    model_overall_rates: Optional[Dict[str, float]] = None,
    model_costs: Optional[Dict[str, float]] = None,
    skill_id_normalizer: Optional[Dict[str, str]] = None,
    lambda_c: float = 0.1,
    verbose: bool = False,
) -> Tuple[Optional[str], str]:
    """Select model via weighted average — skill → category → mode → cost.

    1. Compute skill-weighted score per model.
    2. Find the set of models tied at the top score.
    3. If only one winner, return it directly.
    4. Among tied models: go up to category level (e.g. entertainment_knowledge).
    5. If still tied: go up to mode level (overall success rate).
    6. If still tied: cost-penalize, normalize to probs, sample.

    Returns (display_name, decision_logic).
    """
    import random

    def _canonical(sid: str) -> str:
        if skill_id_normalizer and sid in skill_id_normalizer:
            return skill_id_normalizer[sid]
        if skill_id_normalizer:
            # Try normalized form (lower, underscores)
            norm = sid.lower().strip().replace(" ", "_").replace("-", "_")
            return skill_id_normalizer.get(norm, sid)
        return sid

    if not skill_analysis.required_skills:
        return None, "weighted_avg_no_skills"

    # Step 1: skill-weighted scores
    raw_scores: Dict[str, float] = {}
    for model_name, skill_map in model_skill_scores.items():
        wsum = 0.0
        wtot = 0.0
        for sw in skill_analysis.required_skills:
            pct = sw.percentage / 100.0
            canonical_id = _canonical(sw.skill_id)
            score = skill_map.get(canonical_id, skill_map.get(sw.skill_id, 0.0))
            wsum += score * pct
            wtot += pct
        raw_scores[model_name] = wsum / wtot if wtot > 0 else 0.0

    if not raw_scores:
        return None, "weighted_avg_no_scores"

    # Step 2: find top-performing models (all tied at 0 still go through hierarchy)
    max_score = max(raw_scores.values())
    tied = [m for m, s in raw_scores.items() if abs(s - max_score) < 1e-9]

    # Step 3: single winner — no tie-break needed
    if len(tied) == 1:
        if verbose:
            logger.info(f"[weighted_avg] scores: {json.dumps({m: round(s, 4) for m, s in raw_scores.items()})}")
            logger.info(f"[weighted_avg] clear winner: {tied[0]} (score={max_score:.4f})")
        return tied[0], "weighted_avg_from_skill_analysis"

    # Step 4: among tied models, go up to category level (weighted by skill importance)
    still_tied = tied
    if skill_analysis.required_skills:
        model_category_scores = _category_competence_for_skills(
            model_skill_scores,
            skill_analysis.required_skills,
            skill_id_normalizer=skill_id_normalizer,
        )
        if model_category_scores:
            by_category = sorted(
                tied,
                key=lambda m: model_category_scores.get(m, 0.0),
                reverse=True,
            )
            best_cat = model_category_scores.get(by_category[0], 0.0) if by_category else 0.0
            still_tied = [m for m in by_category
                          if model_category_scores.get(m, 0.0) >= best_cat - 1e-9]

    # Step 5: if still tied, go up to mode level (overall success rate)
    if len(still_tied) > 1 and model_overall_rates:
        by_overall = sorted(
            still_tied,
            key=lambda m: model_overall_rates.get(m, 0.0),
            reverse=True,
        )
        best_overall = model_overall_rates.get(by_overall[0], 0.0) if by_overall else 0.0
        still_tied = [m for m in by_overall
                      if model_overall_rates.get(m, 0.0) >= best_overall - 1e-9]

    # Step 6: if still tied, cost-penalize → normalize → sample
    # Use handbook-learned costs (avg $ per run) when available; else fallback to API prices
    if model_costs and any(model_costs.get(m, 0) > 0 for m in still_tied):
        max_cost = max(model_costs.get(m, 0) for m in still_tied) or 1e-9
        cost_map = {m: model_costs.get(m, 0) / max_cost for m in still_tied}
    else:
        cost_map = {m: MODEL_RELATIVE_COST_FALLBACK.get(m, 0.5) for m in still_tied}
    adjusted: Dict[str, float] = {}
    for m in still_tied:
        cost = cost_map.get(m, 0.5)
        adjusted[m] = max(0.0, 1.0 - lambda_c * cost)

    total = sum(adjusted.values())
    if total <= 0:
        probs = {m: 1.0 / len(still_tied) for m in still_tied}
    else:
        probs = {m: v / total for m, v in adjusted.items()}

    models = list(probs.keys())
    weights = [probs[m] for m in models]
    selected = random.choices(models, weights=weights, k=1)[0]

    if verbose:
        logger.info(f"[weighted_avg] scores: {json.dumps({m: round(s, 4) for m, s in raw_scores.items()})}")
        logger.info(f"[weighted_avg] tied at {max_score:.4f}: {tied}")
        logger.info(f"[weighted_avg] cost-adjusted probs: {json.dumps({m: round(p, 3) for m, p in probs.items()})}")
        logger.info(f"[weighted_avg] sampled: {selected} (prob={probs[selected]:.3f})")

    return selected, "weighted_avg_from_skill_analysis"


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


def load_dataset_from_jsonl(path: str) -> List[Dict]:
    """Load samples from a JSONL file (one sample per line).

    Expected keys per line: question, ground_truths, id (optional).
    """
    samples = []
    with open(path) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            gt = row.get("ground_truths", row.get("golden_answers", []))
            if isinstance(gt, str):
                gt = [gt]
            samples.append({
                "id": row.get("id", f"q_{i}"),
                "sample_id": i,
                "question": row["question"],
                "ground_truths": gt,
            })
    logger.info(f"Loaded {len(samples)} samples from {path}")
    return samples

# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def run_inference(
    sample: Dict,
    handbook: Dict,
    skill_catalog_text: str,
    model_performance_text: str,
    model_skill_scores: Dict[str, Dict[str, float]],
    skill_indicators: Dict[str, Dict],
    model_overall_rates: Optional[Dict[str, float]] = None,
    *,
    router_model: str = "qwen2.5-3b-instruct",
    routing_strategy: str = "weighted_avg",
    temperature: float = 0.6,
    seed: Optional[int] = None,
    max_pool_tokens: int = 1024,
    lambda_c: float = 0.01,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run multi-turn router inference on a single question."""
    question = sample["question"]
    ground_truths = sample["ground_truths"]
    skill_id_normalizer = build_skill_id_normalizer(handbook)
    model_costs = extract_model_costs(handbook)

    conversation = SKILL_ANALYSIS_PROMPT.format(
        skill_catalog=skill_catalog_text,
        model_performance=model_performance_text,
        question=question,
    )

    turns: List[TurnRecord] = []
    models_called: List[str] = []
    total_router_pt, total_router_ct = 0, 0
    total_pool_pt, total_pool_ct = 0, 0
    total_cost = 0.0
    per_model_costs: Dict[str, Dict] = {}
    answer = None
    skill_analysis_parsed = 0
    skill_analysis_failures = 0
    parse_errors = 0

    for turn_idx in range(MAX_TURNS):
        router_output, r_pt, r_ct = call_router(
            conversation, router_model,
            max_tokens=8192, temperature=temperature, seed=seed,
            stop=["</search>", "</answer>"],
        )
        total_router_pt += r_pt
        total_router_ct += r_ct

        # Track router cost
        rp = API_PRICE_1M_TOKENS.get(router_model, {"input": 0, "output": 0})
        r_in_cost = r_pt * rp["input"] / 1_000_000
        r_out_cost = r_ct * rp["output"] / 1_000_000
        total_cost += r_in_cost + r_out_cost
        rkey = f"router:{router_model}"
        if rkey not in per_model_costs:
            per_model_costs[rkey] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                     "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
        per_model_costs[rkey]["calls"] += 1
        per_model_costs[rkey]["prompt_tokens"] += r_pt
        per_model_costs[rkey]["completion_tokens"] += r_ct
        per_model_costs[rkey]["input_cost"] += r_in_cost
        per_model_costs[rkey]["output_cost"] += r_out_cost
        per_model_costs[rkey]["total_cost"] += r_in_cost + r_out_cost

        if not router_output:
            logger.warning(f"Router returned empty for: {question[:60]}")
            break

        # Re-attach stop tokens stripped by SGLang
        if "<search>" in router_output and "</search>" not in router_output:
            router_output += "</search>"
        if "<answer>" in router_output and "</answer>" not in router_output:
            router_output += "</answer>"

        turn = TurnRecord(turn_idx=turn_idx + 1, action="none",
                          router_output=router_output,
                          router_prompt_tokens=r_pt, router_completion_tokens=r_ct)

        # Check for answer
        ans = parse_answer(router_output)
        if ans:
            turn.action = "answer"
            turn.routing_decision_logic = "router_provided_answer_directly"
            answer = ans
            turns.append(turn)
            break

        # Skill analysis + routing (first turn only)
        selected_model = None
        decision_logic = None

        if routing_strategy != "router_decides" and turn_idx == 0:
            sa = parse_skill_analysis(router_output)
            if sa:
                skill_analysis_parsed += 1
                turn.skill_analysis = sa
                selected_model, decision_logic = route_by_weighted_avg(
                    sa, model_skill_scores,
                    model_overall_rates=model_overall_rates,
                    model_costs=model_costs,
                    skill_id_normalizer=skill_id_normalizer,
                    lambda_c=lambda_c, verbose=verbose)
            else:
                skill_analysis_failures += 1
                ind = identify_skills_by_indicators(question, skill_indicators)
                if ind:
                    turn.indicator_fallback_analysis = ind
                    selected_model, decision_logic = route_by_weighted_avg(
                        ind, model_skill_scores,
                        model_overall_rates=model_overall_rates,
                        model_costs=model_costs,
                        skill_id_normalizer=skill_id_normalizer,
                        lambda_c=lambda_c, verbose=verbose)
                    if decision_logic:
                        decision_logic += "_indicator_fallback"
                else:
                    parse_errors += 1

        # If routing selected a model, call it via pool_service
        if selected_model:
            mk = resolve_model_key(selected_model)
            if not mk:
                mk = resolve_model_key(selected_model.split("-")[0])

            result = call_pool_model(mk, question, max_tokens=max_pool_tokens,
                                     temperature=temperature, seed=seed)
            turn.action = "search"
            turn.routed_model = selected_model
            turn.routed_query = question
            turn.routed_response = result.response
            turn.cost = result.cost.total
            turn.routing_decision_logic = decision_logic
            models_called.append(selected_model)

            total_pool_pt += result.cost.prompt_tokens
            total_pool_ct += result.cost.completion_tokens
            total_cost += result.cost.total

            if mk not in per_model_costs:
                per_model_costs[mk] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                       "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
            per_model_costs[mk]["calls"] += 1
            per_model_costs[mk]["prompt_tokens"] += result.cost.prompt_tokens
            per_model_costs[mk]["completion_tokens"] += result.cost.completion_tokens
            per_model_costs[mk]["input_cost"] += result.cost.input_cost
            per_model_costs[mk]["output_cost"] += result.cost.output_cost
            per_model_costs[mk]["total_cost"] += result.cost.total

            conversation += router_output
            resp_text = result.response[:2000] if result.response else ""
            conversation += f"\n\n<information>{resp_text}</information>\n\n"
            turns.append(turn)
            continue

        # Fallback: parse <search> tag from router output
        search = parse_search(router_output, question)
        if search:
            model_name, query = search
            mk = resolve_model_key(model_name)
            if mk:
                result = call_pool_model(mk, question, max_tokens=max_pool_tokens,
                                         temperature=temperature, seed=seed)
                turn.action = "search"
                turn.routed_model = model_name
                turn.routed_query = question
                turn.routed_response = result.response
                turn.cost = result.cost.total
                turn.routing_decision_logic = "router_decides"
                models_called.append(model_name)

                total_pool_pt += result.cost.prompt_tokens
                total_pool_ct += result.cost.completion_tokens
                total_cost += result.cost.total

                if mk not in per_model_costs:
                    per_model_costs[mk] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
                                           "input_cost": 0.0, "output_cost": 0.0, "total_cost": 0.0}
                per_model_costs[mk]["calls"] += 1
                per_model_costs[mk]["prompt_tokens"] += result.cost.prompt_tokens
                per_model_costs[mk]["completion_tokens"] += result.cost.completion_tokens
                per_model_costs[mk]["input_cost"] += result.cost.input_cost
                per_model_costs[mk]["output_cost"] += result.cost.output_cost
                per_model_costs[mk]["total_cost"] += result.cost.total

                conversation += router_output
                resp_text = result.response[:2000] if result.response else ""
                conversation += f"\n\n<information>{resp_text}</information>\n\n"
                turns.append(turn)
                continue

        turns.append(turn)
        break

    if answer is None:
        answer = ""

    em = compute_exact_match(answer, ground_truths)
    f1 = compute_f1(answer, ground_truths)

    turn_dicts = []
    for t in turns:
        td: Dict[str, Any] = {
            "turn_idx": t.turn_idx,
            "action": t.action,
            "router_output": t.router_output,
            "routed_model": t.routed_model,
            "routed_query": t.routed_query,
            "routed_response": t.routed_response,
            "cost": t.cost,
            "router_prompt_tokens": t.router_prompt_tokens,
            "router_completion_tokens": t.router_completion_tokens,
            "routing_decision_logic": t.routing_decision_logic,
        }
        if t.skill_analysis:
            td["skill_analysis"] = {
                "required_skills": [{"skill_id": s.skill_id, "percentage": s.percentage}
                                    for s in t.skill_analysis.required_skills],
                "reasoning": t.skill_analysis.reasoning,
            }
        if t.indicator_fallback_analysis:
            td["indicator_fallback_analysis"] = {
                "required_skills": [{"skill_id": s.skill_id, "percentage": s.percentage}
                                    for s in t.indicator_fallback_analysis.required_skills],
                "reasoning": t.indicator_fallback_analysis.reasoning,
            }
        turn_dicts.append(td)

    return {
        "id": sample["id"],
        "sample_id": sample["sample_id"],
        "question": question,
        "answer": answer,
        "ground_truths": ground_truths,
        "exact_match": em,
        "f1": f1,
        "success": em > 0 or f1 > 0.5,
        "errors": {
            "skill_analysis_parsed_successfully": skill_analysis_parsed,
            "skill_analysis_parse_failures": skill_analysis_failures,
            "parse_errors": parse_errors,
        },
        "num_turns": len(turns),
        "models_called": models_called,
        "tokens": {
            "router_prompt": total_router_pt,
            "router_completion": total_router_ct,
            "pool_prompt": total_pool_pt,
            "pool_completion": total_pool_ct,
            "total": total_router_pt + total_router_ct + total_pool_pt + total_pool_ct,
        },
        "costs": {"total": total_cost},
        "per_model_costs": per_model_costs,
        "turns": turn_dicts,
        "full_trajectory": conversation,
        "timestamp": datetime.now().isoformat(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Test skill-based model routing")
    parser.add_argument("--handbook", required=True, help="Path to RSL handbook JSON")
    parser.add_argument("--dataset", help="HF dataset name (e.g. nq_test_qwen)")
    parser.add_argument("--input-file", help="JSONL file with samples (overrides --dataset)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--router-model", default="qwen2.5-3b-instruct")
    parser.add_argument("--routing-strategy", default="weighted_avg",
                        choices=["router_decides", "weighted_avg"])
    parser.add_argument("--always-use-original-query", action="store_true",
                        help="Use the original question as the pool query")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-pool-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lambda-c", type=float, default=0.01,
                        help="Cost penalty weight for probabilistic model selection")
    parser.add_argument("--distributed-config", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=15,
                        help="Number of parallel workers for sample processing (1=sequential)")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    if args.distributed_config:
        load_distributed_config(args.distributed_config)

    # Seed RNG for reproducible probabilistic routing
    import random as _random
    if args.seed is not None:
        _random.seed(args.seed)

    # Check servers
    status = check_all_servers()
    alive = [k for k, v in status.items() if v]
    logger.info(f"Servers alive: {alive}")

    # Load handbook
    hb = load_handbook(args.handbook)
    skill_catalog_text = extract_skill_catalog_text(hb)
    model_perf_text = extract_model_performance_text(hb)
    model_skill_scores = extract_model_skill_scores(hb)
    model_overall_rates = extract_model_overall_rates(hb)
    skill_indicators = extract_skill_indicators(hb)

    logger.info(f"Handbook: {len(model_skill_scores)} models, "
                f"{sum(len(v) for v in model_skill_scores.values())} total skill scores")
    for mn, ss in model_skill_scores.items():
        logger.info(f"  {mn}: {len(ss)} skills, avg={sum(ss.values())/max(len(ss),1):.3f}")

    # Load dataset
    if args.input_file:
        samples = load_dataset_from_jsonl(args.input_file)
        if args.max_samples:
            samples = samples[: args.max_samples]
    else:
        if not args.dataset:
            raise ValueError("Either --dataset or --input-file is required")
        samples = load_dataset(args.dataset, args.max_samples)

    # Run inference
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def process_sample(i: int, sample: Dict) -> Tuple[int, Dict]:
        """Worker: run inference on one sample, return (index, record)."""
        rec = run_inference(
            sample, hb,
            skill_catalog_text, model_perf_text,
            model_skill_scores, skill_indicators,
            model_overall_rates=model_overall_rates,
            router_model=args.router_model,
            routing_strategy=args.routing_strategy,
            temperature=args.temperature,
            seed=args.seed,
            max_pool_tokens=args.max_pool_tokens,
            lambda_c=args.lambda_c,
            verbose=args.verbose,
        )
        return i, rec

    results_by_idx: Dict[int, Dict] = {}
    t0 = time.time()

    if args.num_workers > 1:
        logger.info(f"Processing {len(samples)} samples with {args.num_workers} workers...")
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = {executor.submit(process_sample, i, s): i for i, s in enumerate(samples)}
            for future in as_completed(futures):
                i, rec = future.result()
                results_by_idx[i] = rec
                logic = rec["turns"][0]["routing_decision_logic"] if rec["turns"] else "?"
                logger.info(f"[{i+1}/{len(samples)}] {rec.get('question', '')[:60]}... "
                            f"EM={rec['exact_match']:.1f} F1={rec['f1']:.2f} "
                            f"model={rec['models_called']} logic={logic}")
        results = [results_by_idx[i] for i in range(len(samples))]
    else:
        results = []
        for i, sample in enumerate(samples):
            logger.info(f"\n[{i+1}/{len(samples)}] {sample['question'][:80]}")
            _, rec = process_sample(i, sample)
            results.append(rec)
            logic = rec["turns"][0]["routing_decision_logic"] if rec["turns"] else "?"
            logger.info(f"  → EM={rec['exact_match']:.1f} F1={rec['f1']:.2f} "
                       f"model={rec['models_called']} logic={logic}")

    total_em = sum(r["exact_match"] for r in results)
    total_f1 = sum(r["f1"] for r in results)
    elapsed = time.time() - t0
    n = len(results)

    # Aggregate costs and tokens from all results
    total_cost = sum(r.get("costs", {}).get("total", 0.0) for r in results)
    total_router_pt = sum(r.get("tokens", {}).get("router_prompt", 0) for r in results)
    total_router_ct = sum(r.get("tokens", {}).get("router_completion", 0) for r in results)
    total_pool_pt = sum(r.get("tokens", {}).get("pool_prompt", 0) for r in results)
    total_pool_ct = sum(r.get("tokens", {}).get("pool_completion", 0) for r in results)
    costs_per_sample = [r.get("costs", {}).get("total", 0.0) for r in results]

    # Aggregate per_model_costs across samples (calls, tokens, costs)
    per_model_agg: Dict[str, Dict[str, Any]] = {}
    for r in results:
        for model_key, mc in r.get("per_model_costs", {}).items():
            if model_key not in per_model_agg:
                per_model_agg[model_key] = {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "input_cost": 0.0,
                    "output_cost": 0.0,
                    "total_cost": 0.0,
                }
            per_model_agg[model_key]["calls"] += mc.get("calls", 0)
            per_model_agg[model_key]["prompt_tokens"] += mc.get("prompt_tokens", 0)
            per_model_agg[model_key]["completion_tokens"] += mc.get("completion_tokens", 0)
            per_model_agg[model_key]["input_cost"] += mc.get("input_cost", 0.0)
            per_model_agg[model_key]["output_cost"] += mc.get("output_cost", 0.0)
            per_model_agg[model_key]["total_cost"] += mc.get("total_cost", 0.0)

    # Save results
    inf_path = out_dir / "inference_results.jsonl"
    with open(inf_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary = {
        "dataset": args.dataset,
        "handbook": args.handbook,
        "router_model": args.router_model,
        "routing_strategy": args.routing_strategy,
        "temperature": args.temperature,
        "max_pool_tokens": args.max_pool_tokens,
        "num_samples": n,
        "exact_match": round(total_em / n, 4) if n else 0,
        "f1": round(total_f1 / n, 4) if n else 0,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": datetime.now().isoformat(),
        # Aggregate cost & tokens
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_sample_usd": round(total_cost / n, 6) if n else 0,
        "min_cost_per_sample_usd": round(min(costs_per_sample), 6) if costs_per_sample else 0,
        "max_cost_per_sample_usd": round(max(costs_per_sample), 6) if costs_per_sample else 0,
        "tokens": {
            "router_prompt": total_router_pt,
            "router_completion": total_router_ct,
            "pool_prompt": total_pool_pt,
            "pool_completion": total_pool_ct,
            "total": total_router_pt + total_router_ct + total_pool_pt + total_pool_ct,
            "avg_per_sample": round((total_router_pt + total_router_ct + total_pool_pt + total_pool_ct) / n, 1) if n else 0,
        },
        # Per-model: aggregated calls, tokens, costs across all samples
        "per_model_costs": {
            k: {kk: round(vv, 6) if isinstance(vv, float) else vv
                for kk, vv in v.items()}
            for k, v in per_model_agg.items()
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        f"\nResults: EM={summary['exact_match']:.4f} F1={summary['f1']:.4f} "
        f"cost=${summary['total_cost_usd']:.4f} ({n} samples, {elapsed:.0f}s)"
    )
    logger.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()
