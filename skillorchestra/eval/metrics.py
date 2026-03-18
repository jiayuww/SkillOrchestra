"""
QA evaluation metrics: exact match and token-level F1.
Supports both general QA and math-specific answer matching.
"""
import string
import re
from typing import List, Union, Optional

import sympy
from sympy.parsing import sympy_parser
from pylatexenc import latex2text


def normalize_answer(s: str) -> str:
    """
    Normalize answer for exact match comparison.
    - Convert to lowercase
    - Remove punctuation
    - Remove articles (a, an, the)
    - Collapse whitespace
    - Normalize common name spelling variants
    """
    if not s:
        return ""

    s = s.lower()

    # Normalize common name spelling variants
    name_variants = {
        'jakob': 'jacob',
        'josef': 'joseph',
        'johann': 'johan',
        'johannes': 'johann',
        'friedrich': 'frederick',
        'wilhelm': 'william',
        'heinrich': 'henry',
        'karl': 'carl',
        'franz': 'francis',
        'ludwig': 'louis',
        'adolf': 'adolph',
        'rudolf': 'rudolph',
        'philipp': 'philip',
    }

    # Apply name variant normalization word by word
    words = s.split()
    normalized_words = []
    for word in words:
        word_clean = word.translate(str.maketrans("", "", string.punctuation))
        if word_clean in name_variants:
            normalized_word = word.replace(word_clean, name_variants[word_clean])
            normalized_words.append(normalized_word)
        else:
            normalized_words.append(word)
    s = ' '.join(normalized_words)

    s = s.translate(str.maketrans("", "", string.punctuation))

    s = re.sub(r'\b(a|an|the)\b', ' ', s)

    s = ' '.join(s.split())

    return s.strip()


def extract_boxed_answer(solution_str: str) -> Optional[str]:
    """Extract the content from the last \\boxed{} notation in the string."""
    idx = solution_str.rfind("\\boxed")
    if idx < 0:
        return None

    if solution_str[idx:idx+7] == "\\boxed{":
        i = idx + 7
        num_braces_open = 1
        right_brace_idx = None

        while i < len(solution_str):
            if solution_str[i] == "{":
                num_braces_open += 1
            elif solution_str[i] == "}":
                num_braces_open -= 1
                if num_braces_open == 0:
                    right_brace_idx = i
                    break
            i += 1

        if right_brace_idx is None:
            return None

        return solution_str[idx+7:right_brace_idx].strip()

    elif solution_str[idx:idx+7] == "\\boxed ":
        content = solution_str[idx+7:].split("$")[0].strip()
        return content

    return None


def extract_answer_from_tags(text: str) -> Optional[str]:
    """Extract answer from <answer></answer> tags."""
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = list(re.finditer(answer_pattern, text, re.DOTALL))
    if matches:
        answer_content = matches[-1].group(1).strip()
        pattern_match = re.search(
            r'(?:the\s+)?(?:answer|result|value|solution)(?:\s+is|:|=)\s*(.*?)(?:\s*$|\.)',
            answer_content,
            re.IGNORECASE
        )
        if pattern_match:
            return pattern_match.group(1).strip()
        return answer_content
    return None


def extract_solution(solution_str: str) -> Optional[str]:
    """
    Extract the final answer from a solution string.
    Priority: \\boxed{} > <answer></answer> > "Final answer:" patterns
    """
    boxed_answer = extract_boxed_answer(solution_str)
    if boxed_answer:
        return boxed_answer

    answer_from_tags = extract_answer_from_tags(solution_str)
    if answer_from_tags:
        return answer_from_tags

    pattern_match = re.search(
        r'(?:Final answer:|Answer:|The answer is:?)\s*(.*?)(?:\n|$)',
        solution_str
    )
    if pattern_match:
        return pattern_match.group(1).strip()

    return None


def _is_math_answer(text: str) -> bool:
    """Heuristic to detect if an answer is mathematical."""
    math_indicators = [
        r'\\boxed',
        r'\\frac',
        r'\\sqrt',
        r'\^',
        r'[0-9]+\s*/\s*[0-9]+',
        r'[0-9]+\.[0-9]+',
        r'[+\-*/=]',
    ]

    text_lower = text.lower()
    for pattern in math_indicators:
        if re.search(pattern, text):
            return True

    math_chars = set('0123456789+-*/=()[]{}^√π∞.,')
    if len(text) > 0 and sum(1 for c in text if c in math_chars) / len(text) > 0.5:
        return True

    return False


def normalize_math_answer(expr: str) -> Optional[str]:
    """
    Normalize math answer for comparison.
    Converts LaTeX to text format that sympy can parse.
    """
    if not expr:
        return None

    expr = expr.strip()

    m = re.search(r"^\\text\{(?P<text>.+?)\}$", expr)
    if m is not None:
        expr = m.group("text")

    expr = expr.replace("\\$", "").replace("$", "")
    expr = expr.replace("\\%", "%").replace("%", "")

    expr = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', expr)
    expr = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', expr)

    try:
        expr = latex2text.LatexNodes2Text().latex_to_text(expr)
    except Exception:
        pass

    expr = re.sub(r'√\(([^)]+)\)', r'sqrt(\1)', expr)
    expr = re.sub(r'√([0-9.]+)', r'sqrt(\1)', expr)
    expr = expr.replace('π', 'pi')
    expr = expr.replace('×', '*')
    expr = expr.replace('·', '*')
    expr = expr.replace('÷', '/')

    expr = expr.replace(" ", "")

    return expr.lower()


def are_equal_math(ground_truth: str, prediction: str) -> bool:
    """
    Check if two math expressions are equivalent using sympy.
    """
    gt_norm = normalize_math_answer(ground_truth)
    pred_norm = normalize_math_answer(prediction)

    if gt_norm == pred_norm:
        return True

    try:
        gt_parsed = None
        pred_parsed = None

        gt_expr = gt_norm.replace("^", "**") if gt_norm else None
        pred_expr = pred_norm.replace("^", "**") if pred_norm else None

        if gt_expr:
            try:
                gt_parsed = sympy_parser.parse_expr(
                    gt_expr,
                    transformations=(
                        sympy_parser.standard_transformations +
                        (sympy_parser.implicit_multiplication_application,)
                    ),
                )
            except Exception:
                pass

        if pred_expr:
            try:
                pred_parsed = sympy_parser.parse_expr(
                    pred_expr,
                    transformations=(
                        sympy_parser.standard_transformations +
                        (sympy_parser.implicit_multiplication_application,)
                    ),
                )
            except Exception:
                pass

        if gt_parsed is not None and pred_parsed is not None:
            try:
                diff = sympy.simplify(gt_parsed - pred_parsed)
                if diff == 0:
                    return True
            except Exception:
                pass

        try:
            if gt_parsed is not None:
                gt_eval = float(gt_parsed.evalf())
            else:
                gt_eval = None

            if pred_parsed is not None:
                pred_eval = float(pred_parsed.evalf())
            else:
                pred_eval = None

            if gt_eval is not None and pred_eval is not None:
                if abs(gt_eval - pred_eval) < 1e-10:
                    return True
        except Exception:
            pass

    except Exception:
        pass

    return False


def compute_exact_match(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    Compute exact match score.
    Supports both general QA (string matching) and math answers (symbolic equivalence).

    For math answers:
    - Extracts answers from \\boxed{} notation or <answer></answer> tags
    - Uses sympy for symbolic equivalence checking (if available)
    - Falls back to normalized string matching

    For general QA:
    - Uses normalized string matching (backward compatible)

    Args:
        prediction: Model's predicted answer (may contain reasoning, boxed notation, etc.)
        ground_truth: Ground truth answer(s) - can be string or list of strings

    Returns:
        1.0 if any ground truth matches prediction, 0.0 otherwise
    """
    if not prediction:
        return 0.0

    # Handle both single string and list of ground truths
    if isinstance(ground_truth, str):
        ground_truths = [ground_truth]
    elif isinstance(ground_truth, (int, float)):
        ground_truths = [str(ground_truth)]
    else:
        ground_truths = ground_truth if ground_truth else []

    if not ground_truths:
        return 0.0

    # Extract answer from prediction (handles boxed notation, answer tags, etc.)
    extracted_pred = extract_solution(prediction)
    if extracted_pred:
        prediction = extracted_pred

    # Check if this looks like a math answer
    is_math = _is_math_answer(prediction) or any(_is_math_answer(gt) for gt in ground_truths)

    if is_math:
        # Math answer matching
        for gt in ground_truths:
            gt_str = str(gt)

            # Extract from ground truth if needed
            extracted_gt = extract_boxed_answer(gt_str)
            if extracted_gt:
                gt_str = extracted_gt

            # Try math equivalence
            if are_equal_math(gt_str, prediction):
                return 1.0

            # Fallback: normalized string comparison
            gt_norm = normalize_math_answer(gt_str)
            pred_norm = normalize_math_answer(prediction)
            if gt_norm and pred_norm and gt_norm == pred_norm:
                return 1.0

    # General QA matching
    pred_normalized = normalize_answer(prediction)

    for gt in ground_truths:
        gt_str = str(gt)
        gt_normalized = normalize_answer(gt_str)
        if gt_normalized and gt_normalized in pred_normalized:
            return 1.0
        if pred_normalized and pred_normalized in gt_normalized:
            return 1.0

    return 0.0


def compute_f1(prediction: str, ground_truth: Union[str, List[str]]) -> float:
    """
    Compute token-level F1 score.
    Returns max F1 across all ground truths.

    Args:
        prediction: Model's predicted answer
        ground_truth: Ground truth answer(s) - can be string or list of strings

    Returns:
        Maximum F1 score across all ground truths
    """
    if not prediction:
        return 0.0

    pred_tokens = set(normalize_answer(prediction).split())

    if isinstance(ground_truth, str):
        ground_truths = [ground_truth]
    else:
        ground_truths = ground_truth if ground_truth else []

    max_f1 = 0.0
    for gt in ground_truths:
        gt_tokens = set(normalize_answer(gt).split())

        if not pred_tokens or not gt_tokens:
            continue

        common = pred_tokens & gt_tokens
        if not common:
            continue

        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(gt_tokens)
        f1 = 2 * precision * recall / (precision + recall)
        max_f1 = max(max_f1, f1)

    return max_f1
