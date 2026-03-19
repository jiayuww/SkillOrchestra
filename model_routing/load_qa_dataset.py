"""Shared QA dataset loader for MilaWang/qa_* with pyarrow fallback.

Handles datasets that have incompatible Features metadata (TypeError
"must be called with a dataclass type or instance" in Features.from_dict).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def load_qa_dataset_raw(
    dataset_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load QA dataset from HuggingFace (MilaWang/qa_validation_qwen or qa_test_qwen).

    Returns list of dicts with keys: id, question, golden_answers.
    Uses pyarrow fallback when datasets Features parsing fails.
    """
    is_validation = "validation" in dataset_name
    source = "MilaWang/qa_validation_qwen" if is_validation else "MilaWang/qa_test_qwen"

    logger.info(f"Loading {source}/{dataset_name} from HuggingFace...")
    try:
        from datasets import load_dataset
        ds = load_dataset(source, dataset_name, split="test")
        samples = []
        for row in ds:
            ga = row.get("golden_answers", row.get("ground_truths", []))
            if hasattr(ga, "tolist"):
                ga = ga.tolist()
            elif not isinstance(ga, list):
                ga = [ga] if ga is not None else []
            samples.append({
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "golden_answers": ga,
            })
    except TypeError as e:
        if "dataclass" in str(e):
            logger.warning(
                f"load_dataset failed ({e}). Falling back to parquet via pyarrow."
            )
            samples = _load_via_parquet(source, dataset_name)
        else:
            raise

    if max_samples:
        samples = samples[:max_samples]
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def _load_via_parquet(source: str, dataset_name: str) -> List[Dict[str, Any]]:
    """Load dataset by fetching parquet directly. Bypasses datasets Features parsing."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    filename = f"{dataset_name}/test-00000-of-00001.parquet"
    path = hf_hub_download(
        repo_id=source,
        filename=filename,
        repo_type="dataset",
    )
    table = pq.read_table(path)
    df = table.to_pandas()

    samples = []
    for _, row in df.iterrows():
        ga = row.get("golden_answers")
        if hasattr(ga, "tolist"):
            ga = ga.tolist()
        elif not isinstance(ga, list):
            ga = [ga] if ga is not None else []
        samples.append({
            "id": str(row.get("id", "")),
            "question": str(row.get("question", "")),
            "golden_answers": ga,
        })
    return samples
