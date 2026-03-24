"""Shared dataset loader for QA and math datasets with pyarrow fallback.

Handles datasets that have incompatible Features metadata (TypeError
"must be called with a dataclass type or instance" in Features.from_dict).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _is_math_dataset(dataset_name: str) -> bool:
    name = dataset_name.lower()
    return (
        name.startswith("math500-")
        or name.startswith("amc-")
        or name.startswith("aime-")
    )


def load_qa_dataset_raw(
    dataset_name: str,
    max_samples: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Load QA or math dataset from HuggingFace.

    Returns list of dicts with keys: id, question, golden_answers.
    Uses pyarrow fallback when datasets Features parsing fails.
    """
    is_math = _is_math_dataset(dataset_name)
    if is_math:
        source = f"MilaWang/{dataset_name}"
        source_subset = None
        logger.info(f"Loading {source} from HuggingFace...")
    else:
        is_validation = "validation" in dataset_name
        source = "MilaWang/qa_validation_qwen" if is_validation else "MilaWang/qa_test_qwen"
        source_subset = dataset_name
        logger.info(f"Loading {source}/{dataset_name} from HuggingFace...")
    try:
        from datasets import load_dataset
        if source_subset is None:
            ds = load_dataset(source, split="test")
        else:
            ds = load_dataset(source, source_subset, split="test")
        samples = []
        for row in ds:
            ga = row.get("golden_answers", row.get("ground_truths", row.get("answer", [])))
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
            if is_math:
                samples = _load_math_via_parquet(source)
            else:
                samples = _load_qa_via_parquet(source, dataset_name)
        else:
            raise

    if max_samples:
        samples = samples[:max_samples]
    logger.info(f"Loaded {len(samples)} samples")
    return samples


def _load_qa_via_parquet(source: str, dataset_name: str) -> List[Dict[str, Any]]:
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
        ga = row.get("golden_answers", row.get("ground_truths", row.get("answer", [])))
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


def _load_math_via_parquet(source: str) -> List[Dict[str, Any]]:
    """Load math dataset by finding a parquet file in the dataset repo."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    files = api.list_repo_files(repo_id=source, repo_type="dataset")
    parquet_files = [f for f in files if f.endswith(".parquet")]
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {source}")

    # Prefer test shard naming when available.
    chosen = next((f for f in parquet_files if "/test-" in f or f.startswith("test-")), parquet_files[0])
    path = hf_hub_download(repo_id=source, filename=chosen, repo_type="dataset")
    table = pq.read_table(path)
    df = table.to_pandas()

    samples = []
    for _, row in df.iterrows():
        ga = row.get("golden_answers", row.get("ground_truths", row.get("answer", [])))
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
