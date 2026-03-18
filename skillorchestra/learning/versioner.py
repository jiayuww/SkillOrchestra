"""
Handbook versioning: save each version to disk and maintain version history.

- handbooks/handbook_v{version}_c{change_number}_{timestamp}.json
- version_history.json with change metadata
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.handbook import SkillHandbook

logger = logging.getLogger(__name__)


def _normalize_to_semver(version: str) -> str:
    """Convert v0, v1 or 1.0.0 to semver format."""
    v = str(version).strip()
    if v.startswith("v"):
        v = v[1:]
    if "." in v:
        parts = v.split(".")
        if len(parts) >= 3:
            return f"{parts[0]}.{parts[1]}.{parts[2]}"
        if len(parts) == 2:
            return f"{parts[0]}.{parts[1]}.0"
        return f"{parts[0]}.0.0"
    try:
        return f"{int(v)}.0.0" if v.isdigit() else "1.0.0"
    except ValueError:
        return "1.0.0"


def _increment_patch(version: str) -> str:
    """Increment patch version: 1.0.0 -> 1.0.1."""
    v = _normalize_to_semver(version)
    parts = v.split(".")
    if len(parts) == 3:
        try:
            patch = int(parts[2]) + 1
            return f"{parts[0]}.{parts[1]}.{patch}"
        except ValueError:
            pass
    return f"{v}.1"


class HandbookVersioner:
    """Save handbook versions to disk and maintain version history."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.handbooks_dir = self.output_dir / "handbooks"
        self.version_history_file = self.output_dir / "version_history.json"
        self._change_count = 0
        self._load_change_count()

    def _load_change_count(self) -> None:
        """Load current change count from version history."""
        if self.version_history_file.exists():
            try:
                with open(self.version_history_file) as f:
                    history = json.load(f)
                if history:
                    self._change_count = max(
                        e.get("change_number", 0) for e in history
                    )
            except Exception:
                pass

    def save_initial_version(self, handbook: SkillHandbook) -> Optional[Path]:
        """
        Save the handbook before any refinement (baseline 1.0.0).

        Call once at the start of refinement when versioner is enabled.
        Sets handbook.version to 1.0.0 so subsequent changes increment correctly.
        """
        v = _normalize_to_semver(handbook.version)
        if v == "0.0.0" or v.startswith("0."):
            handbook.version = "1.0.0"
        else:
            handbook.version = v
        return self.save_version(handbook, "Initial (before refinement)")

    def save_version(
        self,
        handbook: SkillHandbook,
        change: str,
    ) -> Optional[Path]:
        """
        Save handbook version immediately after a change.

        Args:
            handbook: The handbook to save
            change: Description of the change (e.g. "Merged skill_a + skill_b -> merged_skill")

        Returns:
            Path to the saved handbook file, or None if save failed
        """
        self._change_count += 1
        self.handbooks_dir.mkdir(parents=True, exist_ok=True)

        version_suffix = handbook.version.replace(".", "_")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        handbook_file = self.handbooks_dir / (
            f"handbook_v{version_suffix}_c{self._change_count:03d}_{timestamp}.json"
        )
        try:
            handbook.save(str(handbook_file))
            logger.info(f"  Saved handbook version: {handbook_file.name}")
        except Exception as e:
            logger.error(f"Failed to save handbook version: {e}")
            self._change_count -= 1
            return None

        # Update version history
        version_history = []
        if self.version_history_file.exists():
            try:
                with open(self.version_history_file) as f:
                    version_history = json.load(f)
            except Exception:
                pass

        refinement_entry = {
            "timestamp": datetime.now().isoformat(),
            "version": handbook.version,
            "change_number": self._change_count,
            "change": change,
            "handbook_file": str(handbook_file.relative_to(self.output_dir)),
        }
        version_history.append(refinement_entry)

        try:
            with open(self.version_history_file, "w") as f:
                json.dump(version_history, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save version history: {e}")

        return handbook_file

    def get_history(self) -> List[Dict[str, Any]]:
        """Load version history."""
        if not self.version_history_file.exists():
            return []
        try:
            with open(self.version_history_file) as f:
                return json.load(f)
        except Exception:
            return []

    def log_merge_decision(
        self,
        skill_id_1: str,
        skill_id_2: str,
        applied: bool,
        rationale: str = "",
        alternative_explanation: str = "",
        merged_skill_id: Optional[str] = None,
    ) -> None:
        """Append a merge decision (applied or not) to merge_decisions.jsonl."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        decisions_file = self.output_dir / "merge_decisions.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "skill_id_1": skill_id_1,
            "skill_id_2": skill_id_2,
            "applied": applied,
            "rationale": rationale,
            "alternative_explanation": alternative_explanation,
            "merged_skill_id": merged_skill_id,
        }
        try:
            with open(decisions_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log merge decision: {e}")

    def log_split_decision(
        self,
        skill_id: str,
        applied: bool,
        rationale: str = "",
        proposed_splits: Optional[List[str]] = None,
    ) -> None:
        """Append a split decision (applied or not) to split_decisions.jsonl."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        decisions_file = self.output_dir / "split_decisions.jsonl"
        entry = {
            "timestamp": datetime.now().isoformat(),
            "skill_id": skill_id,
            "applied": applied,
            "rationale": rationale,
            "proposed_splits": proposed_splits or [],
        }
        try:
            with open(decisions_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.warning(f"Failed to log split decision: {e}")
