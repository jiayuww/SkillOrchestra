"""
Handbook Store: versioned storage for handbooks and experiments.

Directory layout:
    store_root/
      <experiment_name>/
        learned/
          handbook_full.json          # H* from learning pipeline
        candidates/
          search2_code1_answer0.json  # per-mode depth candidates
          search1_code0_answer0.json
          ...
        selected/
          <orchestrator_name>.json    # best handbook per orchestrator
        evaluation/
          results.json                # evaluation results for all candidates
          live_results.json           # live evaluation results
        snapshots/
          handbook_baseline.json
          handbook_discovery.json
          ...
        learning_log.json
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.handbook import SkillHandbook
from .candidates import CandidateHandbook

logger = logging.getLogger(__name__)


class HandbookStore:
    """Manages versioned storage of handbooks across experiments."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Experiment management
    # ------------------------------------------------------------------

    def experiment_dir(self, experiment_name: str) -> Path:
        return self.root_dir / experiment_name

    def list_experiments(self) -> List[str]:
        """List all experiment names."""
        if not self.root_dir.exists():
            return []
        return sorted(
            d.name
            for d in self.root_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        )

    def create_experiment(self, experiment_name: str) -> Path:
        """Create directories for a new experiment."""
        exp_dir = self.experiment_dir(experiment_name)
        for sub in ["learned", "candidates", "selected", "evaluation", "snapshots"]:
            (exp_dir / sub).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created experiment: {experiment_name} at {exp_dir}")
        return exp_dir

    # ------------------------------------------------------------------
    # Learned handbook (H*)
    # ------------------------------------------------------------------

    def save_learned(
        self, handbook: SkillHandbook, experiment_name: str
    ) -> Path:
        """Save the full learned handbook."""
        exp_dir = self.create_experiment(experiment_name)
        path = exp_dir / "learned" / "handbook_full.json"
        handbook.save(path)
        return path

    def load_learned(self, experiment_name: str) -> SkillHandbook:
        """Load the full learned handbook."""
        path = self.experiment_dir(experiment_name) / "learned" / "handbook_full.json"
        return SkillHandbook.load(path)

    # ------------------------------------------------------------------
    # Candidates
    # ------------------------------------------------------------------

    def save_candidate(
        self, candidate: CandidateHandbook, experiment_name: str
    ) -> Path:
        """Save a candidate handbook."""
        exp_dir = self.create_experiment(experiment_name)
        path = exp_dir / "candidates" / f"{candidate.name}.json"
        candidate.handbook.save(path)

        meta_path = exp_dir / "candidates" / f"{candidate.name}.meta.json"
        meta = {
            "name": candidate.name,
            "granularity": candidate.granularity,
            "depth_map": candidate.depth_map,
            "description": candidate.description,
            "num_skills": candidate.handbook.num_skills,
            "saved_at": datetime.now().isoformat(),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        return path

    def save_all_candidates(
        self, candidates: List[CandidateHandbook], experiment_name: str
    ) -> List[Path]:
        """Save all candidates for an experiment."""
        paths = []
        for c in candidates:
            p = self.save_candidate(c, experiment_name)
            paths.append(p)
        logger.info(f"Saved {len(candidates)} candidates for {experiment_name}")
        return paths

    def load_candidate(
        self, candidate_name: str, experiment_name: str
    ) -> CandidateHandbook:
        """Load a candidate handbook by name."""
        exp_dir = self.experiment_dir(experiment_name)
        hb_path = exp_dir / "candidates" / f"{candidate_name}.json"
        meta_path = exp_dir / "candidates" / f"{candidate_name}.meta.json"

        handbook = SkillHandbook.load(hb_path)

        meta = {}
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)

        return CandidateHandbook(
            name=meta.get("name", candidate_name),
            handbook=handbook,
            granularity=meta.get("granularity", "unknown"),
            depth_map=meta.get("depth_map", {}),
            description=meta.get("description", ""),
        )

    def list_candidates(self, experiment_name: str) -> List[str]:
        """List candidate names for an experiment."""
        cand_dir = self.experiment_dir(experiment_name) / "candidates"
        if not cand_dir.exists():
            return []
        return sorted(
            p.stem
            for p in cand_dir.glob("*.json")
            if not p.name.endswith(".meta.json")
        )

    # ------------------------------------------------------------------
    # Selected handbooks (per-orchestrator)
    # ------------------------------------------------------------------

    def save_selected(
        self,
        handbook: SkillHandbook,
        orchestrator_name: str,
        experiment_name: str,
        eval_result: Optional[Dict[str, Any]] = None,
    ) -> Path:
        """Save the selected handbook for a specific orchestrator."""
        exp_dir = self.create_experiment(experiment_name)
        path = exp_dir / "selected" / f"{orchestrator_name}.json"
        handbook.save(path)

        if eval_result:
            meta_path = exp_dir / "selected" / f"{orchestrator_name}.meta.json"
            with open(meta_path, "w") as f:
                json.dump(eval_result, f, indent=2)

        logger.info(f"Saved selected handbook for orchestrator={orchestrator_name}")
        return path

    def load_selected(
        self, orchestrator_name: str, experiment_name: str
    ) -> SkillHandbook:
        """Load the selected handbook for an orchestrator."""
        path = self.experiment_dir(experiment_name) / "selected" / f"{orchestrator_name}.json"
        return SkillHandbook.load(path)

    def list_selected(self, experiment_name: str) -> List[str]:
        """List orchestrator names that have a selected handbook."""
        sel_dir = self.experiment_dir(experiment_name) / "selected"
        if not sel_dir.exists():
            return []
        return sorted(
            p.stem
            for p in sel_dir.glob("*.json")
            if not p.name.endswith(".meta.json")
        )

    # ------------------------------------------------------------------
    # Evaluation results
    # ------------------------------------------------------------------

    def save_evaluation_results(
        self,
        results: List[Dict[str, Any]],
        experiment_name: str,
        filename: str = "results.json",
    ) -> Path:
        """Save evaluation results."""
        exp_dir = self.create_experiment(experiment_name)
        path = exp_dir / "evaluation" / filename
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Saved evaluation results to {path}")
        return path

    def load_evaluation_results(
        self, experiment_name: str, filename: str = "results.json"
    ) -> List[Dict[str, Any]]:
        """Load evaluation results."""
        path = self.experiment_dir(experiment_name) / "evaluation" / filename
        with open(path) as f:
            return json.load(f)

    # ------------------------------------------------------------------
    # Snapshots (learning pipeline phase snapshots)
    # ------------------------------------------------------------------

    def save_snapshot(
        self,
        handbook: SkillHandbook,
        experiment_name: str,
        event: str,
    ) -> Path:
        """Save a learning pipeline snapshot."""
        exp_dir = self.create_experiment(experiment_name)
        path = exp_dir / "snapshots" / f"handbook_{event}.json"
        handbook.save(path)
        return path

    # ------------------------------------------------------------------
    # Learning log
    # ------------------------------------------------------------------

    def save_learning_log(
        self,
        log_data: Dict[str, Any],
        experiment_name: str,
    ) -> Path:
        """Save the learning pipeline log."""
        exp_dir = self.create_experiment(experiment_name)
        path = exp_dir / "learning_log.json"
        with open(path, "w") as f:
            json.dump(log_data, f, indent=2)
        return path

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def experiment_summary(self, experiment_name: str) -> Dict[str, Any]:
        """Get a summary of an experiment's contents."""
        exp_dir = self.experiment_dir(experiment_name)
        if not exp_dir.exists():
            return {"error": f"Experiment '{experiment_name}' not found"}

        learned_exists = (exp_dir / "learned" / "handbook_full.json").exists()
        candidates = self.list_candidates(experiment_name)
        selected = self.list_selected(experiment_name)

        snapshots = []
        snap_dir = exp_dir / "snapshots"
        if snap_dir.exists():
            snapshots = sorted(p.stem for p in snap_dir.glob("*.json"))

        return {
            "experiment_name": experiment_name,
            "learned_handbook": learned_exists,
            "num_candidates": len(candidates),
            "candidates": candidates,
            "selected_orchestrators": selected,
            "snapshots": snapshots,
        }
