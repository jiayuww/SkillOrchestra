"""
Generate candidate handbooks at different granularity levels.

Supports two strategies:
1. Per-mode depth-based: each mode can have a different depth level,
   generating the cross product of per-mode depth choices. (This is used by SkillOrchestra.)
2. Legacy coarse buckets: leaf / parent / root.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from ..core.handbook import SkillHandbook

logger = logging.getLogger(__name__)


@dataclass
class CandidateHandbook:
    """A candidate handbook with metadata about its granularity."""

    name: str
    handbook: SkillHandbook
    granularity: str
    depth_map: Dict[str, int] = field(default_factory=dict)
    description: str = ""


# ---------------------------------------------------------------------------
# Per-mode depth helpers
# ---------------------------------------------------------------------------

def _compute_max_depth(handbook: SkillHandbook, mode: str) -> int:
    """Compute the maximum depth of the skill hierarchy for a mode.

    depth 0 = root skills only (no children exist, or we ignore children).
    Returns the actual max nesting level.
    """
    mode_skills = handbook.get_skills_for_mode(mode)
    if not mode_skills:
        return 0

    roots = [s for s in mode_skills if s.parent_skill_id is None]
    if not roots:
        return 0

    max_d = 0
    frontier = list(roots)
    depth = 0
    while frontier:
        next_frontier = []
        for parent in frontier:
            children = parent.get_children(handbook.skills)
            children_in_mode = [c for c in children if c.mode == mode]
            next_frontier.extend(children_in_mode)
        if next_frontier:
            depth += 1
            max_d = depth
        frontier = next_frontier

    return max_d


def _compute_category_max_depth(handbook: SkillHandbook, mode: str) -> int:
    """DEPRECATED: Use _compute_path_max_depth for tree-based depth."""
    return _compute_path_max_depth(handbook, mode)


def _compute_path_max_depth(handbook: SkillHandbook, mode: str) -> int:
    """Compute max tree depth when hierarchy is flat. Mode is root.

    Depth = path segments within mode (strips mode prefix from skill_id).
    Returns max depth such that depth in [0, max_depth] gives distinct skill sets.
    """
    return handbook.max_path_depth(mode)


def compute_mode_max_depths(handbook: SkillHandbook) -> Dict[str, int]:
    """Compute max depth for every mode in the handbook.

    Uses hierarchy depth when skills have parent-child structure.
    When hierarchy is flat (all roots), uses path-based depth from dots in skill_id.
    """
    result: Dict[str, int] = {}
    for mode in handbook.all_modes:
        h_depth = _compute_max_depth(handbook, mode)
        if h_depth > 0:
            result[mode] = h_depth
        else:
            result[mode] = _compute_path_max_depth(handbook, mode)
    return result


def _depth_map_to_name(depth_map: Dict[str, int]) -> str:
    """Convert a depth map to a short name: 'search2_code1_answer0'."""
    return "_".join(f"{m}{d}" for m, d in sorted(depth_map.items()))


def _skills_for_depth_map(
    handbook: SkillHandbook,
    depth_map: Dict[str, int],
    mode_depth_types: Optional[Dict[str, str]] = None,
) -> Set[str]:
    """Collect skill IDs for a given per-mode depth configuration.

    When mode_depth_types[mode] == 'category', uses category-based granularity
    (for flat skill sets). Otherwise uses hierarchy-based get_skills_at_depth.
    """
    if mode_depth_types is None:
        mode_depth_types = {}
    skill_ids: Set[str] = set()
    for mode, depth in depth_map.items():
        if mode_depth_types.get(mode) == "category":
            skills = handbook.get_skills_at_category_depth(mode, depth)
        else:
            skills = handbook.get_skills_at_depth(mode, depth)
        skill_ids.update(s.skill_id for s in skills)
    return skill_ids


# ---------------------------------------------------------------------------
# Per-mode depth-based candidate generation
# ---------------------------------------------------------------------------

def generate_depth_candidates(
    full_handbook: SkillHandbook,
    mode_max_depths: Optional[Dict[str, int]] = None,
) -> List[CandidateHandbook]:
    """Generate candidate handbooks by varying per-mode depth levels.

    For each mode, depth ranges from 0 (root-only) up to its max depth
    (finest granularity). Candidates are the cross product of per-mode
    depth choices, deduplicated by effective skill set.

    Args:
        full_handbook: The learned full handbook H*
        mode_max_depths: Optional override for max depth per mode.
            If not provided, computed automatically from the hierarchy.

    Returns:
        List of CandidateHandbook objects (deduplicated)
    """
    if mode_max_depths is None:
        mode_max_depths = compute_mode_max_depths(full_handbook)

    modes = sorted(mode_max_depths.keys())
    if not modes:
        logger.warning("No modes found in handbook")
        return []

    # Determine per-mode depth type: hierarchy vs category (for flat skill sets)
    mode_depth_types: Dict[str, str] = {}
    for m in modes:
        mode_depth_types[m] = (
            "category"
            if _compute_max_depth(full_handbook, m) == 0
            and _compute_category_max_depth(full_handbook, m) > 0
            else "hierarchy"
        )

    depth_ranges = [range(mode_max_depths[m] + 1) for m in modes]

    seen_skill_sets: Dict[frozenset, str] = {}
    candidates: List[CandidateHandbook] = []

    for combo in itertools.product(*depth_ranges):
        depth_map = dict(zip(modes, combo))
        skill_ids = _skills_for_depth_map(
            full_handbook, depth_map, mode_depth_types=mode_depth_types
        )
        frozen = frozenset(skill_ids)

        if frozen in seen_skill_sets:
            logger.debug(
                f"Pruning {_depth_map_to_name(depth_map)} -- "
                f"same skills as {seen_skill_sets[frozen]}"
            )
            continue

        name = _depth_map_to_name(depth_map)
        seen_skill_sets[frozen] = name

        sub = full_handbook.subgraph(skill_ids=skill_ids)
        skills_per_mode = {
            m: len(sub.get_skills_for_mode(m)) for m in modes
        }

        candidates.append(CandidateHandbook(
            name=name,
            handbook=sub,
            granularity="depth_map",
            depth_map=depth_map,
            description=f"Per-mode depths: {depth_map}, skills/mode: {skills_per_mode}",
        ))

    candidates.sort(key=lambda c: sum(c.depth_map.values()))

    n_combos = len(list(itertools.product(*depth_ranges)))
    logger.info(
        f"Generated {len(candidates)} depth-based candidates "
        f"from {n_combos} combinations "
        f"(modes: {modes}, max_depths: {mode_max_depths}, depth_types: {mode_depth_types})"
    )
    for c in candidates:
        logger.info(f"  {c.name}: {c.handbook.num_skills} skills")

    return candidates


# ---------------------------------------------------------------------------
# Legacy coarse-bucket generation (backward compatible)
# ---------------------------------------------------------------------------

def generate_candidates(
    full_handbook: SkillHandbook,
    include_ablations: bool = True,
) -> List[CandidateHandbook]:
    """Generate candidate handbooks at coarse granularity levels.

    Candidates:
    1. Full (leaf): all skills at finest granularity
    2. Parent-only: collapse leaf skills, keep only parents
    3. Root-only: one skill per top-level category per mode
    4. Ablations: no mode metadata, no cost info, etc.
    """
    candidates = []

    candidates.append(CandidateHandbook(
        name="full",
        handbook=full_handbook.subgraph(),
        granularity="leaf",
        description="Full handbook with all skills at finest granularity",
    ))

    parent_skills = _get_parent_only_skills(full_handbook)
    if parent_skills != set(full_handbook.skills.keys()):
        candidates.append(CandidateHandbook(
            name="parent_only",
            handbook=full_handbook.subgraph(skill_ids=parent_skills),
            granularity="parent",
            description="Only parent-level skills (children collapsed)",
        ))

    root_skills = _get_root_only_skills(full_handbook)
    if root_skills != parent_skills:
        candidates.append(CandidateHandbook(
            name="root_only",
            handbook=full_handbook.subgraph(skill_ids=root_skills),
            granularity="root",
            description="Only root-level skills (coarsest granularity)",
        ))

    if include_ablations:
        no_insights = full_handbook.subgraph()
        for mode_meta in no_insights.modes.values():
            mode_meta.insights = []
        candidates.append(CandidateHandbook(
            name="no_mode_insights",
            handbook=no_insights,
            granularity="leaf",
            description="Full skills but no mode-level routing insights",
        ))

    logger.info(f"Generated {len(candidates)} candidate handbooks")
    for c in candidates:
        logger.info(f"  {c.name}: {c.handbook.num_skills} skills ({c.granularity})")

    return candidates


def _get_parent_only_skills(handbook: SkillHandbook) -> Set[str]:
    """Get skill IDs: keep parents, drop leaf children."""
    parent_ids = set()
    for sid, skill in handbook.skills.items():
        children = skill.get_children(handbook.skills)
        if children:
            parent_ids.add(sid)
        elif skill.parent_skill_id is None:
            parent_ids.add(sid)
    return parent_ids


def _get_root_only_skills(handbook: SkillHandbook) -> Set[str]:
    """Get only root-level skill IDs (no parent)."""
    return {sid for sid, s in handbook.skills.items() if s.parent_skill_id is None}
