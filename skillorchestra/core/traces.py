"""
Execution trace data model for SkillOrchestra.

Defines the unified input format for the learning pipeline:
- ExecutionStep: a single step in a trajectory
- ExecutionTrace: a full trajectory tau for a query with one agent configuration
- ExplorationBundle: multiple trajectories per query
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExecutionStep:
    """A single step in an execution trajectory."""

    step_idx: int = 0
    mode: str = ""
    agent_id: str = ""
    model_name: str = ""
    tools_used: List[str] = field(default_factory=list)
    input_text: str = ""
    output_text: str = ""
    observation: str = ""
    cost_usd: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_idx": self.step_idx,
            "mode": self.mode,
            "agent_id": self.agent_id,
            "model_name": self.model_name,
            "tools_used": self.tools_used,
            "input_text": self.input_text,
            "output_text": self.output_text,
            "observation": self.observation,
            "cost_usd": self.cost_usd,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "latency_s": self.latency_s,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExecutionStep:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


@dataclass
class ExecutionTrace:
    """A full execution trajectory for a single query."""

    query_id: str = ""
    query: str = ""
    ground_truths: List[str] = field(default_factory=list)
    steps: List[ExecutionStep] = field(default_factory=list)
    final_answer: Optional[str] = None
    task_success: bool = False
    total_cost_usd: float = 0.0

    varied_mode: str = ""
    varied_agent_id: str = ""

    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_successful(self) -> bool:
        return self.task_success

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def get_steps_for_mode(self, mode: str) -> List[ExecutionStep]:
        return [s for s in self.steps if s.mode == mode]

    def get_agents_used(self) -> Dict[str, List[str]]:
        """Get mapping of mode -> list of agent_ids used in this trace."""
        result: Dict[str, List[str]] = {}
        for step in self.steps:
            if step.mode not in result:
                result[step.mode] = []
            if step.agent_id not in result[step.mode]:
                result[step.mode].append(step.agent_id)
        return result

    def get_cost_by_mode(self) -> Dict[str, float]:
        """Get total cost per mode."""
        costs: Dict[str, float] = {}
        for step in self.steps:
            costs[step.mode] = costs.get(step.mode, 0.0) + step.cost_usd
        return costs

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "ground_truths": self.ground_truths,
            "steps": [s.to_dict() for s in self.steps],
            "final_answer": self.final_answer,
            "task_success": self.task_success,
            "total_cost_usd": self.total_cost_usd,
            "varied_mode": self.varied_mode,
            "varied_agent_id": self.varied_agent_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExecutionTrace:
        steps = [ExecutionStep.from_dict(s) for s in d.get("steps", [])]
        return cls(
            query_id=d.get("query_id", ""),
            query=d.get("query", ""),
            ground_truths=d.get("ground_truths", []),
            steps=steps,
            final_answer=d.get("final_answer"),
            task_success=d.get("task_success", False),
            total_cost_usd=d.get("total_cost_usd", 0.0),
            varied_mode=d.get("varied_mode", ""),
            varied_agent_id=d.get("varied_agent_id", ""),
            metadata=d.get("metadata", {}),
        )


@dataclass
class ExplorationBundle:
    """Multiple trajectories for the same query, varying agent choices."""

    query_id: str = ""
    query: str = ""
    ground_truths: List[str] = field(default_factory=list)
    trajectories: List[ExecutionTrace] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_trajectories(self) -> int:
        return len(self.trajectories)

    @property
    def any_successful(self) -> bool:
        return any(t.task_success for t in self.trajectories)

    @property
    def oracle_accuracy(self) -> float:
        """1.0 if any trajectory succeeded, 0.0 otherwise."""
        return 1.0 if self.any_successful else 0.0

    def get_trajectories_for_mode(self, mode: str) -> List[ExecutionTrace]:
        """Get trajectories where a specific mode's agent was varied."""
        return [t for t in self.trajectories if t.varied_mode == mode]

    def get_successful_traces(self, mode: Optional[str] = None) -> List[ExecutionTrace]:
        """Get successful trajectories, optionally filtered by varied mode."""
        traces = self.trajectories if mode is None else self.get_trajectories_for_mode(mode)
        return [t for t in traces if t.task_success]

    def get_failed_traces(self, mode: Optional[str] = None) -> List[ExecutionTrace]:
        """Get failed trajectories, optionally filtered by varied mode."""
        traces = self.trajectories if mode is None else self.get_trajectories_for_mode(mode)
        return [t for t in traces if not t.task_success]

    def get_contrastive_pairs(self, mode: str) -> List[tuple]:
        successes = self.get_successful_traces(mode)
        failures = self.get_failed_traces(mode)
        pairs = []
        for pos in successes:
            for neg in failures:
                pairs.append((pos, neg))
        return pairs

    def get_modes_explored(self) -> List[str]:
        modes = set()
        for t in self.trajectories:
            if t.varied_mode:
                modes.add(t.varied_mode)
        return sorted(modes)

    def get_agents_for_mode(self, mode: str) -> Dict[str, bool]:
        result: Dict[str, bool] = {}
        for t in self.get_trajectories_for_mode(mode):
            result[t.varied_agent_id] = t.task_success
        return result

    def get_best_agent_for_mode(self, mode: str) -> Optional[str]:
        agents = self.get_agents_for_mode(mode)
        successful = [a for a, s in agents.items() if s]
        if not successful:
            return None
        return successful[0]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "query": self.query,
            "ground_truths": self.ground_truths,
            "trajectories": [t.to_dict() for t in self.trajectories],
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExplorationBundle:
        trajectories = [
            ExecutionTrace.from_dict(t) for t in d.get("trajectories", [])
        ]
        return cls(
            query_id=d.get("query_id", ""),
            query=d.get("query", ""),
            ground_truths=d.get("ground_truths", []),
            trajectories=trajectories,
            metadata=d.get("metadata", {}),
        )
