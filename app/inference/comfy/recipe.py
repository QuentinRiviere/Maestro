"""Recipe contract: how one Maestro model family becomes a ComfyUI graph.

A recipe is code, not a JSON template.  It declares the node types it may
emit, the node whose outputs are the deliverable, and builds the API-format
graph from the validated task parameters.  Graphs are validated locally
before submission because ``/prompt`` only checks shape, not intent.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any, Protocol

from inference.contract import TaskRequest


Graph = dict[str, dict[str, Any]]


@dataclass(frozen=True)
class RecipeInput:
    """A local file the graph needs on the ComfyUI side."""

    role: str
    local_path: str


class ComfyRecipe(Protocol):
    name: str
    model_patterns: tuple[str, ...]
    allowed_node_types: frozenset[str]
    output_node: str

    def collect_inputs(self, request: TaskRequest) -> list[RecipeInput]:
        ...

    def build_graph(self, request: TaskRequest, uploaded: Mapping[str, str]) -> Graph:
        """``uploaded`` maps each input role to its server-side file name."""
        ...


def recipe_matches(recipe: ComfyRecipe, model_type: str) -> bool:
    return any(fnmatchcase(model_type, pattern) for pattern in recipe.model_patterns)


class GraphValidationError(ValueError):
    pass


def validate_graph(
    graph: Mapping[str, Any],
    *,
    allowed_node_types: frozenset[str],
    output_node: str,
    available_node_types: frozenset[str] | None = None,
) -> None:
    """Reject a graph that ComfyUI would accept but that cannot be what we meant.

    Checks: every node has an allowed ``class_type``; every link points at an
    existing node with an integer output slot; the declared output node exists.
    When ``available_node_types`` comes from ``/object_info``, every class_type
    must also be installed on the server.
    """
    if not isinstance(graph, Mapping) or not graph:
        raise GraphValidationError("graph is empty")
    problems: list[str] = []
    for node_id, node in graph.items():
        if not isinstance(node, Mapping):
            problems.append(f"node {node_id}: not an object")
            continue
        class_type = node.get("class_type")
        if not isinstance(class_type, str) or not class_type:
            problems.append(f"node {node_id}: missing class_type")
            continue
        if class_type not in allowed_node_types:
            problems.append(f"node {node_id}: class_type {class_type} not allowed by recipe")
        if available_node_types is not None and class_type not in available_node_types:
            problems.append(f"node {node_id}: class_type {class_type} not installed on server")
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            problems.append(f"node {node_id}: inputs must be an object")
            continue
        for input_name, value in inputs.items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                source, slot = value
                if source not in graph:
                    problems.append(f"node {node_id}.{input_name}: link to missing node {source}")
                if not isinstance(slot, int) or isinstance(slot, bool) or slot < 0:
                    problems.append(f"node {node_id}.{input_name}: invalid output slot {slot!r}")
    if output_node not in graph:
        problems.append(f"output node {output_node} missing from graph")
    if problems:
        raise GraphValidationError("; ".join(problems))
