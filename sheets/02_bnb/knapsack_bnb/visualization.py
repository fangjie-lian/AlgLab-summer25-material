"""
This code creates an interactive visualization of a branch and bound tree.
You do not need to modify this code.
"""

import logging
from pathlib import Path

from jinja2 import Template
from pydantic import BaseModel, Field

from .bnb_nodes import BnBNode
from .instance import Instance
from .relaxation import RelaxedSolution


class BnBTree(BaseModel):
    """
    Provides a structure for the branch-and-bound tree. Because of its recursive nature,
    every node is a BnBTree object, and the children of a node are stored in a list of BnBTree objects.
    """

    node_id: int = Field(..., description="The ID of the root node.")
    created_at: int = Field(..., description="The time when the node was created.")
    processed_at: int | None = Field(
        default=None,
        description="The time when the node was processed. This may be later than its creation time. It may also be None if it got pruned.",
    )
    label: str = Field(..., description="Label of the node in the visualization.")
    color: str = Field(..., description="Color of the node in the visualization.")
    children: list["BnBTree"] = Field(
        default_factory=list, description="Children of the node."
    )


class BnBVisualization:
    def __init__(self, instance: Instance):
        self.root: BnBTree | None = None
        self.node_links: dict[int, BnBTree] = {}
        self.instance = instance
        self.node_detail_texts = {}
        self.iterations = []  # id of node processed in iteration

    def _get_node_color(self, node: BnBNode) -> str:
        if (
            node.relaxed_solution.does_obey_capacity_constraint()
            and node.relaxed_solution.is_integral()
            and not node.relaxed_solution.is_infeasible()
        ):
            return "#20c997"
        return "#adb5bd" if not node.relaxed_solution.is_infeasible() else "#dc3545"

    def on_new_node_in_tree(self, node: BnBNode):
        color = self._get_node_color(node)
        label = f"{node.relaxed_solution.upper_bound:.1f}"
        data = BnBTree(
            node_id=node.node_id,
            label=label,
            color=color,
            children=[],
            created_at=len(self.iterations),
        )
        if node.parent_id is None:
            assert self.root is None, "Root already exists."
            self.root = data
        else:
            self.node_links[node.parent_id].children.append(data)
        self.node_links[node.node_id] = data

    def on_node_processed(
        self,
        node: BnBNode,
        lb: float,
        ub: float,
        best_solution: RelaxedSolution | None,
        heuristic_solutions: list[RelaxedSolution],
    ):
        self.iterations.append(node.node_id)
        self.node_links[node.node_id].processed_at = len(self.iterations) - 1
        if node.parent_id is not None:
            parent_processed_at = self.node_links[node.parent_id].processed_at
            node_processed_at = self.node_links[node.node_id].processed_at
            assert parent_processed_at is not None
            assert node_processed_at is not None
            assert parent_processed_at < node_processed_at
        with (Path(__file__).parent / "./templates/node.jinja2.html").open() as file:
            template_node_info = Template(file.read())
            node_info = template_node_info.render(
                node=node,
                lb=lb,
                ub=ub,
                heuristic_solutions=heuristic_solutions,
                best_solution=best_solution,
                weight=node.relaxed_solution.weight(),
            )
            self.node_detail_texts[node.node_id] = node_info

    def visualize(self, path: str = "output.html"):
        if self.root is None:
            msg = "No nodes to visualize."
            raise ValueError(msg)
        with (
            Path(__file__).parent / "./templates/instance.jinja2.html"
        ).open() as file:
            template_instance = Template(file.read())
            instance_info = template_instance.render(instance=self.instance)
        with (Path(__file__).parent / "./templates/bnb.jinja2.html").open() as file:
            template: Template = Template(file.read())
            with Path(path).open("w") as file:
                data = str(self.root.model_dump_json())
                file.write(
                    template.render(
                        tree_data=data,
                        num_iterations=len(self.iterations) - 1,
                        iterations=self.iterations,
                        instance_info=instance_info,
                        node_details=self.node_detail_texts,
                    )
                )
                logging.info("Visualization saved to %s", path)
                # open the file in the default web browser
                try:
                    import webbrowser

                    webbrowser.open_new_tab(path)
                except Exception as e:
                    logging.error(
                        "Error opening the file in the browser. Please open it manually."
                    )
                    logging.exception(e)
