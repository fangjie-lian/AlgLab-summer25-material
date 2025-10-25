"""
Branching Strategy Module

Defines how to split (branch) a BnB tree node when its current relaxed solution
is not yet a feasible integer solution. At each branching step:
 1. Select a decision variable that has not been fixed.
 2. Create two children by fixing that variable to 0 (exclude) and 1 (include).
 3. If all variables are fixed, no branches are returned (leaf node).

You should implement your own strategies by subclassing `BranchingStrategy`.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

from .bnb_nodes import BnBNode, BranchingDecisions


class BranchingStrategy(ABC):
    """
    Abstract base for branching policies based on a node's relaxed solution.

    Subclasses must implement `make_branching_decisions` to return zero,
    two, or more `BranchingDecisions` objects describing child nodes.
    """

    @abstractmethod
    def make_branching_decisions(self, node: BnBNode) -> Iterable[BranchingDecisions]:
        """
        Return an iterable of `BranchingDecisions` to create child nodes.
        If no decisions can be made (all variables fixed), return an empty iterable.
        """
        ...


class FirstUndecidedBranchingStrategy(BranchingStrategy):
    """
    Branch on the first variable that has not yet been fixed.
    """

    def make_branching_decisions(self, node: BnBNode) -> Tuple[BranchingDecisions, ...]:
        # find the smallest index i where no decision has been made
        first_unfixed = min(
            (i for i, val in enumerate(node.branching_decisions) if val is None),
            default=-1,
        )
        if first_unfixed < 0:
            return ()  # leaf node, nothing to branch
        return node.branching_decisions.split_on(first_unfixed)


class MyBranchingStrategy(BranchingStrategy):
    """
    Your implementation of a branching strategy.

    Decide which variable(s) to branch on at each node using information
    from the node's relaxed solution (e.g., fractional values, scores, etc.).
    The simplest strategy is to pick an unfixed variable and split on 0/1.
    """

    def make_branching_decisions(
        self, fractional_node: BnBNode
    ) -> Tuple[BranchingDecisions, ...]:
        """
        You are given a node with a fractional solution. Decide how you want to split the
        solution space. The easiest way to do so is to select a variable that hasn't been fixed
        yet and split the solution space into two by fixing it to 0 and 1, respectively.
        """
        # Node is integral => optimale Lsg für den Node ist gefunden
        assert not fractional_node.relaxed_solution.is_integral(), "Node is fractional"
        assert (
            fractional_node.relaxed_solution.does_obey_capacity_constraint()
        ), "Node is feasible"
        # find the index of the first non-fixed variable -> the fractional node
        idx = 0
        for i, x in enumerate(fractional_node.relaxed_solution.selection):
            if x > 0 and x < 1:
                idx = i
                break
        return fractional_node.branching_decisions.split_on(idx)


class NoBranchingStrategy(BranchingStrategy):
    """
    Your implementation of a branching strategy.

    Decide which variable(s) to branch on at each node using information
    from the node's relaxed solution (e.g., fractional values, scores, etc.).
    The simplest strategy is to pick an unfixed variable and split on 0/1.
    """

    def make_branching_decisions(
        self, fractional_node: BnBNode
    ) -> Tuple[BranchingDecisions, ...]:
        """
        You are given a node with a fractional solution. Decide how you want to split the
        solution space. The easiest way to do so is to select a variable that hasn't been fixed
        yet and split the solution space into two by fixing it to 0 and 1, respectively.
        """
        # Node is integral => optimale Lsg für den Node ist gefunden
        assert not fractional_node.relaxed_solution.is_integral(), "Node is fractional"
        assert (
            fractional_node.relaxed_solution.does_obey_capacity_constraint()
        ), "Node is feasible"
        # find the index of the first non-fixed variable
        first_unfixed_idx = min(
            i for i, x in enumerate(fractional_node.branching_decisions) if x is None
        )
        return fractional_node.branching_decisions.split_on(first_unfixed_idx)
