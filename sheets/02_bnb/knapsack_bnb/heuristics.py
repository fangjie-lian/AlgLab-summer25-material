"""
Heuristics Module

In branch-and-bound, a relaxation gives an upper bound on the best objective in a branch.
To tighten pruning, you need feasible (integral) solutions to serve as lower bounds.
Instead of waiting for an integral node, you can derive feasible solutions from the relaxation
(e.g., rounding, greedy inclusion) to improve search efficiency.

You can implement heuristics by subclassing `Heuristics` and overriding `search(instance, node)`.
`search` should yield zero or more feasible `RelaxedSolution` objects.
"""

from abc import ABC, abstractmethod
from typing import Iterable, Tuple

from .bnb_nodes import BnBNode, RelaxedSolution
from .instance import Instance


class Heuristics(ABC):
    """
    Abstract base for heuristic generators.

    Implement `search` to produce feasible solutions from a node's relaxed solution.
    """

    @abstractmethod
    def search(self, instance: Instance, node: BnBNode) -> Iterable[RelaxedSolution]:
        """
        Return an iterable of feasible `RelaxedSolution` objects for pruning.
        """
        ...


class MyHeuristic(Heuristics):
    """
    Your heuristic implementation.

    The simplest heuristic returns the node's relaxed solution
    if it is already feasible (integral and within capacity).
    """

    def search(self, instance: Instance, node: BnBNode) -> Tuple[RelaxedSolution, ...]:
        # Maybe we can also obtain a feasible solution from fractional solutions?
        # It doesn't have to be perfect...
        solutions = []
        if (
            node.relaxed_solution.does_obey_capacity_constraint()
            and node.relaxed_solution.is_integral()
        ):
            # WARNING: Do not not modify the solution in place! Create a copy!
            solutions.append(node.relaxed_solution.copy()) # https://www.w3schools.com/python/ref_keyword_yield.asp
        elif ( # Änderung
            node.relaxed_solution.does_obey_capacity_constraint()
        ):
            integral_solution = node.relaxed_solution.copy()
            for i, x in enumerate(integral_solution.selection): # falsch: for i in integral_solution.selection
                if x > 0 and x < 1:
                    integral_solution.selection[i] = 0 # falsch: x = 0
            solutions.append(integral_solution)
        return tuple(solutions)