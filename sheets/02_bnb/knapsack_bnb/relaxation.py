"""
Relaxation Module

In branch-and-bound, a relaxation of the original 0/1 knapsack yields an upper bound
on the best feasible solution within a branch. If this bound does not exceed your
current best feasible solution, you can prune that branch and skip exploring it.

This file provides three example strategies:
  1. VeryNaiveRelaxationSolver:
     - Ignores capacity entirely, sets every unfixed item to 1.
     - Fastest, loosest bound.
  2. NaiveRelaxationSolver:
     - Checks that already-fixed items of 1 fit capacity.
     - Sets all unfixed items to 1, ignoring capacity beyond fixed part.
     - Slightly tighter bound than VeryNaive.
  3. MyRelaxationSolver:
     - Stub for your own algorithm (e.g., fractional knapsack, propagation).

You should subclass `RelaxationSolver` and implement `solve(instance, decisions)`
so that:
  a) fixed decisions remain unchanged;
  b) objective >= best 0/1 solution consistent with those decisions.
"""

import abc

from .instance import Instance
from .branching_decisions import BranchingDecisions
from .relaxed_solution import RelaxedSolution


class RelaxationSolver(abc.ABC):
    """
    Abstract base for relaxation strategies.

    Implement `solve` to compute an upper bound on the best 0/1 solution
    consistent with given decisions.
    """

    @abc.abstractmethod
    def solve(
        self, instance: Instance, decisions: BranchingDecisions
    ) -> RelaxedSolution:
        """
        Return a `RelaxedSolution` satisfying:
          - fixed items in `decisions` remain at 0 or 1;
          - upper_bound >= best feasible 0/1 solution under those decisions.
        """
        ...


class VeryNaiveRelaxationSolver(RelaxationSolver):
    """
    A relaxation solver for the knapsack problem that naively sets every unfixed
    item to 1 without considering the capacity constraint. This approach provides
    a very loose upper bound for the problem.

    Explanation:
    The solver assumes that all unfixed items can be fully included in the knapsack
    (i.e., their selection is set to 1.0) regardless of the capacity constraint.
    This results in an overestimation of the objective value, making it an upper
    bound. The rationale is that the true optimal solution cannot exceed this
    value since it must respect the capacity constraint, which this naive approach
    ignores.
    """

    def solve(
        self, instance: Instance, decisions: BranchingDecisions
    ) -> RelaxedSolution:
        # build selection: 1.0 for fixed 1 or unfixed, 0 for fixed 0
        selection = [0.0 if x == 0 else 1.0 for x in decisions]
        # compute objective value
        upper = sum(item.value * sel for item, sel in zip(instance.items, selection))
        return RelaxedSolution(instance, selection, upper)


class NaiveRelaxationSolver(RelaxationSolver):
    """
    Ensure fixed 1's fit capacity; set every unfixed item to 1.
    """

    def solve(
        self, instance: Instance, decisions: BranchingDecisions
    ) -> RelaxedSolution:
        # compute capacity after fixed 1 items
        used = sum(item.weight for item, x in zip(instance.items, decisions) if x == 1)
        if used > instance.capacity:
            return RelaxedSolution.create_infeasible(instance)

        selection = [0.0 if x == 0 else 1.0 for x in decisions]
        upper = sum(item.value * sel for item, sel in zip(instance.items, selection))
        return RelaxedSolution(instance, selection, upper)


class MyRelaxationSolver(RelaxationSolver):
    """
    Your relaxation solver stub.

    Implement any relaxation (e.g., fractional knapsack, propagation) to tighten bounds.
    """
    def solve_fractional_knapsack(
        self, instance: Instance, decisions: BranchingDecisions
    ) -> RelaxedSolution:
        """
        Solve the fractional knapsack problem from the given instance and deduced
          fixations.
        instance: knapsack problem instance
        fixation: list of predefined item selections, where 0 means not taken,
            1 means fully taken, and None means not fixed
        """
        remaining_capacity = instance.capacity - sum(
            item.weight for item, x in zip(instance.items, decisions) if x == 1
        )
        # Compute solution
        selection = [1.0 if x == 1 else 0.0 for x in decisions]
        remaining_indices = [i for i, x in enumerate(decisions) if x is None]
        remaining_indices.sort(
            key=lambda i: instance.items[i].value / instance.items[i].weight,
            reverse=True,
        )
        for i in remaining_indices:
            # Fill solution with items sorted by value/weight
            if instance.items[i].weight <= remaining_capacity:
                selection[i] = 1.0
                remaining_capacity -= instance.items[i].weight
            else:
                selection[i] = remaining_capacity / instance.items[i].weight
                break  # no capacity left
        assert all(
            x0 == x1 for x0, x1 in zip(decisions, selection) if x0 is not None
        ), "Fixed part is not allowed to change."
        upper = sum(item.value * sel for item, sel in zip(instance.items, selection))
        return RelaxedSolution(instance, selection, upper)


    def solve(
        self, instance: Instance, decisions: BranchingDecisions
    ) -> RelaxedSolution:
        fractional_solution = self.solve_fractional_knapsack(instance, decisions)

        if fractional_solution.is_integral():
            return fractional_solution
        
        frac_index = 0
        choosen_indices = []
        for i, x in enumerate(fractional_solution.selection):
            if x == 1 and decisions[i] is None:
                choosen_indices.append(i)
            if x > 0 and x < 1:
                choosen_indices.append(i)
                frac_index = i
                break
        
        max_new_frac_solution_value = float('-inf')
        max_new_frac_solution = None
        for j in choosen_indices:
            new_fixation = decisions.copy()
            new_fixation.fix(j, 0)
            new_frac_solution = self.solve_fractional_knapsack(instance, new_fixation)
            if max_new_frac_solution_value < new_frac_solution.value():
                max_new_frac_solution = new_frac_solution
                max_new_frac_solution_value = new_frac_solution.value()
        
        return max_new_frac_solution
