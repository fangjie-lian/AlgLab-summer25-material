import math

from data_schema import Instance, Solution
from ortools.sat.python.cp_model import OPTIMAL, CpModel, CpSolver


class MultiKnapsackSolver:
    """
    This class can be used to solve the Multi-Knapsack problem
    (also the standard knapsack problem, if only one capacity is used).

    Attributes:
    - instance (Instance): The multi-knapsack instance
        - items (List[Item]): a list of Item objects representing the items to be packed.
        - capacities (List[int]): a list of integers representing the capacities of the knapsacks.
    - model (CpModel): a CpModel object representing the constraint programming model.
    - solver (CpSolver): a CpSolver object representing the constraint programming solver.
    """

    def __init__(self, instance: Instance, activate_toxic: bool = False):
        """
        Initialize the solver with the given Multi-Knapsack instance.

        Args:
        - instance (Instance): an Instance object representing the Multi-Knapsack instance.
        """
        self.items = instance.items
        self.activate_toxic = activate_toxic
        self.capacities = instance.capacities
        self.model = CpModel()
        self.solver = CpSolver()
        self.solver.parameters.log_search_progress = True
        # TODO: Implement me!
        # j - knapsack-index, i - item-index
        self.x = [
            [self.model.new_bool_var(f"x_{j}_{i}") for i in range(len(self.items))]
            for j in range(len(self.capacities))
        ]
        # die Kapazität jedes Knapsacks darf nicht überschritten werden
        for j in range(len(self.capacities)):
            total_weight = sum(
                self.x[j][i] * self.items[i].weight for i in range(len(self.items))
            )
            self.model.add(total_weight <= self.capacities[j])
        # jedes Item wird nur max. 1 mal verwendet
        for i in range(len(self.items)):
            self.model.add(sum(self.x[j][i] for j in range(len(self.capacities))) <= 1)
        # Zielfkt.
        total_value = []
        for j in range(len(self.capacities)):
            for i in range(len(self.items)):
                # x[j][i] * value[i]
                total_value.append(
                    self.x[j][i] * self.items[i].value
                )  # self.items[i].value

        self.model.maximize(sum(total_value))

    def solve(self, timelimit: float = math.inf) -> Solution:
        """
        Solve the Multi-Knapsack instance with the given time limit.

        Args:
        - timelimit (float): time limit in seconds for the cp-sat solver.

        Returns:
        - Solution: a list of lists of Item objects representing the items packed in each knapsack
        """
        # handle given time limit
        if timelimit <= 0.0:
            return Solution(trucks=[])  # empty solution
        if timelimit < math.inf:
            self.solver.parameters.max_time_in_seconds = timelimit
        # TODO: Implement me!
        # Solve the model
        status = self.solver.Solve(self.model)
        # Check and return the solution
        assert status == OPTIMAL
        trucks_solution = []
        for j in range(len(self.capacities)):
            packed_items = []
            for i in range(len(self.items)):
                if self.solver.Value(self.x[j][i]) == 1:
                    packed_items.append(self.items[i])
            trucks_solution.append(packed_items)

        return Solution(trucks=trucks_solution)
