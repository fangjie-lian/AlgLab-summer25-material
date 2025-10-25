import logging

import gurobipy as gp
from data_schema import Instance, Solution


def _check_linear(model: gp.Model):
    # Check if model has quadratic terms
    if model.NumQConstrs > 0:
        msg = (
            "The model uses quadratic constraints (multiplying variables), "
            "which are less efficient. All exercises can be solved with linear constraints."
        )
        raise ValueError(msg)
    if model.NumQNZs > 0:
        msg = (
            "The model uses quadratic terms (multiplying variables) in the objective, "
            "which are less efficient. All exercises can be solved with linear terms."
        )
        raise ValueError(msg)


class MiningRoutingSolver:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self.budget = instance.budget
        logging.info("Creating model ...")
        logging.info(
            "Instance has %d locations, %d mines, %d tunnels, and a budget of %.2f",
            len(instance.locations),
            len(instance.mines),
            len(instance.tunnels),
            instance.budget,
        )
        # TODO: Implement me!

    def solve(self) -> Solution:
        """
        Calculate the optimal solution to the problem.
        Returns the "flow" as a list of tuples, each tuple with two entries:
            - The *directed* edge tuple. Both entries in the edge should be ints, representing the ids of locations.
            - The throughput/utilization of the edge, in goods per hour
        """
        # TODO: implement me!
        logging.info("Solving model...")
        _check_linear(self.model)
