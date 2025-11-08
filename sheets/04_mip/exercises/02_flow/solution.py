import logging
import typing

import gurobipy as gp
import networkx as nx
from data_schema import Instance, Solution
from gurobipy import GRB

# Define types for implicit documentation
Vertex = typing.Hashable
Edge = tuple[Vertex, Vertex]


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
        self.model = gp.Model()
        self.G = nx.DiGraph()
        self.G.add_node(instance.elevator_location, typ="center")
        self.vars_of_mines = {}
        self.vars_of_tunnels_if_use = {}
        self.vars_of_tunnelflows = {}

        self.G.add_nodes_from(instance.locations, typ="normal")

        self.G.nodes[instance.elevator_location]["typ"] = "center"

        ### super wichtig, str und int (debugging)
        for m_str, mine_obj in instance.mines.items():
            m_int = int(m_str)

            self.G.nodes[m_int]["typ"] = "mine"
            self.G.nodes[m_int]["ore"] = mine_obj.ore_per_hour

            self.vars_of_mines[m_int] = self.model.addVar(
                vtype=gp.GRB.INTEGER,
                name=f"mine_{m_int}",
                lb=0,
                ub=mine_obj.ore_per_hour,
            )

        for tunnel in instance.tunnels:
            s = tunnel.source
            t = tunnel.target
            self.G.add_edge(
                s,
                t,
                throughput=tunnel.throughput_per_hour,
                costs=tunnel.reinforcement_costs,
            )
            self.G.add_edge(
                t,
                s,
                throughput=tunnel.throughput_per_hour,
                costs=tunnel.reinforcement_costs,
            )
            self.vars_of_tunnels_if_use[(s, t)] = self.model.addVar(
                vtype=gp.GRB.BINARY, name=f"tunnel_{s}_{t}"
            )
            self.vars_of_tunnels_if_use[(t, s)] = self.model.addVar(
                vtype=gp.GRB.BINARY, name=f"tunnel_{t}_{s}"
            )
            self.vars_of_tunnelflows[(s, t)] = self.model.addVar(
                vtype=gp.GRB.INTEGER,
                name=f"tunnel_usage_{s}_{t}",
                lb=0,
                ub=tunnel.throughput_per_hour,
            )
            self.vars_of_tunnelflows[(t, s)] = self.model.addVar(
                vtype=gp.GRB.INTEGER,
                name=f"tunnel_usage_{t}_{s}",
                lb=0,
                ub=tunnel.throughput_per_hour,
            )

        v = self.instance.elevator_location
        self.model.setObjective(
            gp.quicksum(x for x in self.vars_of_mines.values()),  # .values()
            gp.GRB.MAXIMIZE,
        )

        # Constraints
        costs = 0
        for tunnel in instance.tunnels:
            s = tunnel.source
            t = tunnel.target
            s_to_t = self.vars_of_tunnels_if_use[(s, t)]
            s_to_t_throughout = self.vars_of_tunnelflows[(s, t)]
            t_to_s = self.vars_of_tunnels_if_use[(t, s)]
            t_to_s_throughout = self.vars_of_tunnelflows[(t, s)]
            self.model.addConstr(
                s_to_t_throughout <= s_to_t * tunnel.throughput_per_hour
            )
            self.model.addConstr(
                t_to_s_throughout <= t_to_s * tunnel.throughput_per_hour
            )
            costs += tunnel.reinforcement_costs * (s_to_t + t_to_s)
            self.model.addConstr(s_to_t + t_to_s <= 1)
        self.model.addConstr(costs <= self.budget)
        for v in self.G.nodes:
            edges_to_v = [
                (u, v)
                for u in self.G.predecessors(v)
                if (u, v) in self.vars_of_tunnelflows
            ]
            edges_from_v = [
                (v, u)
                for u in self.G.successors(v)
                if (v, u) in self.vars_of_tunnelflows
            ]
            incomming_ore = gp.quicksum(
                self.vars_of_tunnelflows[(u, v)] for (u, v) in edges_to_v
            )
            outgoing_ore = gp.quicksum(
                self.vars_of_tunnelflows[(v, u)] for (v, u) in edges_from_v
            )

            v_attributes = self.G.nodes[v]
            if v_attributes["typ"] == "mine":
                consumed_ore = self.vars_of_mines[v]
                self.model.addConstr(incomming_ore + consumed_ore == outgoing_ore)
            if v_attributes["typ"] == "normal":
                self.model.addConstr(incomming_ore == outgoing_ore)
            if v_attributes["typ"] == "center":
                self.model.addConstr(outgoing_ore == 0)
                self.model.addConstr(
                    incomming_ore == gp.quicksum(x for x in self.vars_of_mines.values())
                )

        # self.tmp_solution: Solution | None

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

        self.model.optimize()
        if self.model.Status == GRB.OPTIMAL or self.model.SolCount > 0:
            if self.model.Status == GRB.OPTIMAL:
                logging.info("Optimal solution found.")
                logging.info("Objective value: %f", self.model.ObjVal)
            else:
                logging.info("Feasible solution found, but not proven optimal.")
                logging.info("Objective value: %f", self.model.ObjVal)

            tunnelflows = [
                ((s, t), x.X)
                for (s, t), x in self.vars_of_tunnelflows.items()
                if x.X > 0.01
            ]

            return Solution(flow=tunnelflows)

        logging.warning("No feasible solution found within the time limit.")
        return None
