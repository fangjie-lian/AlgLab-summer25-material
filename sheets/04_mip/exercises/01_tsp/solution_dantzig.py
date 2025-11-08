"""
Implement the Dantzig-Fulkerson-Johnson formulation for the TSP.
"""

import logging
import typing

import gurobipy as gp
import networkx as nx
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


class GurobiTspSolver:
    """
    IMPLEMENT ME!
    """

    def __init__(self, G: nx.Graph, k: int = 2):
        """
        G is a weighted networkx graph, where the weight of an edge is stored in the
        "weight" attribute. It is strictly positive.
        """
        self.graph = G
        # einfache Voraussetzungen
        assert (
            G.number_of_edges() == G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
        ), "Invalid graph"  # vollständiger Graph => |E|=|V|*(|V|-1)/2
        assert all(
            weight > 0
            for _, _, weight in G.edges.data("weight", default=None)  # type: ignore[attr-defined]
        ), "Invalid graph"

        assert k in {1, 2}, "Invalid k"
        self.k = k
        logging.info("Creating model ...")
        logging.info(
            "Graph has %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
        )
        logging.info("Implementing subtour elimination with >= %d", k)

        self._model = gp.Model()
        # TODO: Implement me!
        self.vars = {
            (u, v): self._model.addVar(vtype=gp.GRB.BINARY, name=f"edge_{u}_{v}")
            for u, v in G.edges
        }
        logging.info("Edge variables initialized: %d variables.", len(self.vars))

        self.tmp_solution_graph = None
        self.solution_graph = None
        self.solution_value = None
        self.lb = 0
        self._model.setObjective(
            gp.quicksum(G[u][v]["weight"] * self.vars[(u, v)] for u, v in self.vars),
            gp.GRB.MINIMIZE,
        )

        # Constraint: Genau |V| Kanten werden gewählt
        self._model.addConstr(
            gp.quicksum(x for _, x in self.vars.items()) == len(G.nodes)
        )

        # Constraint: Degree = 2
        for v in G.nodes:
            self._model.addConstr(
                gp.quicksum(x for _, x in self.incident_edges(v)) == 2
            )

    # Kopiere vom Bsp
    def x(self, v: Vertex, w: Vertex) -> gp.Var:
        """
        Return variable for edge (v, w).
        """
        if (v, w) in self.vars:
            return self.vars[v, w]
        # If (v,w) was not found, try (w,v)
        return self.vars[w, v]

    # Kopiere vom Bsp
    def outgoing_edges(
        self, vertices: list[Vertex]
    ) -> typing.Iterable[tuple[Edge, gp.Var]]:
        """
        Return all edges & variables that are outgoing from the given vertices.
        """
        # Not super efficient, but efficient enough for our purposes.
        for (v, w), x in self.vars.items():
            if v in vertices and w not in vertices:
                yield (v, w), x
            elif w in vertices and v not in vertices:
                yield (w, v), x

    # Kopiere vom Bsp
    def incident_edges(self, v: Vertex) -> typing.Iterable[tuple[Edge, gp.Var]]:
        """
        Return all edges & variables that are incident to the given vertex.
        """
        for n in self.graph.neighbors(v):
            yield (v, n), self.x(v, n)

    # Kopiere vom Bsp
    def __iter__(self) -> typing.Iterator[tuple[Edge, gp.Var]]:
        """
        Iterate over all edges & variables.
        """
        return iter(self.vars.items())

    def get_lower_bound(self) -> float:
        """
        Return the current lower bound.
        """
        # TODO: Implement me!
        return self.lb

    def get_solution(self) -> typing.Optional[nx.Graph]:  # wenn keine dann None
        """
        Return the current solution as a graph.
        """
        # TODO: Implement me!
        if self.solution_graph == None:
            return None
        return self.solution_graph

    def get_objective(self) -> typing.Optional[float]:  # entweder Graph oder None
        """
        Return the objective value of the last solution.
        """
        # TODO: Implement me!
        return self.solution_value

    def solve(self, time_limit: float, opt_tol: float = 0.001) -> None:
        """
        Solve the model. After solving the model, the solution, its objective value,
        and the lower bounds should be available via the corresponding methods.
        """
        logging.info("Solving model ...")
        # Set parameters for the solver.
        self._model.Params.LogToConsole = 1
        self._model.Params.TimeLimit = time_limit
        self._model.Params.LazyConstraints = 1
        self._model.Params.MIPGap = (
            opt_tol  # https://www.gurobi.com/documentation/11.0/refman/mipgap.html
        )
        _check_linear(self._model)  # Ensure the model is linear

        # ...
        # TODO: Implement me!

        def callback(model, where):
            # This callback is called by Gurobi on various occasions, and
            # we can react to these occasions.
            if where == gp.GRB.Callback.MIPSOL:
                # We are in a new MIP solution. We can query the solution
                # and add additional constraints, if we want to.
                # We are going to enforce a leaving edge for every component
                # that contains only a part of the terminals.
                used_edges = [
                    vw
                    for vw, x in self.vars.items()
                    if self._model.cbGetSolution(x) > 0.5
                ]
                self.tmp_solution_graph = nx.Graph(used_edges)
                comps = list(nx.connected_components(self.tmp_solution_graph))
                if len(comps) == 1:  # solution is connected
                    return

                for comp in comps:
                    if len(comp) < 2 or len(comp) == len(self.graph.nodes):
                        continue
                    choosed_edges = self.outgoing_edges(comp)
                    # min. eine ausgehende Kante
                    model.cbLazy(gp.quicksum(x for _, x in choosed_edges) >= self.k)

        self._model.Params.LazyConstraints = 1  # Enable lazy constraints
        self._model.optimize(callback)  # Pass the callback with the `solve` call
        self.lb = self._model.ObjBound

        # Log the outcome of the optimization
        if self._model.Status == GRB.OPTIMAL or self._model.SolCount > 0:
            if self._model.Status == GRB.OPTIMAL:
                logging.info("Optimal solution found.")
                logging.info("Objective value: %f", self._model.ObjVal)
            else:
                logging.info("Feasible solution found, but not proven optimal.")
            logging.info("Objective value: %f", self._model.ObjVal)
            used_edges = [vw for vw, x in self if x.X > 0.5]
            self.solution_graph = nx.Graph(used_edges)
            self.solution_value = self._model.ObjVal
            return self.solution_graph
        logging.warning("No feasible solution found within the time limit.")
        return None

        # Ohne Lazy-Constraints und callback
        # while True:
        #     self._model.optimize()
        #     if self._model.Status == GRB.INFEASIBLE:
        #         break
        #     if self._model.Status in [GRB.OPTIMAL, GRB.SUBOPTIMAL]:
        #         used_edges = [vw for vw, x in self if x.X > 0.5]
        #         self.tmp_solution_graph = nx.Graph(used_edges)

        #         # Verifikation für HC
        #         comp_list = list(nx.connected_components(self.tmp_solution_graph))
        #         if len(comp_list) == 1:
        #             value = self._model.ObjVal
        #             if self._model.Status == GRB.OPTIMAL:
        #                 self.solution_graph = self.tmp_solution_graph
        #                 self.solution_value = value
        #                 break
        #         for comp in comp_list:
        #             if len(comp) < 2:
        #                 continue
        #             choosed_edges = self.outgoing_edges(comp)
        #             # min. eine ausgehende Kante
        #             self._model.addConstr(
        #                 gp.quicksum(x for _, x in choosed_edges) >= self.k
        #             )
