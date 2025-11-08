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


class GurobiTspRelaxationSolver:
    """
    IMPLEMENT ME!
    """

    def __init__(self, G: nx.Graph, k: int = 2):
        """
        G is a weighted networkx graph, where the weight of an edge is stored in the
        "weight" attribute. It is strictly positive.
        """
        self.graph = G
        self.k = k
        assert (
            G.number_of_edges() == G.number_of_nodes() * (G.number_of_nodes() - 1) / 2
        ), "Invalid graph"
        assert all(
            weight > 0
            for _, _, weight in G.edges.data("weight", default=None)  # type: ignore[attr-defined]
        ), "Invalid graph"

        assert k in {1, 2}, "Invalid k"
        logging.info("Creating model ...")
        logging.info(
            "Graph has %d nodes and %d edges", G.number_of_nodes(), G.number_of_edges()
        )
        logging.info("Implementing subtour elimination with >= %d", k)
        self._model = gp.Model()
        # TODO: Implement me!
        self.vars = {
            (u, v): self._model.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"edge_{u}_{v}", lb=0, ub=1
            )
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

    def as_graph(self) -> nx.Graph:
        solution_graph = nx.Graph()
        for (u, v), x in self.vars.items():
            value = x.X  # fractional
            if value >= 0.01:  # + value
                solution_graph.add_edge(
                    u, v, weight=self.graph[u][v]["weight"], x=value
                )
        return solution_graph

    def get_solution(self) -> typing.Optional[nx.Graph]:  # entweder Graph oder None
        """
        Return the current solution as a graph.

        The solution should be a networkx Graph were the
        fractional value of the edge is stored in the "x" attribute.
        You do not have to add edges with x=0.

        ```python
        graph = nx.Graph()
        graph.add_edge(0, 1, x=0.5)
        graph.add_edge(1, 2, x=1.0)
        ```
        """
        # TODO: Implement me!
        if self.solution_graph == None:
            return None
        return self.solution_graph

    def get_objective(self) -> typing.Optional[float]:
        """
        Return the objective value of the last solution.
        """
        # TODO: Implement me!
        return self.solution_value

    def solve(self) -> None:
        """
        Solve the model. After solving the model, the solution, its objective value,
        and the lower bounds should be available via the corresponding methods.
        """
        logging.info("Solving model ...")
        # Set parameters for the solver.
        self._model.Params.LogToConsole = 1

        # TODO: Implement me!
        while True:
            self._model.optimize()
            if self._model.Status == GRB.INFEASIBLE:
                break
            if self._model.Status == GRB.OPTIMAL:
                self.tmp_solution_graph = self.as_graph()

                # Verifikation für HC
                comp_list = list(nx.connected_components(self.tmp_solution_graph))
                if len(comp_list) == 1:
                    value = self._model.ObjVal
                    self.solution_graph = self.tmp_solution_graph
                    self.solution_value = value
                    self.lb = value
                    break

                for comp in comp_list:
                    if len(comp) < 2:
                        continue
                    choosed_edges = self.outgoing_edges(comp)
                    # min. eine ausgehende Kante
                    self._model.addConstr(
                        gp.quicksum(x for _, x in choosed_edges) >= self.k
                    )
