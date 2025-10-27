import logging

import networkx as nx
from pysat.solvers import Solver as SATSolver
from typing import Any
from itertools import combinations

# Configure logging to show INFO messages
logging.basicConfig(level=logging.INFO)

Node = Any
Edge = tuple[Node, Node]


class _EdgeVars:  # kopiert von example (node->edge)
    """
    The SAT-Solver interface uses integers for variables, with negative integers for negations
    and zero for a false "dummy" variable. Shifting variable management to its own class
    can enhance code cleanliness and reduce errors.
    """

    def __init__(self, graph: nx.graph, start: int = 1) -> None:
        self._vars = {
            get_edge(edge): i for i, edge in enumerate(graph.edges, start=start)
        }
        self._reverse = {i: edge for edge, i in self._vars.items()}

    def x(self, edge: Edge):
        """
        Return the variable representing the given edge.
        """
        return self._vars[get_edge(edge)]

    def edge(self, x: int) -> tuple[Edge, bool]:
        """
        Return the edge represented by the given variable.
        The second return value indicates whether the edge is negated.
        """
        if x < 0:
            return self._reverse[x], False
        return self._reverse[x], True

    def not_x(self, edge: Edge):
        """
        Return the variable representing the negation of the given edge.
        """
        return -self.x(edge)

    def get_edge_selection(self, model: list[int]) -> set[Edge]:
        """
        Parse the selected edges from a given model (solution for a SAT-formula).
        """
        return {self.edge(x)[0] for x in model if x > 0 and x in self._reverse} # x>0


def get_edge(edge: Edge):  # ein eindeutiger Ausdruck für eine Kante
    u, v = edge
    if u < v:
        return (u, v)
    else:
        return (v, u)


class HamiltonianCycleModel:
    def __init__(self, graph: nx.Graph) -> None:
        # Log model initialization details
        logging.info(
            "Initializing HamiltonianCycleModel with %d nodes and %d edges...",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        self.graph = graph
        self.solver = SATSolver("Minicard")
        self.assumptions = []  # for btsp-Aufgabe
        self.feasible = True
        # TODO: Implement me!
        # Log model initialization completion

        # prüfe zwei einfache Voraussetzungen für Existenz von HC
        if len(self.graph.edges) < len(self.graph.nodes):
            self.feasible = False
        for v in self.graph.nodes:
            if self.graph.degree(v) < 2:
                self.feasible = False
                break
        
        if self.feasible == True:
            self.edge_vars = _EdgeVars(graph)
            edge_list = [self.edge_vars.x(get_edge(e)) for e in graph.edges]
            # >=|V| and <=|V|
            self.solver.add_atmost(edge_list, len(graph.nodes))
            self.solver.add_atmost(
                [-x for x in edge_list],
                len(graph.edges) - len(graph.nodes),
            )
            # Degree Constraint
            for v in graph.nodes:
                all_edges_of_v = [get_edge(e) for e in graph.edges(v)]
                all_edges_of_v_vars = [self.edge_vars.x(e) for e in all_edges_of_v]
                # >=2 and <=2
                self.solver.add_atmost(all_edges_of_v_vars, 2)
                self.solver.add_atmost(
                    [-x for x in all_edges_of_v_vars], len(all_edges_of_v_vars) - 2
                )
            print("Degree Constraint done")

        """ # Subtour Elimination Constraints
        for i in range(2, len(graph.nodes)):
            for m in combinations(graph.nodes, i):
                choosed_nodes = list(m)
                choosed_edges = []
                for u, v in graph.edges:
                    if u in choosed_nodes and v not in choosed_nodes:
                        choosed_edges.append(get_edge((u, v)))
                if len(choosed_edges) < 2:
                    continue
                # >=2 true = >=1 true + !=1 true
                # >=1 true
                self.solver.add_clause([self.edge_vars.x(e) for e in choosed_edges])
                # !=1 true
                for choosed_e in choosed_edges:
                    self.solver.add_clause(
                        [-self.edge_vars.x(e) for e in choosed_edges if e != choosed_e]
                        + [self.edge_vars.x(choosed_e)]
                    ) """

        self._solution: list[Edge] | None = None
        logging.info("HamiltonianCycleModel initialized successfully!")

    def solve(
        self, edges_false_with_w: list[Edge] | None = None
    ) -> list[tuple[int, int]] | None:
        """
        Solves the Hamiltonian Cycle Problem. If a HC is found,
        its edges are returned as a list.
        If the graph has no HC, 'None' is returned.
        """
        self.assumptions = []

        # Initialisierung von Assumptionen
        if edges_false_with_w != None:
            edges_false_without_w = [
                get_edge((u, v)) for (u, v, _w) in edges_false_with_w
            ]
            self.assumptions = [-self.edge_vars.x(e) for e in edges_false_without_w]

        # Log the start of solving process
        logging.info(
            "Starting Hamiltonian cycle search with %d assumptions",
            len(self.assumptions),
        )
        
        # TODO: Implement me!
        if self.feasible == False:
            return None

        while True:
            if not self.solver.solve(assumptions=self.assumptions):
                return None
            model = self.solver.get_model()
            assert (
                model is not None
            ), "We expect a solution. Otherwise, we would have had a timeout."
            self._solution = list(self.edge_vars.get_edge_selection(model))

            solution_graph = nx.Graph()
            solution_graph.add_nodes_from(self.graph.nodes)
            solution_graph.add_edges_from(self._solution)

            # Verifikation:
            comp_list = list(nx.connected_components(solution_graph))
            for comp in comp_list:
                if len(comp) == len(self.graph.nodes):
                    logging.info(
                        "SAT solver solution (Hamiltonkreis): %s", self._solution
                    )
                    return self._solution

            for comp in comp_list:
                if len(comp) < 2:
                    continue
                choosed_edges = []
                # wie man es macht, ist wichtig
                for u in comp:
                    for v in self.graph.neighbors(u):
                        if v not in comp:
                            choosed_edges.append(get_edge((u, v)))
                # >=2 true
                if len(choosed_edges) >= 2:
                    self.solver.add_atmost(
                        [-self.edge_vars.x(e) for e in choosed_edges],
                        len(choosed_edges) - 2,
                    )
