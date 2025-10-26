import logging
import math
from typing import Iterable
import bisect

import networkx as nx
from pysat.solvers import Solver as SATSolver  # pip install python-sat


logging.basicConfig(level=logging.INFO)

# Define the node ID type. It is an integer but this helps to make the code more readable.
NodeId = int


class Distances:
    """
    This class provides a convenient interface to query distances between nodes in a graph.
    All distances are precomputed and stored in a dictionary, making lookups efficient.
    """

    def __init__(self, graph: nx.Graph) -> None:
        self.graph = graph
        # Compute all-pairs shortest paths
        # all_pairs_dijkstra_path_length: Compute shortest path lengths between all nodes in a weighted graph.
        self._distances = dict(nx.all_pairs_dijkstra_path_length(self.graph))
        # Log the computation details
        logging.info(
            "Computed all-pairs shortest paths for %d nodes", len(self._distances)
        )

    def all_vertices(self) -> Iterable[NodeId]:
        """Returns an iterable of all node IDs in the graph."""
        return self._distances.keys()

    def dist(self, u: NodeId, v: NodeId) -> float:
        """Returns the distance between nodes `u` and `v`."""
        return self._distances[u].get(
            v, math.inf
        )  # fall es keinen Pfad zwi u und v gibt, gebe +unendlich zurück

    def max_dist(self, centers: Iterable[NodeId]) -> float:
        """Returns the maximum distance from any node to the closest center."""
        return max(min(self.dist(c, u) for c in centers) for u in self.all_vertices())

    def vertices_in_range(self, u: NodeId, limit: float) -> Iterable[NodeId]:
        """Returns an iterable of nodes within `limit` distance from node `u`."""
        return (v for v, d in self._distances[u].items() if d <= limit)

    def sorted_distances(self) -> list[float]:
        """Returns a sorted list of all pairwise distances in the graph."""
        dists = sorted(
            dist
            for dist_dict in self._distances.values()
            for dist in dist_dict.values()
        )
        logging.info(
            "Collected and sorted %d pairwise distances with a range from %f to %f",
            len(dists),
            dists[0],
            dists[-1],
        )
        return dists


class _NodeVars:  # kopiert von example (graph->distance, Node->NodeID)
    """
    The SAT-Solver interface uses integers for variables, with negative integers for negations
    and zero for a false "dummy" variable. Shifting variable management to its own class
    can enhance code cleanliness and reduce errors.
    """

    def __init__(self, distances: Distances, start: int = 1) -> None:
        self._vars = {
            node: i for i, node in enumerate(distances.all_vertices(), start=start)
        }
        self._reverse = {i: node for node, i in self._vars.items()}

    def x(self, node: NodeId):
        """
        Return the variable representing the given node.
        """
        return self._vars[node]

    def node(self, x: int) -> tuple[NodeId, bool]:
        """
        Return the node represented by the given variable.
        The second return value indicates whether the node is negated.
        """
        if x < 0:
            return self._reverse[x], False
        return self._reverse[x], True

    def not_x(self, node: NodeId):
        """
        Return the variable representing the negation of the given node.
        """
        return -self.x(node)

    def get_node_selection(self, model: list[int]) -> set[NodeId]:
        """
        Parse the selected nodes from a given model (solution for a SAT-formula).
        """
        return {self.node(x)[0] for x in model if x in self._reverse}


class KCenterDecisionVariant:
    def __init__(self, distances: Distances, k: int, upper_bound: float) -> None:
        self.distances = distances
        logging.info("Initializing KCenterDecisionVariant for k=%d", k)
        # TODO: Implement me!
        # Solution model
        self.k = k
        self.upper_bound = upper_bound
        self.node_vars = _NodeVars(distances)
        self.solver = SATSolver("Minicard")
        self.solver.add_atmost(
            [self.node_vars.x(v) for v in distances.all_vertices()], k
        )
        self._solution: list[NodeId] | None = None

    def limit_distance(self, limit: float) -> None:
        """Adds constraints to the SAT solver to ensure coverage within the given distance."""
        logging.info("Limiting to distance: %f", limit)
        # TODO: Implement me!
        for v in self.distances.all_vertices():
            covered_nodes = [
                self.node_vars.x(u) for u in self.distances.vertices_in_range(v, limit)
            ]
            self.solver.add_clause(covered_nodes)

    def solve(self) -> list[NodeId] | None:
        """Solves the SAT problem and returns the list of selected nodes, if feasible."""
        logging.info("Attempting to solve the SAT formulation")
        # TODO: Implement me!
        all_nodes = list(self.distances.all_vertices())
        vars = [self.node_vars.x(v) for v in all_nodes]

        distances_values = sorted(set(self.distances.sorted_distances()))
        distances_values = [x for x in distances_values if x > 0]
        # upper_bound
        i = bisect.bisect_left(distances_values, self.upper_bound)
        if i >= len(distances_values):
            i = len(distances_values) - 1

        while i >= 0:
            self.solver = SATSolver("Minicard")
            self.solver.add_atmost(vars, self.k)
            limit = distances_values[i]
            self.limit_distance(limit)
            if self.solver.solve():
                model = self.solver.get_model()
                if model is None:
                    break
                self._solution = list(self.node_vars.get_node_selection(model))
                # aku. obere Schranke
                self.upper_bound = self.distances.max_dist(self._solution)
                i = min(
                    i - 1, bisect.bisect_left(distances_values, self.upper_bound) - 1
                )
            else:
                break

        """ # binary search - irgendwie to slow 
        while a <= b:
            i = a + (b - a) // 2
            limit = distances_values[i]

            # new solver (wegen binary search)
            self.solver = SATSolver("Minicard")
            self.solver.add_atmost(vars, self.k)
            self.limit_distance(limit)

            if self.solver.solve():
                model = self.solver.get_model()
                if model is None:
                    a = i + 1
                    if b - a == 0:
                        break
                    continue
                self._solution = list(self.node_vars.get_node_selection(model))
                # aku. obere Schranke
                self.upper_bound = self.distances.max_dist(self._solution)
                i = bisect.bisect_left(distances_values, self.upper_bound)
                b = i - 1
            else:
                a = i + 1
        """
        logging.info("SAT solver solution: %s", self._solution)
        return self._solution

    def get_solution(self) -> list[NodeId]:
        """Returns the solution if available; raises an error otherwise."""
        if self._solution is None:
            msg = "No solution available. Ensure `solve` is called first."
            raise ValueError(msg)
        return self._solution


class KCentersSolver:
    def __init__(self, graph: nx.Graph) -> None:
        """
        Creates a solver for the k-centers problem on the given networkx graph.
        The graph may not be complete, and edge weights are used to represent distances.
        """
        self.graph = graph
        # Initialize distances helper
        self.distances = Distances(self.graph)
        logging.info(
            "KCentersSolver initialized with graph of %d nodes and %d edges",
            self.graph.number_of_nodes(),
            self.graph.number_of_edges(),
        )

    def solve_heur(self, k: int) -> list[NodeId]:
        """
        Calculate a heuristic solution to the k-centers problem.
        Returns the k selected centers as a list of node IDs.
        """
        logging.info("Starting heuristic computation for k=%d", k)
        # TODO: Implement me!
        all_nodes = list(self.graph.nodes)
        choosed_node = max(self.graph.nodes, key=lambda v: self.graph.degree(v))
        centers = [choosed_node]
        nodes_nearest_dist = {
            v: self.distances.dist(v, choosed_node) for v in all_nodes
        }

        for _ in range(k - 1):
            choosed_node = max(all_nodes, key=lambda u: nodes_nearest_dist[u])
            centers.append(choosed_node)
            for v in all_nodes:
                new_dist = self.distances.dist(choosed_node, v)
                if new_dist < nodes_nearest_dist[v]:
                    nodes_nearest_dist[v] = new_dist

        logging.info("Heuristic centers selected: %s", centers)
        return centers

    def solve(self, k: int) -> list[NodeId]:
        """
        Calculate the optimal solution to the k-centers problem for the given k.
        Returns the selected centers as a list of node IDs.
        """
        logging.info("Starting exact solve for k=%d", k)
        # Start with a heuristic solution
        centers = self.solve_heur(k)
        obj = self.distances.max_dist(centers)
        logging.info("Initial heuristic objective value: %f", obj)

        # TODO: Implement me!
        centers = KCenterDecisionVariant(self.distances, k, obj).solve()

        logging.info("Exact centers computed: %s", centers)
        logging.info("Final objective value: %f", obj)
        return centers
