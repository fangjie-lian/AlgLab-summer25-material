import logging
import math
from typing import Iterable

import networkx as nx

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
        return self._distances[u].get(v, math.inf)

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


class KCenterDecisionVariant:
    def __init__(self, distances: Distances, k: int) -> None:
        self.distances = distances
        logging.info("Initializing KCenterDecisionVariant for k=%d", k)
        # TODO: Implement me!
        # Solution model
        self._solution: list[NodeId] | None = None

    def limit_distance(self, limit: float) -> None:
        """Adds constraints to the SAT solver to ensure coverage within the given distance."""
        logging.info("Limiting to distance: %f", limit)
        # TODO: Implement me!

    def solve(self) -> list[NodeId] | None:
        """Solves the SAT problem and returns the list of selected nodes, if feasible."""
        logging.info("Attempting to solve the SAT formulation")
        # TODO: Implement me!
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
        centers = None
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
        logging.info("Exact centers computed: %s", centers)
        logging.info("Final objective value: %f", obj)
        return centers
