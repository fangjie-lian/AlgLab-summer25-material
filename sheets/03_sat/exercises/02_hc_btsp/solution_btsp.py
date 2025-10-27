import math
import logging
from enum import Enum
import bisect

import networkx as nx
from _timer import Timer
from solution_hamiltonian import HamiltonianCycleModel
from pysat.solvers import Solver as SATSolver


# Configure logging to show INFO messages
logging.basicConfig(level=logging.INFO)


class SearchStrategy(Enum):
    """
    Different search strategies for the solver.
    """

    SEQUENTIAL_UP = 1  # Try smallest possible k first.
    SEQUENTIAL_DOWN = 2  # Try any improvement.
    BINARY_SEARCH = 3  # Try a binary search for the optimal k.

    def __str__(self):
        return self.name.title()

    @staticmethod
    def from_str(s: str):
        return SearchStrategy[s.upper()]


class BottleneckTSPSolver:
    def __init__(self, graph: nx.Graph) -> None:
        """
        Creates a solver for the Bottleneck Traveling Salesman Problem on the given networkx graph.
        You can assume that the input graph is complete, so all nodes are neighbors.
        The distance between two neighboring nodes is a numeric value (int / float), saved as
        an edge data parameter called "weight".
        There are multiple ways to access this data, and networkx also implements
        several algorithms that automatically make use of this value.
        Check the networkx documentation for more information!
        """
        # Log initialization details
        logging.info(
            "Initializing BottleneckTSPSolver with %d nodes and %d edges...",
            graph.number_of_nodes(),
            graph.number_of_edges(),
        )
        self.graph = graph
        # TODO: Implement me!
        edges_with_weight = (
            e for e in graph.edges(data="weight") if e[2] is not None
        )  # (edge, weight) = (u, v, weight)
        self.sorted_edge = sorted(edges_with_weight, key=lambda e: e[2])
        self._solution: list[tuple[int, int]] | None = None
        self._tmp_solution: list[tuple[int, int]] | None = None
        # Log initialization completion
        logging.info("BottleneckTSPSolver initialized successfully!")

    def lower_bound(self) -> float:
        # TODO: Implement me!
        if len(self._tmp_solution) == 0:
            return float("inf")
        max = 0
        for e in self._tmp_solution:
            u, v = e
            w = self.graph[u][v].get("weight")
            if w > max:
                max = w
        return max

    def optimize_bottleneck(
        self,
        time_limit: float = math.inf,
        search_strategy: SearchStrategy = SearchStrategy.BINARY_SEARCH,
    ) -> list[tuple[int, int]] | None:
        """
        Find the optimal bottleneck tsp tour.
        """
        # Initialize timer
        self.timer = Timer(time_limit)
        logging.info("Timer initialized with limit %f seconds", time_limit)

        # TODO: Implement me!
        weights = [w for (_u, _v, w) in self.sorted_edge]
        # if (search_strategy == SearchStrategy.BINARY_SEARCH):
        a = 0
        b = len(self.sorted_edge) - 1
        while a <= b:
            i = a + (b - a) // 2
            edges_false = [(u, v, w) for (u, v, w) in self.sorted_edge[i + 1 :]]
            self._tmp_solution = HamiltonianCycleModel(self.graph).solve(edges_false)
            if self._tmp_solution is None:
                a = i + 1
                continue
            self._solution = self._tmp_solution
            max_dist = self.lower_bound()
            print("max_dist=", max_dist)
            b = bisect.bisect_left(weights, max_dist) - 1
        # if (search_strategy == SearchStrategy.SEQUENTIAL_UP):
        #     index = len(weights)
        #     while True:
        #         for i in range(len(weights)):
        #             if i != 0:
        #                 edges_false = [(u, v, w) for (u, v, w) in self.sorted_edge[index + 1 :]]
        #                 self._tmp_solution = self.model.solve(edges_false)
        #             else:
        #                 self._tmp_solution = self.model.solve()
        #             if self._tmp_solution is None:
        #                 continue
        #             self._solution = self._tmp_solution
        #             index = bisect.bisect_left(weights, self.lower_bound()) - 1
        #             weights = weights[:index+1]

        return self._solution
