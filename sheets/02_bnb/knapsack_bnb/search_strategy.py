"""
Search Strategy Module

The search strategy determines which node you explore next in the BnB tree to
improve your lower or upper bound as quickly as possible. Provide a `priority`
function that ranks open nodes; this class manages a priority queue accordingly.

You can implement breadth-first, depth-first, best-first, or any custom order
by supplying different priority functions.
"""

import queue
from typing import Callable, Iterator, Tuple, Any

from .bnb_nodes import BnBNode


class SearchStrategy:
    """
    Manage open BnB nodes in a priority queue.

    Args:
        priority: callable mapping a BnBNode to a comparable key.
                  Lower keys are explored first.
    """

    def __init__(self, priority: Callable[[BnBNode], Any]) -> None:
        self._priority = priority
        # use a counter to break ties by insertion order
        self._queue: queue.PriorityQueue[Tuple[Any, int, BnBNode]] = (
            queue.PriorityQueue()
        )
        self._counter = 0

    def enqueue(self, node: BnBNode) -> None:
        """
        Add `node` to the open-set with its priority key.
        Ties are broken by the order nodes were added.
        """
        self._queue.put((self._priority(node), self._counter, node))
        self._counter += 1

    def has_next(self) -> bool:
        """
        Return True if there are still nodes to explore.
        """
        return not self._queue.empty()

    def next(self) -> BnBNode:
        """
        Remove and return the next node by priority.

        Raises:
            ValueError: if no nodes remain.
        """
        if not self.has_next():
            raise ValueError("No more nodes to explore.")
        return self._queue.get()[2]

    def __len__(self) -> int:
        """
        Number of nodes currently in the queue.
        """
        return self._queue.qsize()

    def nodes_in_queue(self) -> Iterator[BnBNode]:
        """
        Iterator over nodes still in the queue (no removal).
        """
        return (item[2] for item in self._queue.queue)

    def upper_bound(self) -> float:
        """
        Return the highest upper_bound among queued nodes, or -inf if empty.

        Note: to get the global BnB upper bound, take the max of this
        and your best feasible solution value.
        """
        if not self.has_next():
            return float("-inf")
        return max(
            self.nodes_in_queue(),
            key=lambda n: n.relaxed_solution.upper_bound,
        ).relaxed_solution.upper_bound


# Default search order: you must supply your own `priority`.
# This stub returns a constant key.


# The node with the smallest priority value is selected first.
# If two nodes have the same priority value, the one that was created first is selected first.
# Maybe there is a better way than to do a first in first out?
def my_search_order(node: BnBNode) -> Any:
    # The returned value must be comparable.
    # It can be a number, a tuple, or any other comparable object.
    # If my_search_order(a)<my_search_order(b), then a is selected first.
    if node.relaxed_solution.value() == 0:
        return (float("inf"), node.node_id)
    else:
        return (1 / node.relaxed_solution.value(), node.node_id)
