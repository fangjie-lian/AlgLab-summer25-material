from dataclasses import dataclass
import logging
import typing

import gurobipy as gp
import networkx as nx
from gurobipy import GRB

# Set up basic logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Prevent double logging from gurobi
logging.getLogger("gurobipy").setLevel(logging.ERROR)

# Define types for implicit documentation
Vertex = typing.Hashable
Edge = tuple[Vertex, Vertex]


@dataclass
class Instance:
    """
    Unweighted Steiner Tree Instance
    -------------------------------
    This class represents an instance, consisting of a graph, and
    the list of terminals that have to be connected with as few edges
    as possible.
    """

    graph: nx.Graph
    terminals: list[Vertex]


class _EdgeVariables:
    """
    A helper class that manages the variables for the edges.
    Such a helper class turns out to be useful in many cases.
    """

    def __init__(self, G: nx.Graph, model: gp.Model):
        self._graph = G
        self._model = model
        self._vars = {
            (u, v): model.addVar(vtype=gp.GRB.BINARY, name=f"edge_{u}_{v}")
            for u, v in G.edges
        }
        # Log the number of edge variables created
        logging.info("Edge variables initialized: %d variables.", len(self._vars))

    def x(self, v: Vertex, w: Vertex) -> gp.Var:
        """
        Return variable for edge (v, w).
        """
        if (v, w) in self._vars:
            return self._vars[v, w]
        # If (v,w) was not found, try (w,v)
        return self._vars[w, v]

    def outgoing_edges(
        self, vertices: list[Vertex]
    ) -> typing.Iterable[tuple[Edge, gp.Var]]:
        """
        Return all edges & variables that are outgoing from the given vertices.
        """
        # Not super efficient, but efficient enough for our purposes.
        for (v, w), x in self._vars.items():
            if v in vertices and w not in vertices:
                yield (v, w), x
            elif w in vertices and v not in vertices:
                yield (w, v), x

    def incident_edges(self, v: Vertex) -> typing.Iterable[tuple[Edge, gp.Var]]:
        """
        Return all edges & variables that are incident to the given vertex.
        """
        for n in self._graph.neighbors(v):
            yield (v, n), self.x(v, n)

    def __iter__(self) -> typing.Iterator[tuple[Edge, gp.Var]]:
        """
        Iterate over all edges & variables.
        """
        return iter(self._vars.items())

    def as_graph(self, in_callback: bool = False) -> nx.Graph:
        """
        Return the current solution as a graph.
        """
        if in_callback:
            # If we are in a callback, we need to use the solution from the callback.
            used_edges = [vw for vw, x in self if self._model.cbGetSolution(x) > 0.5]
        else:
            # Otherwise, we can use the solution from the model.
            used_edges = [vw for vw, x in self if x.X > 0.5]
        return nx.Graph(used_edges)


class SteinerTreeSolver:
    """
    A simple solver for the Unweighted Steiner Tree problem.
    """

    def __init__(self, instance: Instance) -> None:
        logging.info("Initializing Steiner Tree Solver...")
        self.instance = instance
        self.model = gp.Model()
        logging.info("Gurobi model created.")
        self._edge_vars = _EdgeVariables(self.instance.graph, self.model)
        self._enforce_outgoing_edge_for_every_terminal()
        self._minimize_edges()

    def _enforce_outgoing_edge_for_every_terminal(self):
        if len(self.instance.terminals) <= 1:
            # Trivial instance, no need to add constraints
            return
        logging.info(
            "Adding constraints to enforce at least one outgoing edge for each terminal."
        )
        for t in self.instance.terminals:
            self.model.addConstr(
                gp.quicksum(x for _, x in self._edge_vars.incident_edges(t)) >= 1
            )
        logging.info(
            "Added constraints for %d terminals.", len(self.instance.terminals)
        )

    def _minimize_edges(self):
        logging.info("Setting objective to minimize the number of edges.")
        self.model.setObjective(sum(x for _, x in self._edge_vars), GRB.MINIMIZE)

    def lower_bound(self):
        """
        Return the current lower bound.
        """
        # Only works if the model has been optimized once.
        return self.model.ObjBound

    def solve(
        self, time_limit: float = 900, opt_tol: float = 0.0001
    ) -> nx.Graph | None:
        logging.info("Starting to solve the model...")
        logging.info("Time limit set to %f seconds.", time_limit)
        logging.info("Optimality tolerance (MIPGap) set to %f.", opt_tol)
        self.model.Params.TimeLimit = time_limit  # Limit the runtime
        self.model.Params.MIPGap = opt_tol  # Allowing a small optimality gap
        self.model.Params.NonConvex = 0  # Throw an error if the model is non-convex

        def callback(model, where):
            # This callback is called by Gurobi on various occasions, and
            # we can react to these occasions.
            if where == gp.GRB.Callback.MIPSOL:
                # We are in a new MIP solution. We can query the solution
                # and add additional constraints, if we want to.
                # We are going to enforce a leaving edge for every component
                # that contains only a part of the terminals.
                solution = self._edge_vars.as_graph(in_callback=True)
                comps = list(nx.connected_components(solution))
                if len(comps) == 1:
                    return  # solution is connected
                terminals = set(self.instance.terminals)
                for comp in comps:
                    terms_in_comp = terminals & comp
                    if not terms_in_comp:
                        continue  # the objective will remove this component by itself
                    if terms_in_comp != terminals:
                        # We have a component that contains some terminals but not all
                        # -> we need to add a constraint
                        model.cbLazy(
                            gp.quicksum(
                                x for _, x in self._edge_vars.outgoing_edges(comp)
                            )
                            >= 1
                        )
                        # Optionally, add debug logging here if necessary
                        # logging.debug("Added lazy constraint for component with terminals %s", terms_in_comp)

        self.model.Params.LazyConstraints = 1  # Enable lazy constraints
        self.model.optimize(callback)  # Pass the callback with the `solve` call

        # Log the outcome of the optimization
        if self.model.status == GRB.OPTIMAL:
            logging.info("Optimal solution found.")
            logging.info("Objective value: %f", self.model.ObjVal)
            return self._edge_vars.as_graph()
        if self.model.SolCount > 0:
            logging.info("Feasible solution found, but not proven optimal.")
            logging.info("Objective value: %f", self.model.ObjVal)
            # If the solution is not optimal, it may not be a tree. Apply DFS to get a tree.
            return nx.dfs_tree(self._edge_vars.as_graph())
        logging.warning("No feasible solution found within the time limit.")
        return None
