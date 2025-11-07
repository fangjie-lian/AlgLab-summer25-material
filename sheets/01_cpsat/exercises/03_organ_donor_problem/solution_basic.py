import networkx as nx
import math

from data_schema import Solution, Donation
from database import TransplantDatabase
from ortools.sat.python.cp_model import FEASIBLE, OPTIMAL, CpModel, CpSolver


class CrossoverTransplantSolver:
    def __init__(self, database: TransplantDatabase) -> None:
        """
        Constructs a new solver instance, using the instance data from the given database instance.
        :param Database database: The organ donor/recipients database.
        """
        self.database = database
        # TODO: Implement me!
        self.donors = self.database.get_all_donors()
        self.recipients = self.database.get_all_recipients()
        self.recipient_map = {recipient.id: recipient for recipient in self.recipients}

        self.model = CpModel()
        self.solver = CpSolver()
        self.solver.parameters.log_search_progress = True

        # Graphaufbau
        self.graph = nx.DiGraph()
        for recipient in self.recipients:
            self.graph.add_node(recipient.id)

        for recipient_to in self.recipients:
            compatible_donors = self.database.get_compatible_donors(recipient_to)
            if compatible_donors == None:
                continue
            for compatible_donor in compatible_donors:
                # Constraint 0: keine uncompatible Donation
                recipient_from = self.database.get_partner_recipient(compatible_donor)
                if recipient_from == None:
                    continue
                if recipient_to.id != recipient_from.id:
                    # Constraint 4: ∃(di, rj) als pair: x[i][j] = 0
                    self.graph.add_edge(
                        recipient_from.id,
                        recipient_to.id,
                        link_donor=compatible_donor,
                    )  # recipient receive organ from compatible_donor, edge: partner_recipient->recipient

        self.x = {
            (recipient_from, recipient_to): self.model.new_bool_var(
                f"x_{recipient_from}_{recipient_to}"
            )
            for (recipient_from, recipient_to) in self.graph.edges
        }

        # Constraint 1: A donor can donate only once.
        for donor in self.donors:
            edges_with_donor = [
                (recipient_from, recipient_to)
                for (recipient_from, recipient_to) in self.graph.edges
                if self.graph[recipient_from][recipient_to]["link_donor"].id == donor.id
            ]
            if edges_with_donor:  # nicht leer -> Constraints hinzuzufügen
                self.model.add(sum(self.x[edge] for edge in edges_with_donor) <= 1)

        for recipient in self.recipients:
            outgoing_num = sum(
                self.x[(recipient.id, r2_id)]
                for r2_id in self.graph.successors(recipient.id)
            )
            incoming_num = sum(
                self.x[(r1_id, recipient.id)]
                for r1_id in self.graph.predecessors(recipient.id)
            )
            # Constraint 2: A recipient can receive only one organ.
            # (<=1 ausgehende Kante)
            self.model.add(outgoing_num <= 1)
            # Constraint 5: If a recipient has multiple willing donors, only one of them is willing to donate in the final solution.
            # (<=1 eingehende Kante)
            self.model.add(incoming_num <= 1)
            # Constraint 3: A donor is willing to donate only if their associated recipient receives an organ in exchange.
            # (eingehende Kante = ausgehende Kante = 0 oder 1)
            self.model.add(outgoing_num == incoming_num)

        # Zielfkt.
        self.model.maximize(sum(self.x[edge] for edge in self.graph.edges))

    def optimize(self, timelimit: float = math.inf) -> Solution:
        """
        Solves the constraint programming model and returns the optimal solution (if found within time limit).
        :param timelimit: The maximum time limit for the solver.
        :return: A list of Donation objects representing the best solution, or None if no solution was found.
        """
        if timelimit <= 0.0:
            return Solution(donations=[])
        if timelimit < math.inf:
            self.solver.parameters.max_time_in_seconds = timelimit
        # TODO: Implement me!
        status = self.solver.Solve(self.model)
        assert status == OPTIMAL
        Donations = []
        for recipient_from, recipient_to in self.graph.edges:
            if self.solver.Value(self.x[(recipient_from, recipient_to)]) == 1:
                donor = self.graph[recipient_from][recipient_to]["link_donor"]
                recipient = self.recipient_map.get(recipient_to)
                donation = Donation(donor=donor, recipient=recipient)
                Donations.append(donation)

        return Solution(donations=Donations)
