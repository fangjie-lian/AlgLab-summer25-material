import networkx as nx
import math

from data_schema import Solution, Donation
from database import TransplantDatabase
from ortools.sat.python.cp_model import FEASIBLE, OPTIMAL, CpModel, CpSolver


class CycleLimitingCrossoverTransplantSolver:
    def __init__(self, database: TransplantDatabase) -> None:
        """
        Constructs a new solver instance, using the instance data from the given database instance.
        :param Database database: The organ donor/recipients database.
        """

        self.database = database
        # TODO: Implement me!

        self.model = CpModel()
        self.solver = CpSolver()
        self.solver.parameters.log_search_progress = True

        self.donors = self.database.get_all_donors()
        self.recipients = self.database.get_all_recipients()
        self.recipient_map = {recipient.id: recipient for recipient in self.recipients}

        # Graphaufbau
        self.graph = nx.DiGraph()
        for recipient in self.recipients:
            self.graph.add_node(recipient.id)

        for recipient_to in self.recipients:
            compatible_donors = self.database.get_compatible_donors(recipient_to)
            if compatible_donors == None:
                continue
            for compatible_donor in compatible_donors:
                recipient_from = self.database.get_partner_recipient(compatible_donor)
                if recipient_from == None:
                    continue
                if recipient_to.id != recipient_from.id:  # Constraint 4 & 0
                    self.graph.add_edge(
                        recipient_from.id,
                        recipient_to.id,
                        link_donor=compatible_donor,
                    )  # recipient receive organ from compatible_donor, edge: partner_recipient->recipient

        self.cycles = list(nx.simple_cycles(self.graph, 3))
        self.x = {
            tuple(cycle): self.model.new_bool_var(f"x_{cycle}") for cycle in self.cycles
        }

        # Constraint 1:  A recipient can receive only one organ.
        for recipient in self.recipients:
            self.model.add(
                sum(
                    self.x[tuple(cycle)]
                    for cycle in self.cycles
                    if recipient.id in cycle
                )
                <= 1
            )

        # Constraint 2: A donor can donate only once.
        for donor in self.donors:
            cycles_with_donor = []
            for cycle in self.cycles:
                for index in range(len(cycle)):
                    next_recipient_id = cycle[(index + 1) % len(cycle)]
                    donor_in_cycle = self.graph.edges[
                        (cycle[index], next_recipient_id)
                    ]["link_donor"]
                    if donor == donor_in_cycle:
                        cycles_with_donor.append(cycle)
                        break
            self.model.add(
                sum(self.x[tuple(cycle)] for cycle in cycles_with_donor) <= 1
            )

        # Constraint 3: cycle_lenth >= 2
        for cycle in self.cycles:
            if len(cycle) == 1:
                self.model.add(self.x[tuple(cycle)] == 0)

        # Zielfkt.
        self.model.maximize(
            sum(self.x[tuple(cycle)] * len(cycle) for cycle in self.cycles)
        )

    def optimize(self, timelimit: float = math.inf) -> Solution:
        if timelimit <= 0.0:
            return Solution(donations=[])
        if timelimit < math.inf:
            self.solver.parameters.max_time_in_seconds = timelimit
        # TODO: Implement me!
        status = self.solver.Solve(self.model)
        assert status == OPTIMAL
        Donations = []
        for cycle in self.cycles:
            if self.solver.Value(self.x[tuple(cycle)]) == 1:
                for index in range(len(cycle)):
                    next_recipient_id = cycle[(index + 1) % len(cycle)]
                    recipient_to = self.recipient_map.get(next_recipient_id)
                    link_donor = self.graph.edges[(cycle[index], next_recipient_id)][
                        "link_donor"
                    ]
                    donation = Donation(donor=link_donor, recipient=recipient_to)
                    Donations.append(donation)

        return Solution(donations=Donations)
