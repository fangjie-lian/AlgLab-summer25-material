import math

from data_schema import Solution, Donation
from database import TransplantDatabase
from ortools.sat.python.cp_model import OPTIMAL, CpModel, CpSolver


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

        self.model = CpModel()
        self.solver = CpSolver()
        self.solver.parameters.log_search_progress = True

        # Achtung: id ≠ index -> dictionary statt array
        self.x = {}
        for donor in self.donors:
            for recipient in self.recipients:
                self.x[(donor.id, recipient.id)] = self.model.new_bool_var(
                    f"x_{donor.id}_{recipient.id}"
                )  # self.x[(donor.id, recipient.id)] == 1 bedeutet, recipient_id akzeptiert donor_id

        for donor in self.donors:
            compatible_recipients = self.database.get_compatible_recipients(donor)
            # Constraint 0: keine uncompatible Donation (debugging - zu langsam)
            for recipient in self.recipients:
                if recipient not in compatible_recipients:
                    self.model.add(self.x[(donor.id, recipient.id)] == 0)

            # Constraint 1: A donor can only donate once.
            self.model.add(
                sum(self.x[(donor.id, recipient.id)] for recipient in self.recipients)
                <= 1
            )

            partner_recipient = self.database.get_partner_recipient(donor)
            # Constraint 4: ∃(di, rj) als pair: x[i][j] = 0
            self.model.add(self.x[(donor.id, partner_recipient.id)] == 0)

            # Constraint 3: A donor is willing to donate only if their associated recipient receives an organ in exchange.
            # ∀i: ∃j: x[i][j] == 1 => k ist recipient von i ∃j1: x[j1][k] == 1
            # Constraint 3 neu: sum(x[i][j] for j in range(recipients_num)) <= sum (x[k][i] for k in range(doners_num))
            # if self.x[i][j]: ### falsch
            compatible_donors = self.database.get_compatible_donors(partner_recipient)
            if_receive = sum(
                self.x[(donor1.id, partner_recipient.id)]
                for donor1 in compatible_donors
            )
            if_donate = sum(
                self.x[(donor.id, recipient.id)] for recipient in compatible_recipients
            )
            self.model.add(
                if_donate <= if_receive
            )  # nicht = (andere Donors können donate)

        for recipient in self.recipients:
            compatible_donors = self.database.get_compatible_donors(recipient)
            if_receive = sum(
                self.x[(donor.id, recipient.id)] for donor in compatible_donors
            )

            # Constraint 2: A recipient can only receive one organ.
            self.model.add(
                sum(self.x[(donor.id, recipient.id)] for donor in self.donors) <= 1
            )
            partner_donors = self.database.get_partner_donors(recipient)
            donation_sum = 0
            for partner_donor in partner_donors:
                compatible_recipients = self.database.get_compatible_recipients(
                    partner_donor
                )
                donation_sum += sum(
                    self.x[(partner_donor.id, compatible_recipient.id)]
                    for compatible_recipient in compatible_recipients
                )
            # Constraint 5 (debugging): If a recipient has multiple willing donors, only one of them is willing to donate in the final solution.
            self.model.add(if_receive == donation_sum)

        # Zielfkt.
        self.model.maximize(
            sum(
                self.x[(donor.id, recipient.id)]
                for donor in self.donors
                for recipient in self.recipients
            )
        )

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
        for recipient in self.recipients:
            for donor in self.database.get_compatible_donors(recipient):
                if self.solver.Value(self.x[(donor.id, recipient.id)]) == 1:
                    donation = Donation(donor=donor, recipient=recipient)
                    Donations.append(donation)

        return Solution(donations=Donations)
