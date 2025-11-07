from data_schema import Instance, Solution
from ortools.sat.python import cp_model


def solve(instance: Instance) -> Solution:
    """
    Implement your solver for the problem here!
    """
    numbers = instance.numbers
    model = cp_model.CpModel()
    x = [model.new_bool_var(f"x_{i}") for i in range(len(numbers))]
    # check for single_number_text
    if len(numbers) > 1:
        model.add(sum(x) == 2)
    else:
        return Solution(
            number_a=numbers[0],
            number_b=numbers[0],
            distance=abs(numbers[0] - numbers[0]),
        )

    """ 1. Ansatz: (aber nicht mehr linear)
    model.maximize(
        sum(x[i] * x[j] * abs(numbers[i] - numbers[j])
            for i in range(len(numbers))
            for j in range(i + 1, len(numbers)))
    )

    2. Ansatz: Hilfsvar y_ij = x[i] * x[j]
    """

    # 1. create (i, j) pair and their distance
    distance_pairs = []
    n = len(numbers)
    for i in range(n):
        for j in range(i + 1, n):
            distance_val = abs(numbers[i] - numbers[j])

            # 2. y_ij = x[i]*x[j]
            y_ij = model.new_bool_var(f"y_{i}_{j}")
            distance_pairs.append(y_ij * distance_val)
            # model.add_multiplication_equality(y_ij, [x[i], x[j]]) # vorsichtig mit add_multi_equality
            # xi und xj - besser, logische Operation
            model.add_bool_or(
                [x[i].Not(), x[j].Not(), y_ij]
            )  # (y_ij <= x[i] AND x[j]) <=> y_ij OR not(x[i] AND x[j]) <=> y_ij OR not x[i] or not x[j])
            model.add_implication(y_ij, x[i])  # y_ij => x[i]
            model.add_implication(y_ij, x[j])  # y_ij => x[j]

    # 3. linearize the objective function
    model.maximize(sum(distance_pairs))

    solver = cp_model.CpSolver()
    solver.parameters.log_search_progress = True
    status = solver.Solve(model)
    assert status == cp_model.OPTIMAL
    # get index i wenn x[i] == 1
    selected_indices = [i for i, var in enumerate(x) if solver.Value(var) == 1]
    # selected_indices = [index_of_a, index_of_b]
    return Solution(
        number_a=numbers[selected_indices[0]],
        number_b=numbers[selected_indices[1]],
        distance=abs(numbers[selected_indices[0]] - numbers[selected_indices[1]]),
    )
