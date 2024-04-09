import itertools

def calculate_total_cost(assignment, cost_matrix):
    total_cost = 0
    for task, worker in enumerate(assignment):
        total_cost += cost_matrix[task][worker]
    return total_cost

def solve_assignment(cost_matrix):
    num_tasks = len(cost_matrix)
    num_workers = len(cost_matrix[0])

    # Generate all possible permutations of task-worker assignments
    assignments = itertools.permutations(range(num_workers), num_tasks)

    min_cost = float('inf')
    optimal_assignment = None

    # Iterate over all possible assignments to find the one with the minimum total cost
    for assignment in assignments:
        total_cost = calculate_total_cost(assignment, cost_matrix)
        if total_cost < min_cost:
            min_cost = total_cost
            optimal_assignment = assignment

    return min_cost, optimal_assignment

# Example cost matrix
cost_matrix = [
    [9, 2, 7],
    [8, 6, 5],
    [6, 7, 3]
]

min_cost, optimal_assignment = solve_assignment(cost_matrix)

print("Optimal Assignment:", optimal_assignment)
print("Minimum Total Cost:", min_cost)
