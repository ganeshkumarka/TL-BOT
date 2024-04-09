import sys

def nearest_neighbor(distance_matrix):
    num_cities = len(distance_matrix)
    visited = [False] * num_cities
    tour = []
    current_city = 0

    for _ in range(num_cities - 1):
        next_city = None
        min_distance = sys.maxsize

        for city in range(num_cities):
            if not visited[city] and distance_matrix[current_city][city] < min_distance:
                next_city = city
                min_distance = distance_matrix[current_city][city]

        tour.append(next_city)
        visited[next_city] = True
        current_city = next_city

    # Return to the starting city to complete the tour
    tour.append(tour[0])

    return tour

def total_distance(tour, distance_matrix):
    total = 0
    for i in range(len(tour) - 1):
        total += distance_matrix[tour[i]][tour[i+1]]
    return total

# Example distance matrix
distance_matrix = [
    [0, 10, 15, 20],
    [10, 0, 35, 25],
    [15, 35, 0, 30],
    [20, 25, 30, 0]
]

# Find the shortest tour using nearest neighbor heuristic
shortest_tour = nearest_neighbor(distance_matrix)

# Calculate the total distance of the tour
total_dist = total_distance(shortest_tour, distance_matrix)

print("Shortest tour:", shortest_tour)
print("Total distance:", total_dist)
