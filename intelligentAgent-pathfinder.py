""" 
Intelligent Agents and Solving Problems by Searching
Implementation of DFS, BFS, UCS, and A* Search Algorithms

Author: Sammy Eyongorock
"""

from collections import deque
import heapq
import random
import time
from typing import List, Tuple, Optional, Dict


def print_forest(forest):
    """Print the forest grid in a readable format."""
    for row in forest:
        print(" ".join(row))
    print()


def generate_forest_15x15(obstacle_prob=0.18, small_fire_prob=0.10,
                          large_fire_prob=0.06, seed=None):
    """
    Generate a 15x15 forest grid with:
    - Start 'S' at (0,0)
    - Goal '*' at (14,14)
    - Obstacles '#'
    - Small fires 'f'
    - Large fires 'F'
    - Open ground '.'

    Notes:
    - Probabilities are applied only to non-start/non-goal cells.
    - This function does NOT guarantee solvability.
    """
    if seed is not None:
        random.seed(seed)

    size = 15
    forest = [['.' for _ in range(size)] for _ in range(size)]
    forest[0][0] = 'S'
    forest[size - 1][size - 1] = '*'

    for r in range(size):
        for c in range(size):
            if (r, c) == (0, 0) or (r, c) == (size - 1, size - 1):
                continue

            x = random.random()
            if x < obstacle_prob:
                forest[r][c] = '#'
            elif x < obstacle_prob + large_fire_prob:
                forest[r][c] = 'F'
            elif x < obstacle_prob + large_fire_prob + small_fire_prob:
                forest[r][c] = 'f'
            else:
                forest[r][c] = '.'

    return forest


class SearchAgent:
    """Agent that navigates through a forest using various search algorithms."""

    def __init__(self, start: Tuple[int, int], goal: Tuple[int, int], grid: List[List[str]]):
        self.start = start
        self.goal = goal
        self.grid = grid
        self.rows = len(grid)
        self.cols = len(grid[0])

        # Direction vectors: Up, Down, Left, Right (consistent ordering)
        self.directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.direction_names = ['Up', 'Down', 'Left', 'Right']

    def is_valid(self, position: Tuple[int, int]) -> bool:
        """Check if a position is within bounds and not an obstacle."""
        r, c = position
        return (0 <= r < self.rows) and (0 <= c < self.cols) \
            and (self.grid[r][c] != '#')

    def get_cost(self, position: Tuple[int, int]) -> int:
        """Return the traversal cost for stepping into a given cell."""
        cell = self.grid[position[0]][position[1]]
        if cell == 'f':  # small fire
            return 3
        elif cell == 'F':  # large fire
            return 5
        else:
            return 1  # open ground, start, or goal

    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get valid neighboring positions in consistent order."""
        r, c = position
        neighbors = []
        for dr, dc in self.directions:
            new_pos = (r + dr, c + dc)
            if self.is_valid(new_pos):
                neighbors.append(new_pos)
        return neighbors

    def reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from start to current using came_from dictionary."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def calculate_path_cost(self, path: List[Tuple[int, int]]) -> int:
        """Calculate total cost of a path."""
        if not path:
            return 0
        # Start cell has no entry cost
        return sum(self.get_cost(pos) for pos in path[1:])

    def print_path(self, path: List[Tuple[int, int]]):
        """Utility to display the path on the grid (P marks the path)."""
        temp_grid = [row[:] for row in self.grid]
        for (r, c) in path[1:-1]:  # exclude start and goal from marking
            temp_grid[r][c] = 'P'
        for row in temp_grid:
            print(" ".join(row))
        print()

    def dfs(self) -> Optional[Tuple[int, int, int]]:
        """
        Depth-First Search implementation using a stack.
        Returns: (path_length, nodes_expanded, total_cost) or None if no path exists.
        """
        stack = [(self.start, [self.start])]
        visited = set()
        nodes_expanded = 0

        while stack:
            current, path = stack.pop()

            # Skip if already visited
            if current in visited:
                continue

            visited.add(current)
            nodes_expanded += 1

            # Check if goal is reached
            if current == self.goal:
                path_length = len(path)
                total_cost = self.calculate_path_cost(path)
                return (path_length, nodes_expanded, total_cost, path)

            # Add neighbors to stack in reverse order to maintain Up, Down, Left, Right priority
            neighbors = self.get_neighbors(current)
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    stack.append((neighbor, path + [neighbor]))

        # No path found
        return None

    def bfs(self) -> Optional[Tuple[int, int, int]]:
        """
        Breadth-First Search implementation using a queue.
        Returns: (path_length, nodes_expanded, total_cost) or None if no path exists.
        """
        queue = deque([(self.start, [self.start])])
        visited = {self.start}
        nodes_expanded = 0

        while queue:
            current, path = queue.popleft()
            nodes_expanded += 1

            # Check if goal is reached
            if current == self.goal:
                path_length = len(path)
                total_cost = self.calculate_path_cost(path)
                return (path_length, nodes_expanded, total_cost, path)

            # Add neighbors to queue
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        # No path found
        return None

    def ucs(self) -> Optional[Tuple[int, int, int]]:
        """
        Uniform Cost Search implementation using a priority queue.
        Returns: (path_length, nodes_expanded, total_cost) or None if no path exists.
        """
        # Priority queue: (cost, counter, position, path)
        # Counter ensures FIFO ordering for equal costs
        counter = 0
        priority_queue = [(0, counter, self.start, [self.start])]
        visited = set()
        nodes_expanded = 0

        # Track best cost to reach each node
        best_cost = {self.start: 0}

        while priority_queue:
            cost, _, current, path = heapq.heappop(priority_queue)

            # Skip if already visited
            if current in visited:
                continue

            visited.add(current)
            nodes_expanded += 1

            # Check if goal is reached
            if current == self.goal:
                path_length = len(path)
                total_cost = self.calculate_path_cost(path)
                return (path_length, nodes_expanded, total_cost, path)

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    new_cost = cost + self.get_cost(neighbor)

                    # Only add if this is a better path
                    if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                        best_cost[neighbor] = new_cost
                        counter += 1
                        heapq.heappush(priority_queue,
                                       (new_cost, counter, neighbor, path + [neighbor]))

        # No path found
        return None

    def astar(self, heuristic=None) -> Optional[Tuple[int, int, int]]:
        """
        A* Search implementation using f(n) = g(n) + h(n).
        Returns: (path_length, nodes_expanded, total_cost) or None if no path exists.
        """
        # Default heuristic: Manhattan distance
        if heuristic is None:
            def heuristic(pos):
                return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

        # Priority queue: (f_cost, g_cost, counter, position, path)
        # Using g_cost as secondary key for tie-breaking (prefer higher g, closer to goal)
        counter = 0
        h_start = heuristic(self.start)
        priority_queue = [(h_start, 0, counter, self.start, [self.start])]
        visited = set()
        nodes_expanded = 0

        # Track best g-cost to reach each node
        best_g_cost = {self.start: 0}

        while priority_queue:
            f_cost, g_cost, _, current, path = heapq.heappop(priority_queue)

            # Skip if already visited
            if current in visited:
                continue

            visited.add(current)
            nodes_expanded += 1

            # Check if goal is reached
            if current == self.goal:
                path_length = len(path)
                total_cost = self.calculate_path_cost(path)
                return (path_length, nodes_expanded, total_cost, path)

            # Explore neighbors
            for neighbor in self.get_neighbors(current):
                if neighbor not in visited:
                    new_g_cost = g_cost + self.get_cost(neighbor)

                    # Only add if this is a better path
                    if neighbor not in best_g_cost or new_g_cost < best_g_cost[neighbor]:
                        best_g_cost[neighbor] = new_g_cost
                        h_cost = heuristic(neighbor)
                        new_f_cost = new_g_cost + h_cost
                        counter += 1
                        heapq.heappush(priority_queue,
                                       (new_f_cost, new_g_cost, counter, neighbor,
                                        path + [neighbor]))

        # No path found
        return None


def test_search_agent(agent: SearchAgent, show_paths: bool = False) -> Dict:
    """Run all search algorithms on the agent's grid and print results."""
    results = {}

    print("\n" + "=" * 60)
    print("SEARCH ALGORITHM COMPARISON")
    print("=" * 60)

    # Test BFS
    print("\n--- BREADTH-FIRST SEARCH (BFS) ---")
    res_bfs = agent.bfs()
    results['BFS'] = res_bfs
    if res_bfs:
        print(f"✓ Path Found!")
        print(f"  Path Length: {res_bfs[0]} steps")
        print(f"  Nodes Expanded: {res_bfs[1]}")
        print(f"  Total Cost: {res_bfs[2]}")
        if show_paths:
            print("\nPath Visualization:")
            agent.print_path(res_bfs[3])
    else:
        print("✗ No path found")

    # Test DFS
    print("\n--- DEPTH-FIRST SEARCH (DFS) ---")
    res_dfs = agent.dfs()
    results['DFS'] = res_dfs
    if res_dfs:
        print(f"✓ Path Found!")
        print(f"  Path Length: {res_dfs[0]} steps")
        print(f"  Nodes Expanded: {res_dfs[1]}")
        print(f"  Total Cost: {res_dfs[2]}")
        if show_paths:
            print("\nPath Visualization:")
            agent.print_path(res_dfs[3])
    else:
        print("✗ No path found")

    # Test UCS
    print("\n--- UNIFORM COST SEARCH (UCS) ---")
    res_ucs = agent.ucs()
    results['UCS'] = res_ucs
    if res_ucs:
        print(f"✓ Path Found!")
        print(f"  Path Length: {res_ucs[0]} steps")
        print(f"  Nodes Expanded: {res_ucs[1]}")
        print(f"  Total Cost: {res_ucs[2]}")
        if show_paths:
            print("\nPath Visualization:")
            agent.print_path(res_ucs[3])
    else:
        print("✗ No path found")

    # Test A*
    print("\n--- A* SEARCH ---")
    res_astar = agent.astar()
    results['A*'] = res_astar
    if res_astar:
        print(f"✓ Path Found!")
        print(f"  Path Length: {res_astar[0]} steps")
        print(f"  Nodes Expanded: {res_astar[1]}")
        print(f"  Total Cost: {res_astar[2]}")
        if show_paths:
            print("\nPath Visualization:")
            agent.print_path(res_astar[3])
    else:
        print("✗ No path found")

    print("\n" + "=" * 60)

    return results


def main():
    """Main function to run test scenarios."""
    print("\n" + "=" * 70)
    print(" " * 15 + "PATHFINDER SEARCH AGENT")
    print(" " * 10 + "SYSC 4416 - Mini Project 1")
    print("=" * 70)

    # Test Scenario 1: Solvable maze with fires (seed=42)
    print("\n\n### TEST SCENARIO 1: Solvable Maze with Fires ###")
    print("Seed: 42")
    forest_1 = generate_forest_15x15(seed=42)
    print("\nForest Grid:")
    print_forest(forest_1)
    agent_1 = SearchAgent((0, 0), (14, 14), forest_1)
    results_1 = test_search_agent(agent_1, show_paths=False)

    # Test Scenario 2: Another solvable maze with different layout (seed=123)
    print("\n\n### TEST SCENARIO 2: Different Solvable Maze ###")
    print("Seed: 123")
    forest_2 = generate_forest_15x15(seed=123)
    print("\nForest Grid:")
    print_forest(forest_2)
    agent_2 = SearchAgent((0, 0), (14, 14), forest_2)
    results_2 = test_search_agent(agent_2, show_paths=False)

    # Test Scenario 3: High obstacle density (potential unsolvable)
    print("\n\n### TEST SCENARIO 3: High Obstacle Density ###")
    print("Seed: 999, Obstacle Probability: 0.35")
    forest_3 = generate_forest_15x15(obstacle_prob=0.35, seed=999)
    print("\nForest Grid:")
    print_forest(forest_3)
    agent_3 = SearchAgent((0, 0), (14, 14), forest_3)
    results_3 = test_search_agent(agent_3, show_paths=False)

    # Test Scenario 4: Fire-heavy maze
    print("\n\n### TEST SCENARIO 4: Fire-Heavy Maze ###")
    print("Seed: 555, Small Fire: 0.15, Large Fire: 0.10")
    forest_4 = generate_forest_15x15(obstacle_prob=0.15,
                                     small_fire_prob=0.15,
                                     large_fire_prob=0.10,
                                     seed=555)
    print("\nForest Grid:")
    print_forest(forest_4)
    agent_4 = SearchAgent((0, 0), (14, 14), forest_4)
    results_4 = test_search_agent(agent_4, show_paths=False)

    # Test Scenario 5: Explicitly unsolvable maze
    print("\n\n### TEST SCENARIO 5: Unsolvable Maze ###")
    print("Manually constructed to block all paths")

    # Start with an open grid
    forest_5 = [['.' for _ in range(15)] for _ in range(15)]
    forest_5[0][0] = 'S'
    forest_5[14][14] = '*'

    # Create a solid horizontal wall of obstacles
    for c in range(15):
        forest_5[7][c] = '#'

    print("\nForest Grid:")
    print_forest(forest_5)

    agent_5 = SearchAgent((0, 0), (14, 14), forest_5)
    results_5 = test_search_agent(agent_5, show_paths=False)

    print("\n\n" + "=" * 70)
    print("All test scenarios completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
