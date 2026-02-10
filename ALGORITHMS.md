# Search Algorithms - Technical Documentation

This document provides in-depth technical details about the four search algorithms implemented in the Pathfinder project.

## Table of Contents

- [Overview](#overview)
- [Problem Formulation](#problem-formulation)
- [Algorithm Details](#algorithm-details)
  - [Depth-First Search (DFS)](#depth-first-search-dfs)
  - [Breadth-First Search (BFS)](#breadth-first-search-bfs)
  - [Uniform Cost Search (UCS)](#uniform-cost-search-ucs)
  - [A* Search](#a-search)
- [Heuristic Functions](#heuristic-functions)
- [Performance Comparison](#performance-comparison)
- [Implementation Notes](#implementation-notes)

## Overview

All algorithms solve the same pathfinding problem but use different strategies. The key differences are:

| Property | DFS | BFS | UCS | A* |
|----------|-----|-----|-----|-----|
| **Data Structure** | Stack | Queue | Priority Queue | Priority Queue |
| **Expansion Order** | Deepest | Shallowest | Lowest cost | Lowest f(n) |
| **Complete** | Yes* | Yes | Yes | Yes |
| **Optimal** | No | Yes** | Yes | Yes*** |
| **Time Complexity** | O(b^m) | O(b^d) | O(b^(C*/ε)) | Depends on h |
| **Space Complexity** | O(bm) | O(b^d) | O(b^(C*/ε)) | O(b^d) |

\* Finite state spaces only  
\** Uniform costs only  
\*** With admissible heuristic

## Problem Formulation

### State Space

- **State**: Position (row, col) in the grid
- **Initial State**: Start position S at (0, 0)
- **Goal State**: Goal position * at (14, 14)
- **Actions**: Move {Up, Down, Left, Right}
- **Transition Model**: Move to adjacent cell if valid
- **Action Cost**: Depends on cell type (1, 3, or 5)

### State Space Properties

```
Branching factor (b): ≤ 4 (max neighbors)
Maximum depth (m): 225 (entire 15×15 grid)
Solution depth (d): Typically 28-30 (Manhattan distance + detours)
```

## Algorithm Details

### Depth-First Search (DFS)

#### Strategy

DFS explores as deeply as possible along each branch before backtracking. It uses a LIFO (Last In, First Out) stack to maintain the frontier.

#### Pseudocode

```python
function DFS(start, goal):
    stack ← [(start, [start])]
    visited ← empty set
    
    while stack is not empty:
        current, path ← stack.pop()
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == goal:
            return path
            
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                stack.push((neighbor, path + [neighbor]))
    
    return None  # No path found
```

#### Key Characteristics

**Advantages:**
- Low memory usage: O(bm) where b is branching factor, m is max depth
- Simple implementation
- Fast for deep solutions

**Disadvantages:**
- Not optimal - may find very long paths
- Can get stuck in infinite loops without cycle detection
- May explore most of the state space unnecessarily

**When to Use:**
- Memory is severely constrained
- All solutions have similar depth
- Any solution is acceptable (don't need optimal)

#### Implementation Notes

```python
# Our implementation uses consistent neighbor ordering
directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Up, Down, Left, Right

# Neighbors added in reverse order to maintain priority
for neighbor in reversed(neighbors):
    stack.append((neighbor, path + [neighbor]))
```

### Breadth-First Search (BFS)

#### Strategy

BFS explores the state space level by level, expanding all nodes at depth d before moving to depth d+1. It uses a FIFO (First In, First Out) queue.

#### Pseudocode

```python
function BFS(start, goal):
    queue ← Queue([(start, [start])])
    visited ← {start}
    
    while queue is not empty:
        current, path ← queue.dequeue()
        
        if current == goal:
            return path
            
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                visited.add(neighbor)
                queue.enqueue((neighbor, path + [neighbor]))
    
    return None  # No path found
```

#### Key Characteristics

**Advantages:**
- Complete: Always finds a solution if one exists
- Optimal for uniform costs: Finds shortest path in number of steps
- Systematic exploration: Never misses a solution

**Disadvantages:**
- High memory usage: O(b^d) where d is solution depth
- Ignores action costs: Not optimal when costs vary
- Can be slow for deep solutions

**When to Use:**
- All actions have equal cost
- Solutions are relatively shallow
- Need guaranteed shortest path (in steps)

#### Implementation Notes

```python
# Mark nodes as visited when enqueued (not when expanded)
# This prevents adding the same node multiple times
visited.add(neighbor)
queue.append((neighbor, path + [neighbor]))
```

### Uniform Cost Search (UCS)

#### Strategy

UCS (also known as Dijkstra's algorithm) always expands the frontier node with the lowest path cost g(n). It guarantees finding the minimum-cost path.

#### Pseudocode

```python
function UCS(start, goal):
    priority_queue ← PriorityQueue()
    priority_queue.push((0, start, [start]))
    visited ← empty set
    best_cost ← {start: 0}
    
    while priority_queue is not empty:
        cost, current, path ← priority_queue.pop()
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == goal:
            return path
            
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                new_cost ← cost + action_cost(neighbor)
                
                if neighbor not in best_cost or new_cost < best_cost[neighbor]:
                    best_cost[neighbor] ← new_cost
                    priority_queue.push((new_cost, neighbor, path + [neighbor]))
    
    return None  # No path found
```

#### Key Characteristics

**Advantages:**
- Optimal: Guaranteed to find minimum-cost path
- Handles varying action costs correctly
- Complete with positive costs

**Disadvantages:**
- Higher space complexity: O(b^(C*/ε))
- Can be slow without heuristic guidance
- Expands many nodes in all directions

**When to Use:**
- Action costs vary
- Need guaranteed optimal solution
- No good heuristic available

#### Cost Function

```
g(n) = actual cost from start to node n

Priority = g(n)
```

#### Implementation Notes

```python
# Use counter for tie-breaking to ensure FIFO ordering
counter = 0
heapq.heappush(priority_queue, (cost, counter, node, path))
counter += 1

# Track best cost to avoid redundant expansions
if neighbor not in best_cost or new_cost < best_cost[neighbor]:
    best_cost[neighbor] = new_cost
    # Add to queue
```

### A* Search

#### Strategy

A* combines UCS with heuristic guidance. It expands nodes based on f(n) = g(n) + h(n), where:
- g(n) = actual cost from start to n (like UCS)
- h(n) = estimated cost from n to goal (heuristic)

#### Pseudocode

```python
function A_STAR(start, goal, heuristic):
    priority_queue ← PriorityQueue()
    h_start ← heuristic(start)
    priority_queue.push((h_start, 0, start, [start]))
    visited ← empty set
    best_cost ← {start: 0}
    
    while priority_queue is not empty:
        f, g, current, path ← priority_queue.pop()
        
        if current in visited:
            continue
            
        visited.add(current)
        
        if current == goal:
            return path
            
        for neighbor in get_neighbors(current):
            if neighbor not in visited:
                new_g ← g + action_cost(neighbor)
                
                if neighbor not in best_cost or new_g < best_cost[neighbor]:
                    best_cost[neighbor] ← new_g
                    h ← heuristic(neighbor)
                    new_f ← new_g + h
                    priority_queue.push((new_f, new_g, neighbor, path + [neighbor]))
    
    return None  # No path found
```

#### Key Characteristics

**Advantages:**
- Optimal with admissible heuristic
- More efficient than UCS (fewer nodes expanded)
- Proven to be optimally efficient
- Flexible (can use different heuristics)

**Disadvantages:**
- Requires domain knowledge for heuristic
- Memory usage can be high: O(b^d)
- Heuristic must be carefully designed

**When to Use:**
- Good heuristic available
- Need optimal solution efficiently
- State space is large

#### Evaluation Function

```
f(n) = g(n) + h(n)

where:
  g(n) = actual cost from start to n
  h(n) = estimated cost from n to goal
  
Priority = f(n)
```

#### Optimality Conditions

A* is optimal if h(n) is **admissible**:
```
h(n) ≤ h*(n)  for all n

where h*(n) is the true cost from n to goal
```

A* is more efficient if h(n) is also **consistent** (monotonic):
```
h(n) ≤ c(n, n') + h(n')  for all n, n'

where c(n, n') is the cost of the action from n to n'
```

## Heuristic Functions

### Manhattan Distance (L1 Norm)

**Formula:**
```python
h(x, y) = |x - goal_x| + |y - goal_y|
```

**Properties:**
- ✅ Admissible: Never overestimates (each move costs ≥ 1)
- ✅ Consistent: |h(n) - h(n')| ≤ c(n, n')
- ✅ Efficient: O(1) computation
- ✅ Accurate for grid-based navigation

**Example:**
```
Position: (3, 5)
Goal: (10, 12)

h = |3 - 10| + |5 - 12|
  = 7 + 7
  = 14
```

### Euclidean Distance (L2 Norm)

**Formula:**
```python
h(x, y) = sqrt((x - goal_x)² + (y - goal_y)²)
```

**Properties:**
- ✅ Admissible (if diagonal moves not allowed)
- ❌ Less informed than Manhattan for grid worlds
- More expensive to compute (sqrt operation)

### Other Potential Heuristics

**Diagonal Distance:**
```python
dx = abs(x - goal_x)
dy = abs(y - goal_y)
h = max(dx, dy) + (sqrt(2) - 1) * min(dx, dy)
```

**Zero Heuristic:**
```python
h(x, y) = 0  # A* becomes UCS
```

## Performance Comparison

### Theoretical Analysis

| Metric | DFS | BFS | UCS | A* |
|--------|-----|-----|-----|-----|
| **Time Complexity** | O(b^m) | O(b^d) | O(b^(C*/ε)) | O(b^d) |
| **Space Complexity** | O(bm) | O(b^d) | O(b^(C*/ε)) | O(b^d) |
| **Optimality** | ❌ | ⚠️ | ✅ | ✅ |
| **Completeness** | ⚠️ | ✅ | ✅ | ✅ |

Legend:
- b = branching factor (≤ 4 in our grid)
- m = maximum depth
- d = depth of shallowest solution
- C* = cost of optimal solution
- ε = minimum action cost

### Empirical Results

Average performance across all test scenarios:

```
Nodes Expanded (lower is better):
A*   ─────────■────── 125 nodes (baseline)
UCS  ─────────────────■ 171 nodes (+37%)
BFS  ──────────────────■ 176 nodes (+41%)
DFS  ─────────■ 152 nodes (+22% but suboptimal path)

Path Quality (cost):
A*/UCS ─■ 34 (optimal)
BFS    ────■ 45 (+32% worse)
DFS    ────────────■ 163 (+380% worse)
```

### When Each Algorithm Excels

**Use DFS when:**
- Memory is extremely limited
- Solutions are deep
- Any solution is acceptable

**Use BFS when:**
- All actions have equal cost
- Solutions are shallow
- Need shortest path (in steps)

**Use UCS when:**
- Actions have varying costs
- No good heuristic available
- Need guaranteed optimal cost

**Use A* when:**
- Good heuristic exists
- Need optimal solution
- Efficiency matters

## Implementation Notes

### Tie-Breaking

When multiple nodes have the same priority, we use:

1. **For UCS/A***: Use a counter to maintain FIFO order
```python
counter = 0
heapq.heappush(queue, (priority, counter, node))
counter += 1
```

2. **For equal f-values in A***: Prefer higher g (closer to goal)
```python
heapq.heappush(queue, (f, -g, counter, node))
```

### Visited Set vs. Closed List

We use a visited set to track expanded nodes:

```python
if current in visited:
    continue
visited.add(current)
```

This prevents redundant expansions and infinite loops.

### Path Reconstruction

Paths are built incrementally:

```python
# Add current node to path
new_path = path + [neighbor]
```

Alternative approach (more efficient for large paths):
```python
# Store parent pointers
came_from[neighbor] = current

# Reconstruct at the end
path = []
current = goal
while current in came_from:
    path.append(current)
    current = came_from[current]
path.reverse()
```

### Priority Queue Implementation

We use Python's `heapq` (min-heap):

```python
import heapq

# Push: O(log n)
heapq.heappush(queue, (priority, data))

# Pop: O(log n)
priority, data = heapq.heappop(queue)
```

## References

1. **Russell, S., & Norvig, P.** (2020). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson.

2. **Hart, P. E., Nilsson, N. J., & Raphael, B.** (1968). A Formal Basis for the Heuristic Determination of Minimum Cost Paths. *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 100-107.

3. **Dijkstra, E. W.** (1959). A note on two problems in connexion with graphs. *Numerische Mathematik*, 1(1), 269-271.

4. **Cormen, T. H., Leiserson, C. E., Rivest, R. L., & Stein, C.** (2009). *Introduction to Algorithms* (3rd ed.). MIT Press.

---

*Last updated: February 2026*
