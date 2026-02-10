# Intelligent Pathfinding Agent (AI Search Project)

An intelligent agent that navigates a hazardous grid-based environment using
classical Artificial Intelligence search algorithms. The agent computes optimal
paths while accounting for obstacles and variable traversal costs such as fire hazards.

This project demonstrates core AI concepts including state-space modeling,
heuristic search, and algorithmic performance analysis.

---

## 🚀 Features

- Grid-based environment with obstacles and weighted terrain
- Implementations of:
  - Depth-First Search (DFS)
  - Breadth-First Search (BFS)
  - Uniform Cost Search (UCS / Dijkstra)
  - A* Search with admissible Manhattan heuristic
- Guaranteed optimal solutions for weighted environments (UCS, A*)
- Step-by-step visual animations of agent traversal
- Empirical performance comparison (path cost, nodes expanded)

---

## 🧠 Problem Overview

- **State Space:** Grid cells `(row, col)`
- **Actions:** Move Up, Down, Left, Right
- **Terrain Costs:**
  - Open ground (`.`): cost 1
  - Small fire (`f`): cost 3
  - Large fire (`F`): cost 5
  - Obstacle (`#`): impassable
- **Objective:** Reach the goal while minimizing total traversal cost

---

## 🔬 Algorithms Implemented

| Algorithm | Optimal | Cost-Aware | Notes |
|--------|--------|------------|------|
| DFS | ❌ | ❌ | Explores deeply, may be suboptimal |
| BFS | ✔ (steps) | ❌ | Shortest path in steps only |
| UCS | ✔ | ✔ | Guaranteed minimum-cost path |
| A* | ✔ | ✔ | Heuristic-guided, most efficient |

---

## 📊 Performance Insights

- BFS may select shorter but higher-cost paths through fire
- UCS guarantees optimal cost but expands many nodes
- A* reduces node expansions significantly while preserving optimality
- DFS is unpredictable but useful for baseline comparison

---

## 🎥 Visualizations

Animated path traversal for each algorithm is available in the `visuals/` folder:

- `bfs.gif`
- `dfs.gif`
- `ucs.gif`
- `astar.gif`

These clearly illustrate behavioral differences between uninformed and informed search.

---

## 🛠️ Tech Stack

- Python
- Data Structures (Heaps, Queues, Stacks)
- Matplotlib (Visualization)
- Algorithmic Analysis

---

## 📂 How to Run

```bash
git clone https://github.com/your-username/intelligent-pathfinding-agent.git
cd intelligent-pathfinding-agent
python pathfinder_search.py
