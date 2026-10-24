# Intelligent Library Search and Book Collection Agent

> AI-based indoor library search and multi-book route planning using graph search.

## Overview

The Intelligent Library Search and Book Collection Agent models a library as a weighted graph and uses **BFS, DFS, and A*** to find book locations and efficient collection routes. The prototype also handles practical constraints such as closed sections, restricted shelves, unavailable books, crowded aisles, priorities, and user limits.

This is an academic/student prototype designed for demonstrating **AI state-space search, heuristic search, graph modelling, and route optimization**.

## Features

- 📚 Library graph with weighted walking paths
- 🔎 Book catalogue with locations and priorities
- 🌊 Breadth First Search (BFS)
- 🌲 Depth First Search (DFS)
- ⭐ A* search with `f(n) = g(n) + h(n)`
- 📋 Open-list and closed-list traces
- 🚧 Closed sections and restricted shelves
- 📕 Unavailable-book handling
- 👥 Crowded-aisle cost penalties
- 🧭 Multi-book route optimization
- ⏱️ User cost/time-limit handling
- 🗺️ Library route visualization
- 📊 Algorithm comparison dashboard
- 💻 Streamlit web interface

## Project Structure

```text
Intelligent_Library_Search_Agent/
├── app.py
├── requirements.txt
├── run_project.bat
├── run_project.txt
├── project_structure.txt
├── README.md
├── LICENSE
└── docs/
    ├── project_report.md
    ├── viva_questions.md
    └── presentation_content.md
```

## How the AI Works

### 1. Graph representation

- **Nodes:** library sections/shelves/book locations
- **Edges:** walkable connections
- **Weights:** walking/traversal cost

### 2. BFS

Explores the graph level by level using a queue. Useful for shortest paths measured in number of edges when edge costs are uniform.

### 3. DFS

Explores deeply before backtracking. It uses a stack and does not guarantee an optimal route.

### 4. A*

Uses:

```text
f(n) = g(n) + h(n)
```

where `g(n)` is the cost already travelled and `h(n)` is the estimated remaining cost. The prototype uses a Manhattan-distance heuristic based on the simulated 2D library coordinates.

## Constraints

The graph is rebuilt according to the selected conditions:

- **Closed section:** excluded from traversal.
- **Restricted shelf/area:** excluded from traversal.
- **Unavailable book:** flagged and skipped.
- **Crowded aisle:** traversal cost is increased.

## Run Locally

### Windows / Anaconda Prompt

Open a terminal in the project folder, then:

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the Local URL printed by Streamlit, normally:

```text
http://localhost:8501
```

### Windows shortcut

Double-click:

```text
run_project.bat
```

## Demo Scenario

Try:

- Algorithm: **A***
- Start: **Entrance**
- Books: **Artificial Intelligence**, **Python Programming**, **Data Structures**
- Add a crowded section or close a section
- Run the planner
- Inspect the route, path cost, Open List and Closed List

## Academic Scope

The library map, book catalogue, walking costs, and dynamic conditions are simulated for demonstration. No real library database, indoor positioning hardware, or external API is required.

## Future Scope

- Real library database integration
- RFID/barcode/QR-based book tracking
- Indoor positioning using Bluetooth beacons
- Real-time crowd sensing
- Voice navigation
- Mobile application
- ML-based congestion prediction

## License

MIT License. See `LICENSE`.
