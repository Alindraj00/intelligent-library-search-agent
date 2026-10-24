import heapq
import itertools
import math
from collections import deque
from html import escape

import streamlit as st

st.set_page_config(
    page_title="Intelligent Library Search Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# INTELLIGENT LIBRARY SEARCH AND BOOK COLLECTION AGENT
# Enhanced academic prototype based on the supplied 8-page deck.
# ============================================================

BASE_GRAPH = {
    "Entrance": {"Fiction Aisle": 4, "Science Section": 6, "Reference Area": 7},
    "Fiction Aisle": {"Entrance": 4, "History Section": 5, "Children Books": 4},
    "Science Section": {"Entrance": 6, "History Section": 4, "Reference Area": 5},
    "History Section": {"Fiction Aisle": 5, "Science Section": 4, "Upper Level": 6},
    "Children Books": {"Fiction Aisle": 4, "Reference Area": 5},
    "Reference Area": {"Entrance": 7, "Science Section": 5, "Children Books": 5, "Upper Level": 4},
    "Upper Level": {"History Section": 6, "Reference Area": 4},
}

BOOKS = {
    "Artificial Intelligence": {"location": "Science Section", "priority": 5},
    "Computer Networks": {"location": "Science Section", "priority": 4},
    "Python Programming": {"location": "Reference Area", "priority": 5},
    "Java Programming": {"location": "Reference Area", "priority": 3},
    "World History": {"location": "History Section", "priority": 3},
    "Data Structures": {"location": "Upper Level", "priority": 4},
    "English Literature": {"location": "Fiction Aisle", "priority": 2},
    "Children Stories": {"location": "Children Books", "priority": 1},
}

COORDS = {
    "Entrance": (0, 0),
    "Fiction Aisle": (4, 0),
    "Science Section": (0, 4),
    "History Section": (4, 4),
    "Children Books": (7, 0),
    "Reference Area": (4, 7),
    "Upper Level": (7, 7),
}

NODE_INFO = {
    "Entrance": "Main entrance / starting point",
    "Fiction Aisle": "Fiction and literature",
    "Science Section": "AI, networks and computing",
    "History Section": "History and humanities",
    "Children Books": "Children's collection",
    "Reference Area": "Programming and reference books",
    "Upper Level": "Advanced / upper-level collection",
}

# -----------------------------
# Search engine
# -----------------------------
def build_graph(closed_sections, crowded_sections, restricted_shelves):
    closed_sections = set(closed_sections)
    crowded_sections = set(crowded_sections)
    restricted_shelves = set(restricted_shelves)
    graph = {}
    for node, neighbours in BASE_GRAPH.items():
        if node in closed_sections or node in restricted_shelves:
            continue
        graph[node] = {}
        for dest, base_cost in neighbours.items():
            if dest in closed_sections or dest in restricted_shelves:
                continue
            multiplier = 2 if node in crowded_sections or dest in crowded_sections else 1
            graph[node][dest] = base_cost * multiplier
    return graph


def heuristic(a, b):
    x1, y1 = COORDS[a]
    x2, y2 = COORDS[b]
    return abs(x1 - x2) + abs(y1 - y2)


def reconstruct(parent, goal):
    path = []
    current = goal
    while current is not None:
        path.append(current)
        current = parent.get(current)
    return list(reversed(path))


def bfs(graph, start, goal):
    if start not in graph or goal not in graph:
        return [], math.inf, [], []
    queue = deque([start])
    parent = {start: None}
    open_trace = [start]
    closed = []
    while queue:
        current = queue.popleft()
        if current in closed:
            continue
        closed.append(current)
        if current == goal:
            path = reconstruct(parent, goal)
            cost = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
            return path, cost, open_trace, closed
        for neighbour in graph.get(current, {}):
            if neighbour not in parent:
                parent[neighbour] = current
                queue.append(neighbour)
                open_trace.append(neighbour)
    return [], math.inf, open_trace, closed


def dfs(graph, start, goal):
    if start not in graph or goal not in graph:
        return [], math.inf, [], []
    stack = [start]
    parent = {start: None}
    open_trace = [start]
    closed = []
    while stack:
        current = stack.pop()
        if current in closed:
            continue
        closed.append(current)
        if current == goal:
            path = reconstruct(parent, goal)
            cost = sum(graph[path[i]][path[i + 1]] for i in range(len(path) - 1))
            return path, cost, open_trace, closed
        for neighbour in reversed(list(graph.get(current, {}))):
            if neighbour not in parent:
                parent[neighbour] = current
                stack.append(neighbour)
                open_trace.append(neighbour)
    return [], math.inf, open_trace, closed


def astar(graph, start, goal):
    if start not in graph or goal not in graph:
        return [], math.inf, [], []
    counter = itertools.count()
    pq = [(heuristic(start, goal), next(counter), 0, start)]
    g_score = {start: 0}
    parent = {start: None}
    closed = []
    open_trace = []
    while pq:
        f_score, _, current_g, current = heapq.heappop(pq)
        open_trace.append({
            "node": current,
            "g(n)": current_g,
            "h(n)": heuristic(current, goal),
            "f(n)": f_score,
        })
        if current in closed:
            continue
        if current == goal:
            return reconstruct(parent, goal), g_score[goal], open_trace, closed
        closed.append(current)
        for neighbour, edge_cost in graph.get(current, {}).items():
            tentative = g_score[current] + edge_cost
            if tentative < g_score.get(neighbour, math.inf):
                g_score[neighbour] = tentative
                parent[neighbour] = current
                f = tentative + heuristic(neighbour, goal)
                heapq.heappush(pq, (f, next(counter), tentative, neighbour))
    return [], math.inf, open_trace, closed


def search(graph, start, goal, algorithm):
    if algorithm == "BFS":
        return bfs(graph, start, goal)
    if algorithm == "DFS":
        return dfs(graph, start, goal)
    return astar(graph, start, goal)


# -----------------------------
# Multi-book planner
# -----------------------------
def leg_matrix(graph, start, books, algorithm):
    targets = [start] + [BOOKS[b]["location"] for b in books]
    matrix = {}
    for a in targets:
        for b in targets:
            matrix[(a, b)] = ([], 0, [], []) if a == b else search(graph, a, b, algorithm)
    return matrix


def build_route_from_order(graph, start, ordered_books, algorithm, matrix=None):
    current = start
    full_path = [start]
    total_cost = 0
    legs = []
    for book in ordered_books:
        target = BOOKS[book]["location"]
        result = matrix[(current, target)] if matrix and (current, target) in matrix else search(graph, current, target, algorithm)
        path, cost, open_trace, closed = result
        if not path:
            return None
        full_path.extend(path[1:])
        total_cost += cost
        legs.append({
            "book": book,
            "location": target,
            "priority": BOOKS[book]["priority"],
            "cost": cost,
            "path": path,
            "open": open_trace,
            "closed": closed,
        })
        current = target
    return full_path, total_cost, legs


def optimize_books(graph, start, selected_books, algorithm, limit):
    feasible = [b for b in selected_books if BOOKS[b]["location"] in graph]
    unavailable = [b for b in selected_books if b not in feasible]
    if not feasible:
        return None, unavailable, "No requested books are currently reachable."

    matrix = leg_matrix(graph, start, feasible, algorithm)
    best = None

    if len(feasible) <= 7:
        for order in itertools.permutations(feasible):
            result = build_route_from_order(graph, start, order, algorithm, matrix)
            if result is None:
                continue
            route, cost, legs = result
            if cost <= limit:
                priority_score = sum(BOOKS[b]["priority"] for b in order)
                key = (cost, -priority_score)
                if best is None or key < best[0]:
                    best = (key, result)
        if best:
            return best[1], unavailable, "Exact minimum-cost ordering selected for the requested books."

        remaining = set(feasible)
        order, current, spent = [], start, 0
        while remaining:
            candidates = []
            for book in remaining:
                path, cost, _, _ = matrix[(current, BOOKS[book]["location"])]
                if path and spent + cost <= limit:
                    candidates.append((BOOKS[book]["priority"], cost, book))
            if not candidates:
                break
            _, cost, chosen = max(candidates, key=lambda x: (x[0], -x[1]))
            order.append(chosen)
            spent += cost
            current = BOOKS[chosen]["location"]
            remaining.remove(chosen)
        if order:
            result = build_route_from_order(graph, start, order, algorithm, matrix)
            skipped = [b for b in feasible if b not in order] + unavailable
            return result, skipped, "Priority-aware fallback used because the limit prevents full collection."
        return None, feasible + unavailable, "No feasible route fits the selected limit."

    # Scalable greedy planner for larger selections.
    remaining, order = set(feasible), []
    current, spent = start, 0
    while remaining:
        candidates = []
        for book in remaining:
            path, cost, _, _ = search(graph, current, BOOKS[book]["location"], algorithm)
            if path and spent + cost <= limit:
                candidates.append((BOOKS[book]["priority"], cost, book))
        if not candidates:
            break
        _, cost, chosen = max(candidates, key=lambda x: (x[0], -x[1]))
        order.append(chosen)
        spent += cost
        current = BOOKS[chosen]["location"]
        remaining.remove(chosen)
    if not order:
        return None, feasible + unavailable, "No feasible route fits the selected limit."
    result = build_route_from_order(graph, start, order, algorithm)
    return result, list(remaining) + unavailable, "Priority-aware heuristic used for a larger book set."


# -----------------------------
# Visuals
# -----------------------------
def edge_list(graph):
    edges = set()
    for a, neighbours in graph.items():
        for b, weight in neighbours.items():
            key = tuple(sorted((a, b)))
            edges.add((key[0], key[1], weight))
    return sorted(edges)


def draw_svg(graph, path=None, closed=None, restricted=None, crowded=None, animate=True):
    path = path or []
    closed, restricted, crowded = set(closed or []), set(restricted or []), set(crowded or [])
    width, height = 900, 510
    sx, sy, ox, oy = 100, 55, 70, 60

    def point(node):
        x, y = COORDS[node]
        return ox + x * sx, oy + (7 - y) * sy

    svg = [
        f'<svg width="100%" height="{height}" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">',
        '''<defs>
          <filter id="glow"><feGaussianBlur stdDeviation="5" result="b"/><feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
          <style>
            .pulse { animation: pulse 1.8s ease-in-out infinite; transform-origin:center; }
            .routeDash { stroke-dasharray: 12 10; animation: dash 2.5s linear infinite; }
            @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.45} }
            @keyframes dash { to { stroke-dashoffset:-44; } }
          </style>
        </defs>''',
        '<rect width="100%" height="100%" rx="26" fill="#07131d"/>',
        '<text x="32" y="32" fill="#ff7a18" font-size="13" font-family="Arial" font-weight="700" letter-spacing="2">LIVE LIBRARY FLOOR MAP</text>',
    ]

    for a, b, weight in edge_list(graph):
        x1, y1 = point(a); x2, y2 = point(b)
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#2b3e4a" stroke-width="4"/>')
        mx, my = (x1+x2)/2, (y1+y2)/2
        svg.append(f'<rect x="{mx-12}" y="{my-17}" width="24" height="18" rx="7" fill="#0c1c27"/>')
        svg.append(f'<text x="{mx}" y="{my-4}" fill="#80929d" font-size="10" text-anchor="middle">{weight}</text>')

    for a, b in zip(path, path[1:]):
        if a in COORDS and b in COORDS:
            x1, y1 = point(a); x2, y2 = point(b)
            cls = "routeDash" if animate else ""
            svg.append(f'<line class="{cls}" x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#ff7a18" stroke-width="9" stroke-linecap="round" filter="url(#glow)"/>')

    for node in COORDS:
        x, y = point(node)
        active = node in path
        is_closed = node in closed
        is_restricted = node in restricted
        is_crowded = node in crowded
        if is_closed:
            fill, stroke, label = "#481b20", "#ff4d5e", "CLOSED"
        elif is_restricted:
            fill, stroke, label = "#3b2b12", "#ffb347", "RESTRICTED"
        elif active:
            fill, stroke, label = "#ff7a18", "#ffd6ad", "ROUTE"
        elif is_crowded:
            fill, stroke, label = "#332916", "#dcae4b", "CROWDED"
        else:
            fill, stroke, label = "#132632", "#718894", ""
        cls = "pulse" if active else ""
        svg.append(f'<circle class="{cls}" cx="{x}" cy="{y}" r="18" fill="{fill}" stroke="{stroke}" stroke-width="3"/>')
        svg.append(f'<text x="{x+27}" y="{y+5}" fill="#f2f4f5" font-size="14" font-family="Arial">{escape(node)}</text>')
        if label:
            svg.append(f'<text x="{x}" y="{y+35}" fill="{stroke}" font-size="10" text-anchor="middle" font-family="Arial" font-weight="700">{label}</text>')
    svg.append('</svg>')
    return ''.join(svg)


def metric_card(label, value, note=""):
    return f'<div class="metric-card"><div class="metric-label">{escape(label)}</div><div class="metric-value">{escape(str(value))}</div><div class="metric-note">{escape(note)}</div></div>'


def inject_css():
    st.markdown("""
    <style>
      :root { --bg:#06121b; --panel:#0c1b25; --panel2:#101f2a; --line:#263b48; --muted:#9aadb8; --orange:#ff7a18; --orange2:#ff9a4a; }
      .stApp { background: radial-gradient(circle at 85% 10%, #10232f 0, var(--bg) 32%); color:#f3f5f6; }
      .block-container { max-width:1380px; padding-top:1.3rem; padding-bottom:3rem; }
      h1,h2,h3,h4 { font-family:Arial,sans-serif !important; letter-spacing:-.4px; }
      .hero { min-height:340px; border:1px solid var(--line); border-radius:28px; overflow:hidden; position:relative; background:linear-gradient(100deg,rgba(5,16,24,.98) 0%,rgba(5,16,24,.92) 43%,rgba(5,16,24,.55) 100%), radial-gradient(circle at 80% 35%,#1c3c4c 0,#07131d 55%); padding:46px; box-shadow:0 22px 60px rgba(0,0,0,.3); }
      .hero:after { content:'✦  ✦  ✦'; position:absolute; right:45px; top:38px; color:#ff7a18; opacity:.45; font-size:28px; letter-spacing:18px; }
      .eyebrow { color:var(--orange); font-size:12px; font-weight:800; letter-spacing:2.5px; text-transform:uppercase; }
      .hero-title { font-size:54px; line-height:1.02; margin:14px 0; font-weight:800; max-width:720px; }
      .hero-sub { color:#a9b8c0; font-size:17px; line-height:1.6; max-width:720px; }
      .accent { color:var(--orange); }
      .metric-card { background:linear-gradient(145deg,#10212c,#0b1922); border:1px solid #2b414e; border-radius:18px; padding:16px; min-height:108px; box-shadow:0 8px 25px rgba(0,0,0,.15); }
      .metric-label { color:#8fa3ae; font-size:11px; text-transform:uppercase; letter-spacing:1.4px; }
      .metric-value { color:var(--orange); font-size:28px; font-weight:800; margin-top:6px; }
      .metric-note { color:#718894; font-size:11px; }
      .panel { background:#0b1a24; border:1px solid var(--line); border-radius:18px; padding:20px; }
      .badge { display:inline-block; border:1px solid #34505e; background:#0c1c27; color:#b6c4cb; border-radius:999px; padding:5px 10px; margin:3px; font-size:12px; }
      .badge.orange { background:rgba(255,122,24,.12); color:#ff9a4a; border-color:rgba(255,122,24,.35); }
      .feature { background:linear-gradient(145deg,#0d1d28,#08151e); border:1px solid #263c49; border-radius:18px; padding:20px; height:150px; }
      .feature-icon { font-size:26px; }
      .feature-title { font-size:16px; font-weight:800; margin-top:10px; }
      .feature-text { color:#8fa1ab; font-size:12px; margin-top:5px; line-height:1.5; }
      .login-card { max-width:560px; margin:70px auto 0; background:linear-gradient(145deg,#0d1d28,#07131d); border:1px solid #29404d; border-radius:24px; padding:36px; box-shadow:0 30px 90px rgba(0,0,0,.35); }
      .section-kicker { color:var(--orange); font-size:11px; text-transform:uppercase; letter-spacing:2px; font-weight:800; }
      .big-number { font-size:40px; font-weight:800; color:var(--orange); }
      .stButton > button { border-radius:12px; border:1px solid #ff7a18; background:#ff7a18; color:#07131d; font-weight:800; }
      .stButton > button:hover { background:#ff913f; border-color:#ff913f; color:#07131d; }
      div[data-testid="stMetric"] { background:var(--panel2); border:1px solid #2b414e; padding:10px; border-radius:14px; }
      [data-testid="stSidebar"] { background:#06121b; border-right:1px solid #1d303b; }
      [data-testid="stSidebar"] h2 { color:#ff7a18; }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------
# Screens
# -----------------------------
def landing_screen():
    st.markdown("""
    <div class="hero">
      <div class="eyebrow">AI SEARCH • GRAPH PATHFINDING • ROUTE OPTIMIZATION</div>
      <div class="hero-title">Intelligent Library Search<br><span class="accent">and Book Collection Agent</span></div>
      <div class="hero-sub">A student-built AI system that finds books, understands library constraints, and generates an efficient multi-book collection route using BFS, DFS and A*.</div>
      <div style="margin-top:22px"><span class="badge orange">Graph Search</span><span class="badge">Constraint Aware</span><span class="badge">Multi-Book Planning</span><span class="badge">Interactive Map</span></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### What this agent does")
    cols = st.columns(4)
    cards = [
        ("🔎", "Find books", "Locate requested books from the library graph."),
        ("🧠", "Choose a search", "Compare BFS, DFS and heuristic A* search."),
        ("🚧", "Handle reality", "Avoid closed/restricted areas and penalize crowds."),
        ("🧭", "Plan the route", "Optimize the order of multiple book pickups."),
    ]
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(f'<div class="feature"><div class="feature-icon">{icon}</div><div class="feature-title">{title}</div><div class="feature-text">{desc}</div></div>', unsafe_allow_html=True)
    st.write("")
    a, b, c = st.columns([1, 1, 2])
    with a:
        if st.button("🚀 Enter Agent", use_container_width=True):
            st.session_state.screen = "agent"
            st.rerun()
    with b:
        if st.button("📚 Browse Books", use_container_width=True):
            st.session_state.screen = "books"
            st.rerun()
    with c:
        st.caption("Demo authentication is intentionally local-only; it is a project UI feature, not production security.")


def login_screen():
    st.markdown('<div class="login-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Student Demo Access</div>', unsafe_allow_html=True)
    st.markdown("## Sign in to Library Agent")
    st.caption("Use any name and demo PIN to enter. No credentials are stored.")
    name = st.text_input("Student name", placeholder="Enter your name")
    pin = st.text_input("Demo PIN", type="password", placeholder="1234")
    if st.button("Sign In", use_container_width=True):
        if name.strip() and pin.strip():
            st.session_state.user = name.strip()
            st.session_state.screen = "agent"
            st.rerun()
        st.error("Enter a name and PIN to continue.")
    if st.button("← Back", use_container_width=True):
        st.session_state.screen = "home"
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


def book_browser():
    st.markdown('<div class="section-kicker">Library Catalog</div>', unsafe_allow_html=True)
    st.title("Book Search")
    st.caption("Search the project database before building a collection route.")
    query = st.text_input("Search books", placeholder="Try: Python, Networks, History...")
    rows = []
    for title, info in BOOKS.items():
        if query.lower() in title.lower() or query.lower() in info["location"].lower():
            rows.append({"Book": title, "Section": info["location"], "Priority": info["priority"], "Status": "Available"})
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if st.button("🧭 Open Route Planner"):
        st.session_state.screen = "agent"
        st.rerun()


def agent_screen():
    user = st.session_state.get("user", "Student")
    with st.sidebar:
        st.markdown("## 📚 Library Agent")
        st.caption(f"Signed in as **{user}**")
        screen = st.radio("Workspace", ["Route Planner", "Search Trace", "Library Map", "Algorithm Comparison", "Book Database"], index=0)
        st.divider()
        st.markdown("### ⚙️ Agent Configuration")
        algorithm = st.selectbox("Search algorithm", ["A*", "BFS", "DFS"], index=0)
        start = st.selectbox("Starting location", list(BASE_GRAPH), index=0)
        selected_books = st.multiselect("Books to collect", list(BOOKS), default=["Artificial Intelligence", "Python Programming", "Data Structures"])
        limit = st.number_input("Maximum walking cost / time units", min_value=1, max_value=500, value=35, step=1)
        st.divider()
        st.markdown("### 🚧 Dynamic Constraints")
        closed = st.multiselect("Closed sections", list(BASE_GRAPH), default=[])
        crowded = st.multiselect("Crowded aisles", list(BASE_GRAPH), default=["History Section"])
        restricted = st.multiselect("Restricted shelves", list(BASE_GRAPH), default=[])
        st.caption("Crowded areas double traversal cost. Closed and restricted locations are removed from traversal.")
        if st.button("← Home", use_container_width=True):
            st.session_state.screen = "home"
            st.rerun()

    blocked = set(closed) | set(restricted)
    graph = build_graph(closed, crowded, restricted)
    available = {book: BOOKS[book]["location"] not in blocked for book in BOOKS}
    if start not in graph:
        st.error("The selected starting location is blocked. Choose an accessible starting location.")
        return
    if not selected_books:
        st.warning("Select at least one book from the sidebar to run the agent.")
        return

    result, skipped, planner_note = optimize_books(graph, start, selected_books, algorithm, limit)
    if result:
        route, total_cost, legs = result
        status = "Search Complete"
    else:
        route, total_cost, legs = [], math.inf, []
        status = "No Feasible Route"

    st.markdown(f"<div class='section-kicker'>LIVE AGENT • {escape(status.upper())}</div>", unsafe_allow_html=True)
    st.title("Intelligent Library Search Agent")
    st.caption("Find → reason → avoid constraints → optimize → guide")

    cols = st.columns(5)
    values = [
        ("Requested", len(selected_books), "books"),
        ("Collected", len(legs), "planned stops"),
        ("Path Cost", "—" if math.isinf(total_cost) else f"{total_cost:g}", "walking units"),
        ("Algorithm", algorithm, "selected search"),
        ("Status", status, "agent result"),
    ]
    for col, (label, value, note) in zip(cols, values):
        with col:
            st.markdown(metric_card(label, value, note), unsafe_allow_html=True)

    if skipped:
        st.warning("Skipped / unavailable: " + ", ".join(skipped))
    st.info("Planner decision: " + planner_note)

    if screen == "Route Planner":
        if result:
            left, right = st.columns([1.1, 2.1])
            with left:
                st.markdown("### Collection sequence")
                for i, leg in enumerate(legs, 1):
                    st.markdown(f'<div class="panel" style="margin-bottom:8px"><span class="section-kicker">STOP {i}</span><br><b>{escape(leg["book"])}</b><br><span class="muted">{escape(leg["location"])} • priority {leg["priority"]} • cost {leg["cost"]}</span></div>', unsafe_allow_html=True)
            with right:
                st.markdown("### 🗺️ Animated collection map")
                st.components.v1.html(draw_svg(graph, route, closed, restricted, crowded, animate=True), height=540, scrolling=False)
            st.markdown("### Ordered book collection plan")
            rows = []
            for index, leg in enumerate(legs, 1):
                rows.append({"Stop": index, "Book": leg["book"], "Shelf / Section": leg["location"], "Priority": leg["priority"], "Leg Cost": leg["cost"], "Navigation": " → ".join(leg["path"])})
            st.dataframe(rows, use_container_width=True, hide_index=True)
        else:
            st.error("The agent could not create a feasible route under the current constraints and limit.")

    elif screen == "Search Trace":
        st.subheader("Open List and Closed List")
        if legs:
            leg_index = st.selectbox("Inspect search leg", range(len(legs)), format_func=lambda i: f"{i+1}. {legs[i]['book']} → {legs[i]['location']}")
            leg = legs[leg_index]
            left, right = st.columns(2)
            with left:
                st.markdown("#### Open List")
                if algorithm == "A*":
                    st.dataframe(leg["open"], use_container_width=True, hide_index=True)
                else:
                    st.write(leg["open"])
            with right:
                st.markdown("#### Closed List")
                st.write(leg["closed"])
            st.markdown("#### Selected path")
            st.code(" → ".join(leg["path"]))
            st.caption("A*: f(n) = g(n) + h(n), with Manhattan distance used as h(n).")
        else:
            st.info("No search trace is available.")

    elif screen == "Library Map":
        st.subheader("Library Graph Representation")
        st.write("Locations are states, walking paths are actions, and edge weights represent walking cost. Closed/restricted locations are excluded; crowded areas increase cost.")
        st.components.v1.html(draw_svg(graph, route, closed, restricted, crowded, animate=True), height=540, scrolling=False)
        with st.expander("View graph adjacency data"):
            st.json(graph)

    elif screen == "Algorithm Comparison":
        st.subheader("BFS vs DFS vs A*")
        targets = list(dict.fromkeys(BOOKS[b]["location"] for b in selected_books if BOOKS[b]["location"] in graph))
        comparison = []
        for target in targets:
            for algo in ["BFS", "DFS", "A*"]:
                path, cost, open_trace, closed_trace = search(graph, start, target, algo)
                comparison.append({"Target": target, "Algorithm": algo, "Found": bool(path), "Path Cost": None if math.isinf(cost) else cost, "Nodes Closed": len(closed_trace), "Nodes Generated": len(open_trace), "Path": " → ".join(path) if path else "No path"})
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        a, b, c = st.columns(3)
        with a: st.markdown('<div class="feature"><div class="feature-title">BFS</div><div class="feature-text">Level-by-level exploration. Best for equal-cost step search.</div></div>', unsafe_allow_html=True)
        with b: st.markdown('<div class="feature"><div class="feature-title">DFS</div><div class="feature-text">Depth-first exploration with backtracking. Not guaranteed optimal.</div></div>', unsafe_allow_html=True)
        with c: st.markdown('<div class="feature"><div class="feature-title">A*</div><div class="feature-text">Uses g(n)+h(n) to guide search toward efficient routes.</div></div>', unsafe_allow_html=True)

    else:
        st.subheader("Library Book Database")
        query = st.text_input("Quick search", placeholder="Search title or section")
        rows = []
        for title, info in BOOKS.items():
            if not query or query.lower() in title.lower() or query.lower() in info["location"].lower():
                loc = info["location"]
                rows.append({"Book": title, "Location": loc, "Priority": info["priority"], "Availability": "Available" if available[title] else "Blocked", "Aisle Status": "Crowded" if loc in crowded else "Normal"})
        st.dataframe(rows, use_container_width=True, hide_index=True)

    st.divider()
    st.caption("Academic prototype • Graph-based modeling • BFS / DFS / A* • Dynamic constraints • Multi-book route optimization")


# -----------------------------
# App shell
# -----------------------------
inject_css()
if "screen" not in st.session_state:
    st.session_state.screen = "home"
if "user" not in st.session_state:
    st.session_state.user = "Student"

# Top navigation
if st.session_state.screen in {"home", "login", "books"}:
    nav1, nav2, nav3 = st.columns([1, 1, 1])
    with nav1:
        if st.button("📚 Library Agent", use_container_width=True):
            st.session_state.screen = "home"
            st.rerun()
    with nav2:
        if st.button("🔎 Book Search", use_container_width=True):
            st.session_state.screen = "books"
            st.rerun()
    with nav3:
        if st.button("🚀 Sign In", use_container_width=True):
            st.session_state.screen = "login"
            st.rerun()

if st.session_state.screen == "home":
    landing_screen()
elif st.session_state.screen == "login":
    login_screen()
elif st.session_state.screen == "books":
    book_browser()
else:
    agent_screen()
