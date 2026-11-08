# Q2a - A* Search untuk navigasi rute (heuristik jarak garis lurus)
import heapq, math

coords = {"Kampus": (0, 0), "A": (2, 1), "B": (4, 0), "C": (3, 3),
          "D": (6, 2), "Mall": (8, 1)}
graph = {                     # jarak jalan nyata (km)
    "Kampus": {"A": 2.4, "B": 4.5},
    "A": {"Kampus": 2.4, "C": 2.4, "D": 4.6},
    "B": {"Kampus": 4.5, "D": 2.8},
    "C": {"A": 2.4, "D": 3.4},
    "D": {"A": 4.6, "B": 2.8, "C": 3.4, "Mall": 2.3},
    "Mall": {"D": 2.3},
}

def h(n, goal):               # heuristik admissible: jarak Euclidean
    (x1, y1), (x2, y2) = coords[n], coords[goal]
    return math.hypot(x1 - x2, y1 - y2)

def a_star(start, goal):
    open_set = [(h(start, goal), 0, start, [start])]   # (f, g, node, path)
    best_g = {start: 0}
    while open_set:
        f, g, node, path = heapq.heappop(open_set)
        if node == goal:
            return path, g
        for nxt, cost in graph[node].items():
            g2 = g + cost
            if g2 < best_g.get(nxt, float("inf")):
                best_g[nxt] = g2
                heapq.heappush(open_set, (g2 + h(nxt, goal), g2, nxt, path + [nxt]))
    return None, float("inf")

path, dist = a_star("Kampus", "Mall")
print("Rute  :", " -> ".join(path))
print(f"Jarak : {dist:.1f} km")
