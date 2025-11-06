import heapq


graph = {
    'A': [('B', 1), ('C', 3)],
    'B': [('D', 1), ('E', 4)],
    'C': [('F', 2)],
    'D': [],
    'E': [],
    'F': []
}


heuristic = {
    'A': 5,
    'B': 3,
    'C': 2,
    'D': 6,
    'E': 4,
    'F': 0  # Goal node
}

def best_first_search(start, goal):
    visited = set()
    pq = [(heuristic[start], start)]  

    while pq:
        h, node = heapq.heappop(pq)
        if node in visited:
            continue

        print("Visiting:", node)
        visited.add(node)

        if node == goal:
            print("Goal found:", node)
            return

        for neighbor, cost in graph[node]:
            if neighbor not in visited:
                heapq.heappush(pq, (heuristic[neighbor], neighbor))



best_first_search('A', 'F')
