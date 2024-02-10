import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

def dfs_recursive(graph, vertex, visited=None):
    if visited is None:
        visited = set()
    visited.add(vertex)
    print(vertex, end=' ')  # Відвідуємо вершину
    for neighbor in graph[vertex]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

def bfs_recursive(graph, queue, visited=None):
    # Перевіряємо, чи існує множина відвіданих вершин, якщо ні, то ініціалізуємо нову
    if visited is None:
        visited = set()
    # Якщо черга порожня, завершуємо рекурсію
    if not queue:
        return
    # Вилучаємо вершину з початку черги
    vertex = queue.popleft()
    # Перевіряємо, чи відвідували раніше дану вершину
    if vertex not in visited:
        # Якщо не відвідували, друкуємо вершину
        print(vertex, end=" ")
        # Додаємо вершину до множини відвіданих вершин.
        visited.add(vertex)
        # Додаємо невідвіданих сусідів даної вершини в кінець черги.
        queue.extend(set(graph[vertex]) - visited)
    # Рекурсивний виклик для обробки наступних вершин черги
    bfs_recursive(graph, queue, visited)

def dijkstra(graph, start):
    # Ініціалізація відстаней та множини невідвіданих вершин
    distances = {vertex: float('infinity') for vertex in graph}
    distances[start] = 0
    unvisited = list(graph.keys())

    while unvisited:
        # Знаходження вершини з найменшою відстанню серед невідвіданих
        current_vertex = min(unvisited, key=lambda vertex: distances[vertex])

        # Якщо поточна відстань є нескінченністю, то ми завершили роботу
        if distances[current_vertex] == float('infinity'):
            break

        for neighbor, weight in graph[current_vertex].items():
            distance = distances[current_vertex] + weight

            # Якщо нова відстань коротша, то оновлюємо найкоротший шлях
            if distance < distances[neighbor]:
                distances[neighbor] = distance

        # Видаляємо поточну вершину з множини невідвіданих
        unvisited.remove(current_vertex)

    return distances

# Створюємо вершину
G = nx.Graph()

G.add_nodes_from(["Куліндор", "Теплодар", "Ринок 7км", "Пос.Котовського", 
                 "Авангард", "Кілія", "В.Дальник", 
                 "Черешуки", "Дерібасівська", "Слободка",
                 "Ізмаїл", "Ринок Анжеліка", 
                 "Торгова", "Ринок Староконий", "Склад", "Пересипський міст", 
                 "7ма пересипська"])
G.add_edges_from([("Куліндор", "Пос.Котовського",{'weight': 6}), 
                  ("Теплодар", "Ринок 7км", {'weight': 45}), 
                  ("Ринок 7км", "Авангард", {'weight': 10}), 
                  ("Ринок 7км", "Кілія", {'weight': 150}),
                  ("Ринок 7км", "В.Дальник", {'weight': 20}), 
                  ("Ринок 7км", "Черешуки", {'weight': 5}),
                  ("Дерібасівська", "Слободка", {'weight': 4}),
                  ("Кілія", "Ізмаїл", {'weight': 25}), 
                  ("Авангард", "Ринок Анжеліка", {'weight': 3}), 
                  ("Дерібасівська", "Торгова", {'weight': 1}), 
                  ("Дерібасівська", "Ринок Староконий", {'weight': 2}),
                  ("Черешуки","Склад", {'weight': 8}), 
                  ("Слободка","Склад", {'weight': 3}),
                  ("Ринок Староконий","Склад", {'weight': 2}),
                  ("Ізмаїл","Склад", {'weight': 160}), 
                  ("Ринок Анжеліка","Склад",{'weight': 18}), 
                  ("В.Дальник", "Склад", {'weight': 27}), 
                  ("Склад", "Пересипський міст", {'weight': 1}),
                  ("Торгова", "Пересипський міст", {'weight': 2}),
                  ("Пересипський міст", "7ма пересипська", {'weight': 5}),
                  ("Пос.Котовського", "7ма пересипська", {'weight': 8})])

DG = nx.DiGraph(G)

DG.remove_edges_from([("Ринок Анжеліка", "Склад"), ("Авангард", "Ринок Анжеліка"), ("Ринок 7км", "Авангард")])

# Застосування алгоритму PageRank для визначення ступыню вершини
pagerank = nx.pagerank(DG, alpha=0.85)  # alpha - це фактор згладжування
# Створення кортеджу результатів PageRank
pagerank_list = [(node, rank) for node, rank in pagerank.items()]
# Сортування кортеджу результатів PageRank від великого до найменьшого
pagerank_list_sortad = sorted(pagerank_list, key=lambda x: x[1], reverse=True)
# Виведення результатів відсортованого кортеджу PageRank
print("Рейтинг вершин за їх ступенню: ")
for node, rank in pagerank_list_sortad:
    print(f'{node}: {round(rank,4)}')
print(" ")


print("Amount of nodes: ", DG.number_of_nodes())
print("Amount of edges: ", DG.number_of_edges())
print(" ")

print("The best way: ", nx.shortest_path(DG,"Куліндор", "Ринок Староконий"))
print(" ")
print("centralness: ")
list_degree_centrality = nx.degree_centrality(DG)
sorted_list_degree_centrality = sorted(list_degree_centrality.items(), key=lambda x: x[1], reverse=True)
for node, rank in sorted_list_degree_centrality:
    print(f'{node}: {round(rank,4)}')
print(" ")  
print("Betweenness: ")
list_betweenness = nx.betweenness_centrality(DG)
sorted_list_betweenness = sorted(list_betweenness.items(), key=lambda x: x[1], reverse=True)
for node, rank in sorted_list_betweenness:
    print(f'{node}: {round(rank,4)}')
print(" ")
print("Closenness: ")
list_closenness = nx.closeness_centrality(DG)
sorted_list_closenness = sorted(list_closenness.items(), key=lambda x: x[1], reverse=True)
for node, rank in sorted_list_closenness:
    print(f'{node}: {round(rank,4)}')
print(" ")
dfs_tree = dfs_recursive(DG, "Склад")
print("DFS: ", dfs_tree)
print(" ")
initial_queue = deque(["Склад"])
bfs_tree = bfs_recursive(DG, initial_queue)
print("BFS: ", bfs_tree)
print(" ")
edge_labels = nx.get_edge_attributes(DG, 'weight')
print(" ")
# Застосування алгоритму Дейкстри
shortest_paths = nx.single_source_dijkstra_path(DG, source='Склад')
shortest_path_lengths = nx.single_source_dijkstra_path_length(DG, source='Склад')

print(shortest_paths)  # виведе найкоротші шляхи від вузла 'A' до всіх інших вузлів
print(shortest_path_lengths)  # виведе довжини найкоротших шляхів від вузла 'A' до всіх інших вузлів



plt.figure(figsize=(6,6))
pos = nx.spring_layout(DG, seed=42)
nx.draw_networkx(DG, pos, with_labels=True, node_size = 400, font_weight="bold", font_size=8)
nx.draw_networkx_edge_labels(DG, pos, edge_labels=edge_labels, font_color='red')
plt.show()