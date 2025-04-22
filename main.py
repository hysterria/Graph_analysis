import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import networkx as nx
import time
import math
import random


class GraphApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Задача коммивояжера")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')

        # Стили
        button_style = {
            'font': ('Arial', 10),
            'bg': '#e1e1e1',
            'activebackground': '#d0d0d0',
            'width': 12,
            'height': 1
        }
        label_style = {'font': ('Arial', 10), 'bg': '#f0f0f0'}
        entry_style = {'font': ('Arial', 10), 'width': 8, 'relief': tk.SUNKEN, 'borderwidth': 1}

        # Основной фрейм
        self.main_frame = tk.Frame(root, bg='#f0f0f0')
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Фрейм для графиков
        self.graph_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        self.graph_frame.pack(fill=tk.BOTH, expand=True)

        # Canvas для исходного графа
        self.canvas = tk.Canvas(self.graph_frame, width=600, height=400, bg="white", highlightthickness=1,
                                highlightbackground="#999")
        self.canvas.grid(row=0, column=0, padx=(0, 10), pady=5, sticky='nsew')

        # Canvas для результата
        self.result_canvas = tk.Canvas(self.graph_frame, width=600, height=400, bg="white", highlightthickness=1,
                                       highlightbackground="#999")
        self.result_canvas.grid(row=0, column=1, pady=5, sticky='nsew')

        # Нижний фрейм (таблица + управление)
        self.bottom_frame = tk.Frame(self.main_frame, bg='#f0f0f0')
        self.bottom_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        # Таблица
        self.table_container = tk.Frame(self.bottom_frame, bg='#f0f0f0', height=100)
        self.table_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.tree = ttk.Treeview(
            self.table_container,
            columns=("From", "To", "Weight"),
            show="headings",
            style="Custom.Treeview",
            height=5
        )
        self.tree.heading("From", text="Из")
        self.tree.heading("To", text="В")
        self.tree.heading("Weight", text="Вес")
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scroll_y = ttk.Scrollbar(self.table_container, orient=tk.VERTICAL, command=self.tree.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.configure(yscrollcommand=scroll_y.set)

        scroll_x = ttk.Scrollbar(self.table_container, orient=tk.HORIZONTAL, command=self.tree.xview)
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)
        self.tree.configure(xscrollcommand=scroll_x.set)

        # Фрейм управления (правая часть)
        self.control_frame = tk.Frame(self.bottom_frame, bg='#f0f0f0', width=350)
        self.control_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=(10, 0))

        # Алгоритмы
        algo_frame = tk.Frame(self.control_frame, bg='#f0f0f0')
        algo_frame.grid(row=0, column=0, padx=5, pady=5, sticky='nw')

        tk.Label(algo_frame, text="Алгоритмы", **label_style).pack(anchor='w')

        self.calc_button = tk.Button(algo_frame, text="NN", command=self.run_nn, **button_style)
        self.calc_button.pack(fill=tk.X, pady=2)

        self.rnn_button = tk.Button(algo_frame, text="RNN", command=self.run_rnn, **button_style)
        self.rnn_button.pack(fill=tk.X, pady=2)

        self.annealing_button = tk.Button(algo_frame, text="Отжиг", command=self.run_annealing, **button_style)
        self.annealing_button.pack(fill=tk.X, pady=2)

        self.boltzmann_button = tk.Button(algo_frame, text="Больцман", command=self.run_boltzmann_annealing,
                                          **button_style)
        self.boltzmann_button.pack(fill=tk.X, pady=2)

        # Управление
        control_frame = tk.Frame(self.control_frame, bg='#f0f0f0')
        control_frame.grid(row=0, column=1, padx=5, pady=5, sticky='nw')

        tk.Label(control_frame, text="Управление", **label_style).pack(anchor='w')

        self.clear_button = tk.Button(control_frame, text="Очистить", command=self.clear_graph, **button_style)
        self.clear_button.pack(fill=tk.X, pady=2)

        self.cencelator_button = tk.Button(control_frame, text="Отменить", command=self.cencelator, **button_style)
        self.cencelator_button.pack(fill=tk.X, pady=2)

        # Параметры + создание полного графа
        params_frame = tk.Frame(self.control_frame, bg='#f0f0f0')
        params_frame.grid(row=0, column=2, padx=5, pady=5, sticky='nw')

        tk.Label(params_frame, text="Параметры", **label_style).pack(anchor='w')

        tk.Label(params_frame, text="T0:", **label_style).pack(anchor='w')
        self.initial_temp_entry = tk.Entry(params_frame, **entry_style)
        self.initial_temp_entry.insert(0, "1000")
        self.initial_temp_entry.pack(fill=tk.X, pady=2)

        tk.Label(params_frame, text="Охлаждение:", **label_style).pack(anchor='w')
        self.cooling_rate_entry = tk.Entry(params_frame, **entry_style)
        self.cooling_rate_entry.insert(0, "0.995")
        self.cooling_rate_entry.pack(fill=tk.X, pady=2)

        tk.Label(params_frame, text="Итерации:", **label_style).pack(anchor='w')
        self.iterations_entry = tk.Entry(params_frame, **entry_style)
        self.iterations_entry.insert(0, "1000")
        self.iterations_entry.pack(fill=tk.X, pady=2)

        tk.Label(params_frame, text="T0 (Больцман):", **label_style).pack(anchor='w')
        self.boltzmann_temp_entry = tk.Entry(params_frame, **entry_style)
        self.boltzmann_temp_entry.insert(0, "1000")
        self.boltzmann_temp_entry.pack(fill=tk.X, pady=2)

        tk.Label(params_frame, text="Макс. итер:", **label_style).pack(anchor='w')
        self.boltzmann_iter_entry = tk.Entry(params_frame, **entry_style)
        self.boltzmann_iter_entry.insert(0, "10000")
        self.boltzmann_iter_entry.pack(fill=tk.X, pady=2)

        # Кнопка создания полного графа
        self.complete_graph_frame = tk.Frame(params_frame, bg='#f0f0f0')
        self.complete_graph_frame.pack(fill=tk.X, pady=(10, 0))

        tk.Label(self.complete_graph_frame, text="Вершин:", **label_style).pack(side=tk.LEFT)

        # Создаем копию стиля без width, чтобы избежать дублирования
        entry_style_no_width = entry_style.copy()
        if 'width' in entry_style_no_width:
            entry_style_no_width.pop('width')

        self.vertex_count_entry = tk.Entry(
            self.complete_graph_frame,
            width=5,  # Явно задаем ширину здесь
            **entry_style_no_width  # Все остальные параметры стиля
        )
        self.vertex_count_entry.pack(side=tk.LEFT, padx=5)

        self.create_complete_button = tk.Button(
            self.complete_graph_frame,
            text="Создать граф",
            command=self.create_complete_graph,
            **button_style
        )
        self.create_complete_button.pack(side=tk.LEFT)

        # Вывод результатов (длина пути и время)
        self.info_frame = tk.Frame(self.main_frame, bg='#f0f0f0', height=40)
        self.info_frame.pack(fill=tk.X, pady=(5, 0))

        self.distance_label = tk.Label(
            self.info_frame,
            text="Длина пути: -",
            font=('Arial', 12),
            bg='#f0f0f0'
        )
        self.distance_label.pack(side=tk.LEFT, padx=20)

        self.time_label = tk.Label(
            self.info_frame,
            text="Время: -",
            font=('Arial', 12),
            bg='#f0f0f0'
        )
        self.time_label.pack(side=tk.LEFT, padx=20)

        # Инициализация графа
        self.graph = nx.DiGraph()
        self.nodes = []
        self.selected_node = None
        self.history = []

        # Настройка сетки
        self.graph_frame.columnconfigure(0, weight=1)
        self.graph_frame.columnconfigure(1, weight=1)
        self.graph_frame.rowconfigure(0, weight=1)

        # Привязка событий
        self.canvas.bind("<Button-1>", self.handle_click)

        # Стили для таблицы
        style = ttk.Style()
        style.configure("Custom.Treeview", font=('Arial', 9), rowheight=20)
        style.configure("Custom.Treeview.Heading", font=('Arial', 9, 'bold'))

    def create_complete_graph(self):
        try:
            n = int(self.vertex_count_entry.get())
            if n < 2:
                messagebox.showerror("Ошибка", "Количество вершин должно быть не менее 2")
                return

        except ValueError:
            messagebox.showerror("Ошибка", "Введите целое число")
            return

        self.clear_graph()

        # Получаем размеры canvas
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if canvas_width < 10 or canvas_height < 10:
            canvas_width, canvas_height = 650, 450

        center_x, center_y = canvas_width // 2, canvas_height // 2
        radius = min(canvas_width, canvas_height) * 0.4

        self.nodes = []

        for i in range(n):
            angle = 2 * math.pi * i / n
            x = center_x + radius * math.cos(angle)
            y = center_y + radius * math.sin(angle)
            self.nodes.append((x, y))
            self.graph.add_node(i, pos=(x, y))

        for i in range(n):
            for j in range(n):
                if i != j:
                    x1, y1 = self.nodes[i]
                    x2, y2 = self.nodes[j]
                    weight = random.randint(1,100)
                    self.graph.add_edge(i, j, weight=round(weight, 2))
                    self.update_table(i, j, round(weight, 2))

        self.draw_graph()
        messagebox.showinfo("Успех", f"Создан полный граф с {n} вершинами")

    def get_annealing_params(self):
        try:
            return {
                'initial_temp': float(self.initial_temp_entry.get()),
                'cooling_rate': float(self.cooling_rate_entry.get()),
                'iterations': int(self.iterations_entry.get())
            }
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры отжига")
            return None

    def get_boltzmann_params(self):
        try:
            return {
                'initial_temp': float(self.boltzmann_temp_entry.get()),
                'max_iterations': int(self.boltzmann_iter_entry.get())
            }
        except ValueError:
            messagebox.showerror("Ошибка", "Некорректные параметры Больцмановского отжига")
            return None

    def handle_click(self, event):
        for node, (x, y) in enumerate(self.nodes):
            if (x - event.x) ** 2 + (y - event.y) ** 2 <= 100:
                self.add_edge(node)
                return
        self.add_node(event)

    def add_node(self, event):
        node_id = len(self.nodes)
        self.nodes.append((event.x, event.y))
        self.graph.add_node(node_id, pos=(event.x, event.y))
        self.history.append(("add_node", node_id))
        self.draw_graph()

    def add_edge(self, node):
        if self.selected_node is None:
            self.selected_node = node
        else:
            if self.selected_node != node:
                first_node = self.selected_node
                second_node = node
                weight = simpledialog.askinteger("Вес ребра", "Введите вес ребра:")

                if weight is not None:
                    if not self.graph.has_edge(first_node, second_node):
                        self.graph.add_edge(first_node, second_node, weight=weight)
                        self.history.append(("add_edge", (first_node, second_node, weight)))
                        self.update_table(first_node, second_node, weight)
                    else:
                        old_weight = self.graph[first_node][second_node]['weight']
                        self.graph[first_node][second_node]['weight'] = weight
                        self.history.append(("update_edge", (first_node, second_node, old_weight)))
                        self.update_table(first_node, second_node, weight)
                    self.selected_node = None
                    self.draw_graph()

    def update_table(self, from_node, to_node, weight):
        self.tree.insert("", "end", values=(from_node, to_node, weight))

    def draw_graph(self):
        self.canvas.delete("all")
        pos = nx.get_node_attributes(self.graph, 'pos')

        for edge in self.graph.edges(data=True):
            node1, node2, data = edge
            weight = data.get('weight', 0)
            x1, y1 = pos[node1]
            x2, y2 = pos[node2]

            dx, dy = (x2 - x1), (y2 - y1)
            length = (dx ** 2 + dy ** 2) ** 0.5
            unit_dx, unit_dy = dx / length, dy / length

            arrow_dx, arrow_dy = unit_dx * 15, unit_dy * 15
            text_dx, text_dy = unit_dx * 20, unit_dy * 20

            self.canvas.create_line(
                x1 + arrow_dx, y1 + arrow_dy, x2 - arrow_dx, y2 - arrow_dy,
                fill="blue", width=2, arrow=tk.LAST
            )

            text_x = (x1 + x2) / 2 + text_dx
            text_y = (y1 + y2) / 2 + text_dy

            if abs(dx) > abs(dy):
                text_y += 15 if dy > 0 else -15
            else:
                text_x += 15 if dx > 0 else -15

            self.canvas.create_text(
                text_x, text_y, text=str(weight), fill="red", font=("Helvetica", 10, "bold")
            )

        for node, (x, y) in pos.items():
            self.canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="orange")
            self.canvas.create_text(x, y, text=str(node), fill="black")

    def cencelator(self):
        if not self.history:
            messagebox.showinfo("Информация", "Нет действий для отмены")
            return

        last_action = self.history.pop()
        action_type, data = last_action

        if action_type == "add_node":
            node_id = data
            self.graph.remove_node(node_id)
            self.nodes.pop(node_id)
        elif action_type == "add_edge":
            first_node, second_node, weight = data
            self.graph.remove_edge(first_node, second_node)
            self.remove_from_table(first_node, second_node)
        elif action_type == "update_edge":
            first_node, second_node, old_weight = data
            self.graph[first_node][second_node]['weight'] = old_weight
            self.update_table(first_node, second_node, old_weight)

        self.draw_graph()

    def remove_from_table(self, from_node, to_node):
        for item in self.tree.get_children():
            values = self.tree.item(item, "values")
            if values and int(values[0]) == from_node and int(values[1]) == to_node:
                self.tree.delete(item)
                break

    def nearest_neighbor(self, start_node):
        path = [start_node]
        unvisited = set(self.graph.nodes) - {start_node}

        while unvisited:
            last_node = path[-1]
            nearest_node = None
            min_weight = float('inf')

            for node in unvisited:
                if self.graph.has_edge(last_node, node):
                    weight = self.graph[last_node][node]['weight']
                    if weight < min_weight:
                        nearest_node = node
                        min_weight = weight

            if nearest_node is None:
                return None

            path.append(nearest_node)
            unvisited.remove(nearest_node)

        if not self.graph.has_edge(path[-1], path[0]):
            return None

        return path

    def run_nn(self):
        if len(self.graph.nodes) < 2:
            messagebox.showerror("Ошибка", "Граф должен содержать хотя бы 2 вершины")
            return

        start_time = time.perf_counter()
        start_node = random.choice(list(self.graph.nodes))
        path = self.nearest_neighbor(start_node)
        elapsed = time.perf_counter() - start_time

        if path:
            distance = self.calculate_path_distance(path)
            self.update_output(f"{distance:.2f}", f"{elapsed:.6f} сек")
            self.draw_result(path)
        else:
            messagebox.showinfo("Информация", "Путь не найден")

    def run_rnn(self):
        if len(self.graph.nodes) < 2:
            messagebox.showerror("Ошибка", "Граф должен содержать хотя бы 2 вершины")
            return

        start_time = time.perf_counter()
        best_path = None
        best_distance = float('inf')

        for start_node in self.graph.nodes:
            path = self.nearest_neighbor(start_node)
            if path:
                distance = self.calculate_path_distance(path)
                if distance < best_distance:
                    best_path = path
                    best_distance = distance

        elapsed = time.perf_counter() - start_time

        if best_path:
            self.update_output(f"{best_distance:.2f}", f"{elapsed:.6f} сек")
            self.draw_result(best_path)
        else:
            messagebox.showinfo("Информация", "Путей нет")

    def calculate_path_distance(self, path):
        distance = 0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i + 1])
            if edge_data is None:
                return float('inf')
            distance += edge_data['weight']
        edge_data = self.graph.get_edge_data(path[-1], path[0])
        if edge_data is None:
            return float('inf')
        distance += edge_data['weight']
        return distance

    def draw_result(self, path):
        self.result_canvas.delete("all")
        pos = nx.get_node_attributes(self.graph, 'pos')

        for i in range(len(path)):
            x1, y1 = pos[path[i]]
            x2, y2 = pos[path[(i + 1) % len(path)]]

            dx, dy = x2 - x1, y2 - y1
            length = (dx ** 2 + dy ** 2) ** 0.5
            if length == 0:
                continue

            unit_dx, unit_dy = dx / length, dy / length
            arrow_start_x = x1 + unit_dx * 15
            arrow_start_y = y1 + unit_dy * 15
            arrow_end_x = x2 - unit_dx * 15
            arrow_end_y = y2 - unit_dy * 15

            self.result_canvas.create_line(
                arrow_start_x, arrow_start_y,
                arrow_end_x, arrow_end_y,
                fill="green", width=3, arrow=tk.LAST
            )

            text_x = (x1 + x2) / 2 + unit_dx * 20
            text_y = (y1 + y2) / 2 + unit_dy * 20

            if abs(dx) > abs(dy):
                text_y += 15 if dy > 0 else -15
            else:
                text_x += 15 if dx > 0 else -15

            weight = self.graph[path[i]][path[(i + 1) % len(path)]]['weight']
            self.result_canvas.create_text(
                text_x, text_y,
                text=str(weight),
                fill="red", font=("Helvetica", 10, "bold")
            )

        for node, (x, y) in pos.items():
            self.result_canvas.create_oval(x - 10, y - 10, x + 10, y + 10, fill="orange")
            self.result_canvas.create_text(x, y, text=str(node), fill="black")

    def clear_graph(self):
        self.graph.clear()
        self.nodes = []
        self.selected_node = None
        self.history = []
        self.canvas.delete("all")
        self.result_canvas.delete("all")
        self.distance_label.config(text="Длина пути: -")
        self.time_label.config(text="Время: -")
        self.tree.delete(*self.tree.get_children())

    def simulated_annealing(self, initial_temp, cooling_rate, iterations):
        current_path = self.get_initial_path()
        if current_path is None:
            return None

        current_distance = self.calculate_path_distance(current_path)
        best_path = current_path.copy()
        best_distance = current_distance
        temp = initial_temp

        for i in range(iterations):
            new_path = self.get_neighbor_path(current_path)
            new_distance = self.calculate_path_distance(new_path)

            if new_distance == float('inf'):
                continue

            if new_distance < current_distance or \
                    (temp > 0 and random.random() < math.exp((current_distance - new_distance) / temp)):
                current_path, current_distance = new_path, new_distance

                if current_distance < best_distance:
                    best_path, best_distance = current_path.copy(), current_distance

            temp *= cooling_rate

        return best_path if best_distance != float('inf') else None

    def run_annealing(self):
        if not self.is_graph_valid():
            messagebox.showerror("Ошибка", "Граф должен содержать хотя бы 2 вершины")
            return

        params = self.get_annealing_params()
        if not params:
            return

        start_time = time.perf_counter()
        path = self.simulated_annealing(**params)
        elapsed = time.perf_counter() - start_time

        if path is None:
            messagebox.showinfo("Информация",
                                "Не удалось найти допустимый маршрут.\n"
                                "Попробуйте увеличить количество итераций\n"
                                "или проверить связность графа.")
            return

        distance = self.calculate_path_distance(path)
        self.update_output(f"{distance:.2f}", f"{elapsed:.6f} сек")
        self.draw_result(path)

    def boltzmann_annealing(self, initial_temp, max_iterations):
        best_path = None
        for start_node in self.graph.nodes:
            path = self.nearest_neighbor(start_node)
            if path and self.calculate_path_distance(path) != float('inf'):
                best_path = path
                break

        if not best_path:
            best_path = list(self.graph.nodes)
            random.shuffle(best_path)
            if self.calculate_path_distance(best_path) == float('inf'):
                return None

        best_distance = self.calculate_path_distance(best_path)
        current_path = best_path.copy()
        current_distance = best_distance
        temp = initial_temp

        for k in range(1, max_iterations + 1):
            new_path = current_path.copy()
            a, b = random.sample(range(len(new_path)), 2)
            new_path[a], new_path[b] = new_path[b], new_path[a]

            new_distance = self.calculate_path_distance(new_path)
            if new_distance == float('inf'):
                continue

            if new_distance < current_distance or random.random() < math.exp((current_distance - new_distance) / temp):
                current_path, current_distance = new_path, new_distance
                if current_distance < best_distance:
                    best_path, best_distance = current_path.copy(), current_distance
            temp = initial_temp / math.log(1 + k)
        return best_path if best_distance != float('inf') else None

    def run_boltzmann_annealing(self):
        params = self.get_boltzmann_params()
        if not params:
            return

        if len(self.graph.nodes) < 2:
            messagebox.showerror("Ошибка", "Граф должен содержать хотя бы 2 вершины")
            return

        start_time = time.perf_counter()
        path = self.boltzmann_annealing(**params)
        elapsed = time.perf_counter() - start_time

        if path:
            distance = self.calculate_path_distance(path)
            self.update_output(f"{distance:.2f}", f"{elapsed:.6f} сек")
            self.draw_result(path)
        else:
            messagebox.showinfo("Информация", "Путь не найден")

    def get_initial_path(self):
        for _ in range(5):
            start_node = random.choice(list(self.graph.nodes))
            path = self.nearest_neighbor(start_node)
            if path and self.calculate_path_distance(path) != float('inf'):
                return path

        for _ in range(100):
            path = list(self.graph.nodes)
            random.shuffle(path)
            if self.calculate_path_distance(path) != float('inf'):
                return path

        return None

    def get_neighbor_path(self, path):
        new_path = path.copy()
        n = len(new_path)
        if n < 4:
            i, j = random.sample(range(n), 2)
            new_path[i], new_path[j] = new_path[j], new_path[i]
        else:
            i, j = sorted(random.sample(range(1, n - 1), 2))
            new_path[i:j + 1] = reversed(new_path[i:j + 1])
        return new_path

    def is_graph_valid(self):
        if len(self.graph.nodes) < 2:
            return False
        for node in self.graph.nodes:
            if self.graph.out_degree(node) == 0:
                return False
        return True
    def update_output(self, distance_text, time_text):
        """Обновляет поля вывода длины пути и времени"""
        self.distance_label.config(text=f"Длина пути: {distance_text}")
        self.time_label.config(text=f"Время: {time_text}")

if __name__ == "__main__":
    root = tk.Tk()
    app = GraphApp(root)
    root.mainloop()