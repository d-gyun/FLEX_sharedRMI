import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# -----------------------------
# Linear Regression Model
# -----------------------------
class LinearModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, keys, positions):
        if len(keys) == 0:
            self.model.coef_ = np.array([0.0])
            self.model.intercept_ = 0.0
            return
        X = np.array(keys).reshape(-1, 1)
        y = np.array(positions)
        self.model.fit(X, y)

    def predict(self, key):
        X = np.array([[key]])
        pred = self.model.predict(X)[0]
        return int(pred)

# -----------------------------
# DataNode
# -----------------------------
class DataNode:
    def __init__(self, keys, density=0.7):
        self.keys = sorted(keys)
        self.model = LinearModel()
        self.shared = False
        self.num_keys = len(self.keys)
        self.capacity = int(len(self.keys) / density) if self.keys else 64
        positions = list(range(self.num_keys))
        self.model.train(self.keys, positions)

    @property
    def keys_list(self):
        return self.keys

# -----------------------------
# InternalNode
# -----------------------------
class InternalNode:
    def __init__(self, split_keys):
        self.children = []
        self.split_keys = split_keys
        self.model = LinearModel()

    def finalize_children(self, children):
        self.children = children
        positions = list(range(len(self.children)))
        if len(self.split_keys) != len(positions):
            print(f"[WARNING] split_keys({len(self.split_keys)}) and positions({len(positions)}) mismatch.")
        self.model.train(self.split_keys, positions)

    def route(self, key):
        child_idx = self.model.predict(key)
        child_idx = max(0, min(child_idx, len(self.children) - 1))
        return self.children[child_idx]

    @property
    def keys_list(self):
        all_keys = []
        for child in self.children:
            all_keys.extend(child.keys_list)
        return sorted(all_keys)

# -----------------------------
# FLEX RMI (BFS Implementation)
# -----------------------------
class FLEX_RMI:
    def __init__(self):
        self.root = None
        self.data_nodes = []
        self.shared_nodes = []

    def build(self, data, min_partition_size=1024, max_data_node_size=4096):
        sorted_data = sorted(data)
        current_level = [(sorted_data, None)]
        parents_without_parent = []
        level = 0

        while current_level:
            print(f"\n[LEVEL {level}] Processing...")
            next_level = []
            parent_to_children = {}

            for partition, parent in current_level:
                if len(partition) <= min_partition_size:
                    node = DataNode(partition)
                    self.data_nodes.append(node)
                elif len(partition) > max_data_node_size or not self.is_linear(partition):
                    split_points = self.find_split_points(partition, num_splits=4)
                    partitions = self.partition_data(partition, split_points)

                    if not partitions or any(len(p) == 0 for p in partitions):
                        print(f"[WARNING] Empty partitions found for data {partition[:5]}...")
                        node = DataNode(partition)
                        self.data_nodes.append(node)
                        continue

                    split_keys = [part[0] for part in partitions]
                    if len(split_keys) <= 1:
                        print(f"[WARNING] Only one split key found. Making DataNode.")
                        node = DataNode(partition)
                        self.data_nodes.append(node)
                        continue

                    print(f"[DEBUG] Creating InternalNode with split_keys {split_keys}")
                    node = InternalNode(split_keys)
                    for part in partitions:
                        next_level.append((part, node))
                else:
                    node = DataNode(partition)
                    self.data_nodes.append(node)

                if parent:
                    if parent not in parent_to_children:
                        parent_to_children[parent] = []
                    parent_to_children[parent].append(node)
                else:
                    parents_without_parent.append(node)

            for parent, children in parent_to_children.items():
                parent.finalize_children(children)
                print(f"[DEBUG] Parent {parent} assigned {len(children)} children")

            parent_list = list(parent_to_children.keys())
            for i in range(1, len(parent_list)):
                prev_parent = parent_list[i - 1]
                curr_parent = parent_list[i]

                if not prev_parent.children or not curr_parent.children:
                    print(f"[WARNING] Parent {i - 1} or {i} has no children")
                    continue

                prev_last_child = prev_parent.children[-1]
                curr_first_child = curr_parent.children[0]

                print(f"[DEBUG] Merge check between {prev_last_child} and {curr_first_child}")

                if self.is_cdf_similar(prev_last_child, curr_first_child):
                    shared_node = self.create_shared_node(prev_last_child, curr_first_child)
                    prev_parent.children[-1] = shared_node
                    curr_parent.children[0] = shared_node
                    print(f"[DEBUG] SharedNode created at Level {level}")

            current_level = next_level
            level += 1

        if len(parents_without_parent) == 1:
            self.root = parents_without_parent[0]
        else:
            split_keys = [node.keys_list[0] for node in parents_without_parent if node.keys_list]
            if not split_keys:
                print("[ERROR] No valid split keys found when creating root InternalNode.")
                self.root = DataNode(sorted_data)
            else:
                print(f"[DEBUG] Creating root InternalNode with split_keys {split_keys}")
                self.root = InternalNode(split_keys)
                self.root.finalize_children(parents_without_parent)

    def create_shared_node(self, left_node, right_node):
        merged_keys = sorted(left_node.keys_list + right_node.keys_list)

        if len(merged_keys) <= 1024 or self.is_linear(merged_keys):
            shared_node = DataNode(merged_keys, density=0.6)
        else:
            split_points = self.find_split_points(merged_keys, num_splits=4)
            partitions = self.partition_data(merged_keys, split_points)
            if len(partitions) <= 1 or any(len(p) == 0 for p in partitions):
                shared_node = DataNode(merged_keys, density=0.6)
            else:
                split_keys = [p[0] for p in partitions]
                shared_node = InternalNode(split_keys)
                shared_node.finalize_children([DataNode(p) for p in partitions])

        shared_node.shared = True
        self.shared_nodes.append(shared_node)

        print(f"[DEBUG] SharedNode created with {len(merged_keys)} keys")
        return shared_node

    def is_linear(self, keys):
        if len(keys) < 2:
            return True
        x = np.array(keys)
        y = np.arange(len(keys))
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        predicted = m * x + c
        error = np.abs(predicted - y)
        return np.max(error) < len(keys) * 0.2

    def is_cdf_similar(self, node_a, node_b, threshold=0.1):
        keys_a = node_a.keys_list
        keys_b = node_b.keys_list

        if not keys_a or not keys_b:
            print("[WARNING] One or both nodes have no keys during CDF similarity check")
            return False

        len_sample = min(len(keys_a), len(keys_b), 100)
        sample_a = np.array(keys_a)[np.linspace(0, len(keys_a) - 1, len_sample, dtype=int)]
        sample_b = np.array(keys_b)[np.linspace(0, len(keys_b) - 1, len_sample, dtype=int)]

        mae = np.mean(np.abs(sample_a - sample_b))
        print(f"[MERGE TEST] MAE: {mae}, Threshold: {threshold}")

        return mae < threshold

    def find_split_points(self, keys, num_splits=4):
        n = len(keys)
        if n < 2:
            return []

        x = np.array(keys)
        unique_x, _ = np.unique(x, return_index=True)

        if len(unique_x) < 2:
            return []

        cdf = np.arange(len(unique_x)) / len(unique_x)

        try:
            gradients = np.gradient(cdf, unique_x)
        except Exception:
            return []

        gradients = np.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)

        top_k = min(num_splits - 1, len(gradients) - 1)
        if top_k <= 0:
            return []

        split_indices = np.argsort(-gradients)[:top_k]
        split_indices.sort()
        split_points = [unique_x[idx] for idx in split_indices]

        return split_points

    def partition_data(self, keys, split_points):
        partitions = []
        current = []
        idx = 0

        for point in split_points:
            while idx < len(keys) and keys[idx] < point:
                current.append(keys[idx])
                idx += 1
            if current:
                partitions.append(current)
                current = []

        remaining = keys[idx:]
        if remaining:
            partitions.append(remaining)

        partitions = [p for p in partitions if p]

        if len(partitions) <= 1:
            return [keys]

        return partitions

# -----------------------------
# Visualization (Optional for debugging)
# -----------------------------
def visualize_tree_structure(level_counts):
    levels = list(level_counts.keys())
    internal_nodes = [counts['internal'] for counts in level_counts.values()]
    data_nodes = [counts['data'] for counts in level_counts.values()]

    fig, ax = plt.subplots()
    ax.bar(levels, internal_nodes, label='Internal Nodes')
    ax.bar(levels, data_nodes, bottom=internal_nodes, label='Data Nodes')

    ax.set_xlabel("Tree Levels")
    ax.set_ylabel("Number of Nodes")
    ax.set_title("FLEX RMI Tree Structure by Level")
    ax.legend()
    plt.show()
