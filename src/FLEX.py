import numpy as np
from sklearn.linear_model import LinearRegression

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
# FLEX DataNode
# -----------------------------
class DataNode:
    def __init__(self, keys, density=0.7, min_density=0.6, max_density=0.8):
        self.keys = sorted(keys)
        self.model = LinearModel()
        self.parents = []
        self.shared = False
        self.density = density
        self.min_density = min_density
        self.max_density = max_density
        self.capacity = int(len(keys) / density) if keys else 64
        self.num_keys = len(self.keys)
        positions = list(range(self.num_keys))
        self.model.train(self.keys, positions)

# -----------------------------
# FLEX InternalNode
# -----------------------------
class InternalNode:
    def __init__(self, keys, partitions, children=None):
        self.model = LinearModel()
        self.children = children if children else []
        self.partitions = partitions

        positions = self.assign_partition_labels(keys, partitions)
        self.model.train(keys, positions)

    def assign_partition_labels(self, keys, partitions):
        labels = []
        for key in keys:
            label = self.find_partition_index(key, partitions)
            labels.append(label)
        return labels

    def find_partition_index(self, key, partitions):
        for idx, part in enumerate(partitions):
            if part[0] <= key <= part[-1]:
                return idx
        return len(partitions) - 1

    def route(self, key):
        child_idx = self.model.predict(key)
        child_idx = max(0, min(child_idx, len(self.children) - 1))
        return self.children[child_idx]

    @property
    def keys(self):
        all_keys = []
        for child in self.children:
            all_keys.extend(child.keys)
        return sorted(all_keys)

# -----------------------------
# FLEX Recursive Model Index with Shared RMI (BFS Implementation)
# -----------------------------
class FLEX_RMI:
    def __init__(self):
        self.root = None
        self.data_nodes = []
        self.shared_nodes = []

    def build(self, data, min_partition_size=32):
        sorted_data = sorted(data)
        queue = [(sorted_data, None)]
        parents_without_parent = []

        while queue:
            current_level = queue
            queue = []
            current_nodes = []
            parent_to_children = {}

            for partition, parent in current_level:
                if len(partition) <= min_partition_size or self.is_linear(partition):
                    node = DataNode(partition)
                    self.data_nodes.append(node)
                    current_nodes.append((node, parent))
                else:
                    split_points = self.find_split_points(partition)
                    if not split_points:
                        node = DataNode(partition)
                        self.data_nodes.append(node)
                        current_nodes.append((node, parent))
                        continue
                    part_list = self.partition_data(partition, split_points)
                    if len(part_list) <= 1:
                        node = DataNode(partition)
                        self.data_nodes.append(node)
                        current_nodes.append((node, parent))
                        continue
                    internal_node = InternalNode(partition, part_list)
                    current_nodes.append((internal_node, parent))
                    for part in part_list:
                        queue.append((part, internal_node))

            for node, parent in current_nodes:
                if parent:
                    if parent not in parent_to_children:
                        parent_to_children[parent] = []
                    parent_to_children[parent].append(node)
                else:
                    parents_without_parent.append(node)

            for parent, children in parent_to_children.items():
                parent.children = children
                print(f"[DEBUG] Parent {parent} assigned {len(children)} children")

            internal_nodes = [node for node, _ in current_nodes if isinstance(node, InternalNode)]
            print(f"[DEBUG] InternalNodes count at this level: {len(internal_nodes)}")

            for i in range(1, len(internal_nodes)):
                prev_internal = internal_nodes[i - 1]
                curr_internal = internal_nodes[i]

                if not prev_internal.children or not curr_internal.children:
                    print("[WARNING] One of the internals has no children")
                    continue

                prev_last_child = prev_internal.children[-1]
                curr_first_child = curr_internal.children[0]

                print(f"[DEBUG] Checking merge between {prev_last_child} and {curr_first_child}")

                if self.is_cdf_similar(prev_last_child, curr_first_child, threshold=0.1):
                    shared_node = self.create_shared_node(prev_last_child, curr_first_child)
                    prev_internal.children[-1] = shared_node
                    curr_internal.children[0] = shared_node
                    shared_node.parents.append(prev_internal)
                    shared_node.parents.append(curr_internal)

        if len(parents_without_parent) == 1:
            self.root = parents_without_parent[0]
        else:
            self.root = InternalNode(sorted_data, [n.keys for n in parents_without_parent], parents_without_parent)

    def create_shared_node(self, left_node, right_node, min_partition_size=32):
        shared_keys = sorted(left_node.keys + right_node.keys)

        if len(shared_keys) <= min_partition_size or self.is_linear(shared_keys):
            shared_node = DataNode(shared_keys, density=0.6)
        else:
            split_points = self.find_split_points(shared_keys)
            partitions = self.partition_data(shared_keys, split_points)
            if len(partitions) <= 1 or self.is_linear(shared_keys):
                shared_node = DataNode(shared_keys, density=0.6)
            else:
                children = [DataNode(part) for part in partitions]
                shared_node = InternalNode(shared_keys, partitions, children)

        shared_node.shared = True
        shared_node.parents = []
        self.shared_nodes.append(shared_node)

        print(f"[DEBUG] Shared node created with {len(shared_keys)} keys")
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
        return np.max(error) < len(keys) * 0.05

    def is_cdf_similar(self, node_a, node_b, threshold=0.1):
        keys_a = node_a.keys
        keys_b = node_b.keys

        if not keys_a or not keys_b:
            return False

        len_sample = min(len(keys_a), len(keys_b), 100)
        sample_indices_a = np.linspace(0, len(keys_a) - 1, len_sample, dtype=int)
        sample_indices_b = np.linspace(0, len(keys_b) - 1, len_sample, dtype=int)

        sample_a = np.array(keys_a)[sample_indices_a]
        sample_b = np.array(keys_b)[sample_indices_b]

        mae = np.mean(np.abs(sample_a - sample_b))
        print(f"[MERGE TEST] MAE: {mae}, Threshold: {threshold}")

        return mae < threshold

    def find_split_points(self, keys, num_splits=2):
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
        except Exception as e:
            print(f"[Warning] Gradient computation failed: {e}")
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

        if len(partitions) <= 1:
            return [keys]

        return partitions
