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

# -----------------------------
# FLEX Recursive Model Index with Shared RMI
# -----------------------------
class FLEX_RMI:
    def __init__(self):
        self.root = None
        self.data_nodes = []
        self.shared_nodes = []

    def build(self, data, min_partition_size=32):
        sorted_data = sorted(data)
        self.root = self._build_recursive([sorted_data], min_partition_size)

    def _build_recursive(self, partitions, min_partition_size):
        parent_nodes = []
        child_partitions = []

        for keys in partitions:
            if len(keys) <= min_partition_size or self.is_linear(keys):
                node = DataNode(keys)
                self.data_nodes.append(node)
                parent_nodes.append(node)
                continue

            split_points = self.find_split_points(keys)
            part_list = self.partition_data(keys, split_points)

            if len(part_list) <= 1:
                node = DataNode(keys)
                self.data_nodes.append(node)
                parent_nodes.append(node)
                continue

            internal_node = InternalNode(keys, part_list, [None] * len(part_list))
            parent_nodes.append(internal_node)
            child_partitions.append(part_list)

        if not child_partitions:
            return parent_nodes[0] if len(parent_nodes) == 1 else parent_nodes

        flattened_partitions = [part for sublist in child_partitions for part in sublist]
        children_nodes = self._build_recursive(flattened_partitions, min_partition_size)

        idx = 0
        for parent in parent_nodes:
            if isinstance(parent, InternalNode):
                child_count = len(parent.partitions)
                parent.children = children_nodes[idx: idx + child_count]
                idx += child_count

        self._merge_adjacent_nodes(parent_nodes)

        return parent_nodes[0] if len(parent_nodes) == 1 else parent_nodes

    def _merge_adjacent_nodes(self, nodes, threshold=0.1):
        for i in range(1, len(nodes)):
            prev_node = nodes[i - 1]
            curr_node = nodes[i]

            if not isinstance(prev_node, InternalNode) or not isinstance(curr_node, InternalNode):
                continue

            prev_last_child = prev_node.children[-1] if prev_node.children else None
            curr_first_child = curr_node.children[0] if curr_node.children else None

            if prev_last_child and curr_first_child and self.is_cdf_similar(prev_last_child, curr_first_child, threshold):
                shared_node = self.create_shared_node(prev_last_child, curr_first_child)

                prev_node.children[-1] = shared_node
                curr_node.children[0] = shared_node

                shared_node.parents.append(prev_node)
                shared_node.parents.append(curr_node)

    def create_shared_node(self, left_node, right_node, min_partition_size=32):
        shared_keys = sorted(left_node.keys + right_node.keys)

        if len(shared_keys) <= min_partition_size or self.is_linear(shared_keys):
            shared_node = DataNode(shared_keys, density=0.6)
        else:
            split_points = self.find_split_points(shared_keys)
            partitions = self.partition_data(shared_keys, split_points)

            if len(partitions) <= 1:
                shared_node = DataNode(shared_keys, density=0.6)
            else:
                children = self._build_recursive(partitions, min_partition_size)
                shared_node = InternalNode(shared_keys, partitions, children)

        shared_node.shared = True
        shared_node.parents = []
        self.shared_nodes.append(shared_node)

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
        keys_a = node_a.keys if isinstance(node_a, DataNode) else []
        keys_b = node_b.keys if isinstance(node_b, DataNode) else []

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