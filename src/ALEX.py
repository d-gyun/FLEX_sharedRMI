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
# DataNode (Leaf)
# -----------------------------
class DataNode:
    def __init__(self, keys, density=0.7, min_density=0.6, max_density=0.8):
        self.keys = sorted(keys)
        self.model = LinearModel()
        self.density = density
        self.min_density = min_density
        self.max_density = max_density
        self.capacity = int(len(keys) / density) if keys else 64
        self.num_keys = len(self.keys)
        positions = list(range(self.num_keys))
        self.model.train(self.keys, positions)

    def is_full(self):
        return self.num_keys / self.capacity >= self.max_density

    def search_with_cost(self, key):
        pred_pos = self.model.predict(key)
        intra_node_cost = 1
        if pred_pos < 0:
            pred_pos = 0
        elif pred_pos >= self.num_keys:
            pred_pos = self.num_keys - 1
        if self.keys[pred_pos] == key:
            return self.keys[pred_pos], intra_node_cost
        found_idx, exp_search_cost = self.exponential_search_with_cost(key, pred_pos)
        intra_node_cost += exp_search_cost
        if found_idx is not None:
            return self.keys[found_idx], intra_node_cost
        return None, intra_node_cost

    def insert(self, key):
        idx = self.binary_search_insert_position(key)
        self.keys.insert(idx, key)
        self.num_keys += 1
        return self.is_full()

    def split(self):
        mid = len(self.keys) // 2
        left_node = DataNode(self.keys[:mid])
        right_node = DataNode(self.keys[mid:])
        return left_node, right_node

    def binary_search_insert_position(self, key):
        left, right = 0, self.num_keys
        while left < right:
            mid = (left + right) // 2
            if self.keys[mid] < key:
                left = mid + 1
            else:
                right = mid
        return left

    def exponential_search_with_cost(self, key, pred_pos):
        search_cost = 0
        bound = 1
        while pred_pos + bound < self.num_keys and self.keys[pred_pos + bound] < key:
            bound *= 2
            search_cost += 1
        left = pred_pos + bound // 2
        right = min(pred_pos + bound, self.num_keys - 1)
        l_bound = 1
        while pred_pos - l_bound >= 0 and self.keys[pred_pos - l_bound] > key:
            l_bound *= 2
            search_cost += 1
        l_left = max(pred_pos - l_bound, 0)
        l_right = pred_pos - l_bound // 2
        result, binary_search_cost = self._binary_search_range_with_cost(left, right, key)
        search_cost += binary_search_cost
        if result is not None:
            return result, search_cost
        result, binary_search_cost = self._binary_search_range_with_cost(l_left, l_right, key)
        search_cost += binary_search_cost
        return result, search_cost

    def _binary_search_range_with_cost(self, left, right, key):
        cost = 0
        while left <= right:
            mid = (left + right) // 2
            cost += 1
            if self.keys[mid] == key:
                return mid, cost
            elif self.keys[mid] < key:
                left = mid + 1
            else:
                right = mid - 1
        return None, cost

# -----------------------------
# InternalNode (RMI 트리 노드)
# -----------------------------
class InternalNode:
    def __init__(self, split_keys, children=None):
        self.model = LinearModel()
        self.children = children if children else []
        self.split_keys = split_keys
        positions = list(range(len(self.children)))
        self.model.train(self.split_keys, positions)

    def route(self, key):
        child_idx = self.model.predict(key)
        child_idx = max(0, min(child_idx, len(self.children) - 1))
        return self.children[child_idx]

# -----------------------------
# Recursive Model Index (RMI)
# -----------------------------
class ALEX_RMI:
    def __init__(self):
        self.root = None
        self.data_nodes = []

    def build(self, data, min_partition_size=1024, max_data_node_size=4096):
        sorted_data = sorted(data)
        self.root = self._build_recursive(sorted_data, min_partition_size, max_data_node_size)

    def _build_recursive(self, keys, min_partition_size, max_data_node_size):
        if len(keys) <= min_partition_size:
            node = DataNode(keys)
            self.data_nodes.append(node)
            return node

        # 강제로 내부 노드 생성하기 위한 조건
        if len(keys) > max_data_node_size:
            is_linear_result = False
        else:
            is_linear_result = self.is_linear(keys)

        if is_linear_result:
            node = DataNode(keys)
            self.data_nodes.append(node)
            return node

        split_points = self.find_split_points(keys, num_splits=4)
        if not split_points:
            node = DataNode(keys)
            self.data_nodes.append(node)
            return node

        partitions = self.partition_data(keys, split_points)
        if len(partitions) <= 1:
            node = DataNode(keys)
            self.data_nodes.append(node)
            return node

        children = []
        split_keys = []
        for part in partitions:
            child = self._build_recursive(part, min_partition_size, max_data_node_size)
            children.append(child)
            split_keys.append(part[0])

        return InternalNode(split_keys, children)

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
        if len(partitions) <= 1:
            return [keys]
        return partitions

    def search_with_cost(self, key):
        node = self.root
        traverse_to_leaf_cost = 0
        while isinstance(node, InternalNode):
            traverse_to_leaf_cost += 1
            node = node.route(key)
        found, intra_node_cost = node.search_with_cost(key)
        total_cost = traverse_to_leaf_cost + intra_node_cost
        return {
            "found": found is not None,
            "TraverseToLeafCost": traverse_to_leaf_cost,
            "IntraNodeCost": intra_node_cost,
            "TotalCost": total_cost
        }

    def insert_with_split_logging(self, key):
        node = self.root
        path = []
        while isinstance(node, InternalNode):
            path.append(node)
            node = node.route(key)
        split_needed = node.insert(key)
        if split_needed:
            self.handle_split(path, node)
            return True
        return False

    def handle_split(self, path, node):
        left_node, right_node = node.split()
        idx = self.data_nodes.index(node)
        self.data_nodes[idx] = left_node
        self.data_nodes.insert(idx + 1, right_node)
        if not path:
            self.root = InternalNode([left_node.keys[0], right_node.keys[0]], [left_node, right_node])
        else:
            parent = path[-1]
            insert_idx = parent.children.index(node)
            parent.children.pop(insert_idx)
            parent.children.insert(insert_idx, left_node)
            parent.children.insert(insert_idx + 1, right_node)
            parent.split_keys[insert_idx] = left_node.keys[0]
            parent.split_keys.insert(insert_idx + 1, right_node.keys[0])

