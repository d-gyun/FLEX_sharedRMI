import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------
# Linear Regression Model
# -----------------------------
class LinearModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, keys, positions):
        """
        keys: 리스트 형태
        positions: 리스트 형태
        """
        if len(keys) == 0:
            # 비어있을 경우 기본값
            self.model.coef_ = np.array([0.0])
            self.model.intercept_ = 0.0
            return

        X = np.array(keys).reshape(-1, 1)  # sklearn은 2차원 배열 요구
        y = np.array(positions)
        self.model.fit(X, y)

    def predict(self, key):
        X = np.array([[key]])  # 2차원으로 넣어줘야 함
        pred = self.model.predict(X)[0]
        return int(pred)

# -----------------------------
# DataNode (Leaf)
# -----------------------------
class DataNode:
    def __init__(self, keys, density=0.7, min_density=0.6, max_density=0.8):
        self.keys = []
        self.model = LinearModel()

        self.density = density
        self.min_density = min_density
        self.max_density = max_density
        self.capacity = int(len(keys) / density) if keys else 64

        self.num_keys = 0
        self.sorted_insert(keys)

    def sorted_insert(self, keys):
        sorted_keys = sorted(keys)
        for k in sorted_keys:
            self.keys.append(k)
        self.num_keys = len(self.keys)
        positions = list(range(self.num_keys))
        self.model.train(self.keys, positions)

    def is_full(self):
        return self.num_keys / self.capacity >= self.max_density

    def search(self, key):
        pred_pos = self.model.predict(key)
        if pred_pos < 0:
            pred_pos = 0
        elif pred_pos >= self.num_keys:
            pred_pos = self.num_keys - 1

        if self.keys[pred_pos] == key:
            return self.keys[pred_pos]

        found_idx = self.exponential_search(key, pred_pos)
        if found_idx is not None:
            return self.keys[found_idx]
        return None

    def insert(self, key):
        idx = self.binary_search_insert_position(key)
        self.keys.insert(idx, key)
        self.num_keys += 1

        if self.is_full():
            return True
        return False

    def binary_search_insert_position(self, key):
        left, right = 0, self.num_keys
        while left < right:
            mid = (left + right) // 2
            if self.keys[mid] < key:
                left = mid + 1
            else:
                right = mid
        return left

    def exponential_search(self, key, pred_pos):
        bound = 1
        while pred_pos + bound < self.num_keys and self.keys[pred_pos + bound] < key:
            bound *= 2
        left = pred_pos + bound // 2
        right = min(pred_pos + bound, self.num_keys - 1)

        l_bound = 1
        while pred_pos - l_bound >= 0 and self.keys[pred_pos - l_bound] > key:
            l_bound *= 2
        l_left = max(pred_pos - l_bound, 0)
        l_right = pred_pos - l_bound // 2

        result = self._binary_search_range(left, right, key)
        if result is not None:
            return result
        return self._binary_search_range(l_left, l_right, key)

    def _binary_search_range(self, left, right, key):
        while left <= right:
            mid = (left + right) // 2
            if self.keys[mid] == key:
                return mid
            elif self.keys[mid] < key:
                left = mid + 1
            else:
                right = mid - 1
        return None

# -----------------------------
# InternalNode (RMI 트리 노드)
# -----------------------------
class InternalNode:
    def __init__(self, keys, children=None):
        self.model = LinearModel()
        self.children = children if children is not None else []
        self.split_keys = keys  # 자식 노드별 key range 나누기 용도

        # 모델 학습 (자식 노드 선택)
        positions = list(range(len(self.children)))
        self.model.train(self.split_keys, positions)

    def route(self, key):
        # key에 대해 적절한 child 선택
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

    def build(self, data, min_partition_size=32):
        sorted_data = sorted(data)
        self.root = self._build_recursive(sorted_data, min_partition_size)

    def _build_recursive(self, data, min_partition_size):
        keys = data

        if len(keys) <= min_partition_size or self.is_linear(keys):
            node = DataNode(keys)
            self.data_nodes.append(node)
            return node

        # CDF 변화량 기반으로 split points 찾기
        split_points = self.find_split_points(keys)

        # 데이터를 파티션으로 나누기
        partitions = self.partition_data(keys, split_points)

        children = []
        split_keys = []
        for part in partitions:
            if not part:
                continue
            split_keys.append(part[0])
            child = self._build_recursive(part, min_partition_size)
            children.append(child)

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
        return np.max(error) < len(keys) * 0.05

    def find_split_points(self, keys, num_splits=2):
        """
        CDF 기반 split point 계산:
        변화량(gradient)을 기준으로 변동이 심한 구간에서 분할 포인트 선정
        """
        n = len(keys)
        x = np.array(keys)
        cdf = np.arange(n) / n

        # CDF 변화량 (기울기) 계산
        gradients = np.gradient(cdf, x)

        # 변화량이 큰 지점 우선으로 split point 후보 선정
        top_k = num_splits - 1
        split_indices = np.argsort(-gradients)[0:top_k]

        # split index를 정렬하고 split point 반환
        split_indices.sort()
        split_points = [keys[idx] for idx in split_indices]

        return split_points

    def partition_data(self, keys, split_points):
        partitions = []
        current = []
        idx = 0

        for point in split_points:
            while idx < len(keys) and keys[idx] < point:
                current.append(keys[idx])
                idx += 1
            partitions.append(current)
            current = []
        partitions.append(keys[idx:])
        return partitions

    def search(self, key):
        node = self.root
        while isinstance(node, InternalNode):
            node = node.route(key)
        return node.search(key)

    def insert(self, key):
        node = self.root
        path = []

        while isinstance(node, InternalNode):
            path.append(node)
            node = node.route(key)

        split_needed = node.insert(key)

        if split_needed:
            self.handle_split(path, node)

    def handle_split(self, path, node):
        idx = self.data_nodes.index(node)
        keys = node.keys

        mid = len(keys) // 2
        left_keys = keys[:mid]
        right_keys = keys[mid:]

        left_node = DataNode(left_keys)
        right_node = DataNode(right_keys)

        self.data_nodes[idx] = left_node
        self.data_nodes.insert(idx + 1, right_node)

        if not path:
            self.root = InternalNode(
                [left_keys[0], right_keys[0]],
                [left_node, right_node]
            )
        else:
            parent = path[-1]
            insert_idx = parent.children.index(node)
            parent.children[insert_idx] = left_node
            parent.children.insert(insert_idx + 1, right_node)
            parent.split_keys.insert(insert_idx + 1, right_keys[0])

            parent.model.train(parent.split_keys, list(range(len(parent.children))))
