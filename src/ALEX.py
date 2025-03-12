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

    def search_with_cost(self, key):
        pred_pos = self.model.predict(key)
        intra_node_cost = 1  # 모델 예측 1회 포함

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

    def exponential_search_with_cost(self, key, pred_pos):
        search_cost = 0

        # 오른쪽 방향 탐색
        bound = 1
        while pred_pos + bound < self.num_keys and self.keys[pred_pos + bound] < key:
            bound *= 2
            search_cost += 1

        left = pred_pos + bound // 2
        right = min(pred_pos + bound, self.num_keys - 1)

        # 왼쪽 방향 탐색
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

        # 분할 조건 강화
        if len(keys) <= min_partition_size or self.is_linear(keys):
            node = DataNode(keys)
            self.data_nodes.append(node)
            return node

        split_points = self.find_split_points(keys)

        # split_points가 없다면 더 이상 나눌 수 없음
        if not split_points:
            node = DataNode(keys)
            self.data_nodes.append(node)
            return node

        partitions = self.partition_data(keys, split_points)

        # 파티션이 제대로 생성되지 않았을 경우
        if len(partitions) <= 1:
            node = DataNode(keys)
            self.data_nodes.append(node)
            return node

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
        CDF 기반 split point 계산 + 안전성 강화:
        변화량(gradient)을 기준으로 변동이 심한 구간에서 분할 포인트 선정
        """
        n = len(keys)
        if n < 2:
            return []  # 더 이상 나눌 수 없음

        x = np.array(keys)

        # 중복값 제거 후 gradient 계산
        unique_x, unique_indices = np.unique(x, return_index=True)

        if len(unique_x) < 2:
            # 모든 값이 동일하거나 분할 불가 → 더 이상 분할하지 않음
            return []

        cdf = np.arange(len(unique_x)) / len(unique_x)

        try:
            gradients = np.gradient(cdf, unique_x)
        except Exception as e:
            print(f"[Warning] Gradient computation failed: {e}")
            return []

        # NaN, inf 필터링
        gradients = np.nan_to_num(gradients, nan=0.0, posinf=0.0, neginf=0.0)

        # 변화량이 큰 지점 선택
        top_k = min(num_splits - 1, len(gradients) - 1)
        if top_k <= 0:
            return []

        split_indices = np.argsort(-gradients)[0:top_k]
        split_indices.sort()

        # split point 변환 (원본 인덱스 기준)
        split_points = [unique_x[idx] for idx in split_indices]

        return split_points

    def partition_data(self, keys, split_points):
        """
        split_points에 따라 데이터를 나누는 함수.
        파티션이 제대로 나뉘지 않는 경우 대비 안전장치 포함.
        """
        partitions = []
        current = []
        idx = 0

        for point in split_points:
            while idx < len(keys) and keys[idx] < point:
                current.append(keys[idx])
                idx += 1

            if len(current) == 0:
                # 분할 실패 → 이전 파티션을 그대로 넣는다.
                continue

            partitions.append(current)
            current = []

        # 남은 데이터 추가
        remaining = keys[idx:]
        if len(remaining) > 0:
            partitions.append(remaining)

        # 하나의 파티션만 존재하는 경우 → 더 이상 분할이 의미 없음
        if len(partitions) <= 1:
            return [keys]

        return partitions

    def search_with_cost(self, key):
        """
        Search 동작에서 TraverseToLeafCost, IntraNodeCost, TotalCost 반환
        """
        node = self.root
        traverse_to_leaf_cost = 0

        # InternalNode 탐색 단계
        while isinstance(node, InternalNode):
            traverse_to_leaf_cost += 1
            node = node.route(key)

        # DataNode 탐색 단계
        found, intra_node_cost = node.search_with_cost(key)

        total_cost = traverse_to_leaf_cost + intra_node_cost

        return {
            "found": found is not None,
            "TraverseToLeafCost": traverse_to_leaf_cost,
            "IntraNodeCost": intra_node_cost,
            "TotalCost": total_cost
        }

    def insert_with_split_logging(self, key):
        """
        삽입 수행 후 split 발생 여부 반환
        True -> split 발생
        False -> split 없음
        """
        node = self.root
        path = []

        while isinstance(node, InternalNode):
            path.append(node)
            node = node.route(key)

        split_needed = node.insert(key)

        split_occurred = False
        if split_needed:
            self.handle_split(path, node)
            split_occurred = True

        return split_occurred

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

            # ALEX에서 fanout의 크기를 2의 제곱으로 설정하여 부모 노드의 재학습을 최소화
            # parent.model.train(parent.split_keys, list(range(len(parent.children))))
