import numpy as np
from sklearn.linear_model import LinearRegression

# -----------------------------
# Linear Regression Model
# -----------------------------
class LinearModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, keys, positions):
        if len(keys) == 0 or len(positions) == 0:
            print("[ERROR] LinearModel.train() received empty data")
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
    def __init__(self, split_keys, partitions=None):
        self.split_keys = split_keys
        self.partitions = partitions  # 자식 노드 생성을 위한 partition 정보
        self.children = []
        self.model = LinearModel()

    def finalize_children(self):
        if not self.children:
            print("[ERROR] InternalNode finalize_children() called with no children.")
            return
        positions = list(range(len(self.children)))
        self.model.train(self.split_keys, positions)

    @property
    def keys_list(self):
        all_keys = []
        for part in self.partitions:
            all_keys.extend(part)
        return sorted(all_keys)

# -----------------------------
# FLEX RMI
# -----------------------------
class FLEX_RMI:
    def __init__(self):
        self.root = None
        self.data_nodes = []
        self.shared_nodes = []

    def build(self, data, min_partition_size=1024, max_data_node_size=4096):
        sorted_data = sorted(data)

        # ---------------------------
        # 1. 루트 노드 생성 및 자식 노드 초기화
        # ---------------------------
        if len(sorted_data) <= min_partition_size:
            self.root = DataNode(sorted_data)
            self.data_nodes.append(self.root)
            print(f"[INFO] Root is a DataNode")
            return

        split_points = self.find_split_points(sorted_data)
        partitions = self.partition_data(sorted_data, split_points)

        if len(partitions) <= 1:
            self.root = DataNode(sorted_data)
            self.data_nodes.append(self.root)
            print(f"[INFO] Root fallback to DataNode")
            return

        split_keys = [part[0] for part in partitions]
        self.root = InternalNode(split_keys, partitions)
        print(f"[INFO] Root InternalNode created with split_keys {split_keys}")

        # ---------------------------
        # 2. BFS 순회
        # ---------------------------
        current_level = [self.root]
        level = 0

        while current_level:
            print(f"\n[LEVEL {level}] Processing...")

            next_level = []
            prev_node = None

            for node in current_level:
                if isinstance(node, DataNode):
                    continue  # 리프 노드일 경우 패스

                # 자식 노드 생성
                children = []
                for partition in node.partitions:
                    if len(partition) <= min_partition_size:
                        child_node = DataNode(partition)
                        self.data_nodes.append(child_node)
                    else:
                        sub_split_points = self.find_split_points(partition)
                        sub_partitions = self.partition_data(partition, sub_split_points)

                        if len(sub_partitions) <= 1:
                            child_node = DataNode(partition)
                            self.data_nodes.append(child_node)
                        else:
                            sub_split_keys = [p[0] for p in sub_partitions]
                            child_node = InternalNode(sub_split_keys, sub_partitions)
                    children.append(child_node)

                node.children = children
                node.finalize_children()

                next_level.extend(children)

                # 병합 판단
                if prev_node and isinstance(prev_node, InternalNode) and isinstance(node, InternalNode):
                    prev_last_child = prev_node.children[-1]
                    curr_first_child = node.children[0]

                    print(f"[DEBUG] Merge check between {prev_last_child} and {curr_first_child}")

                    if self.is_cdf_similar(prev_last_child, curr_first_child):
                        # InternalNode끼리 병합일 경우만 InternalNode 생성
                        if isinstance(prev_last_child, InternalNode) and isinstance(curr_first_child, InternalNode):
                            shared_node = self.create_shared_node(prev_last_child, curr_first_child, force_internal=True)
                        else:
                            shared_node = self.create_shared_node(prev_last_child, curr_first_child, force_internal=False)

                        prev_node.children[-1] = shared_node
                        node.children[0] = shared_node
                        prev_node.finalize_children()
                        node.finalize_children()

                        print(f"[DEBUG] SharedNode created at Level {level}")

                prev_node = node  # 이전 노드 갱신

            current_level = next_level
            level += 1

    def create_shared_node(self, left_node, right_node, force_internal=False):
        merged_keys = sorted(left_node.keys_list + right_node.keys_list)

        # --- ✅ InternalNode 병합만 InternalNode로 유지 ---
        if force_internal:
            split_points = self.find_split_points(merged_keys)
            partitions = self.partition_data(merged_keys, split_points)

            if len(partitions) <= 1:
                shared_node = DataNode(merged_keys)
                self.data_nodes.append(shared_node)
            else:
                split_keys = [p[0] for p in partitions]
                shared_node = InternalNode(split_keys, partitions)
        else:
            # 무조건 DataNode로 처리
            shared_node = DataNode(merged_keys)
            self.data_nodes.append(shared_node)

        shared_node.shared = True
        self.shared_nodes.append(shared_node)
        return shared_node

    # -----------------------------
    # 헬퍼 메소드들
    # -----------------------------
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

    def is_cdf_similar(self, node_a, node_b, threshold=0.01):
        keys_a = node_a.keys_list
        keys_b = node_b.keys_list

        if not keys_a or not keys_b:
            print("[WARNING] One or both nodes have no keys during similarity check")
            return False

        def normalize(keys):
            min_v, max_v = keys[0], keys[-1]
            if min_v == max_v:
                return [0.5] * len(keys)
            return [(k - min_v) / (max_v - min_v) for k in keys]

        norm_a = normalize(keys_a)
        norm_b = normalize(keys_b)

        len_sample = min(len(norm_a), len(norm_b), 100)
        sample_a = np.array(norm_a)[np.linspace(0, len(norm_a)-1, len_sample, dtype=int)]
        sample_b = np.array(norm_b)[np.linspace(0, len(norm_b)-1, len_sample, dtype=int)]

        mae = np.mean(np.abs(sample_a - sample_b))
        print(f"[MERGE TEST] Normalized MAE: {mae}, Threshold: {threshold}")

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
        return [unique_x[idx] for idx in split_indices]

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

        return partitions if len(partitions) > 1 else [keys]
