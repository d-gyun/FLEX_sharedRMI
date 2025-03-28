import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# -----------------------------
# 실험 1: 트리 구조 통계 출력
# -----------------------------
def report_tree_structure(index_name, rmi, data_node_class):
    internal_node_count = _count_internal_nodes(rmi.root, data_node_class)
    data_node_count = len(rmi.data_nodes)
    tree_depth = get_tree_depth(rmi.root, data_node_class)

    print(f"\n[{index_name}] --- RMI Tree Structure ---")
    print(f"Internal Nodes : {internal_node_count}")
    print(f"Data Nodes     : {data_node_count}")
    print(f"Tree Depth     : {tree_depth}")

    if index_name == "FLEX":
        print(f"Shared Nodes   : {len(rmi.shared_nodes)}")

    print("----------------------------------------")

    return {
        "internal_nodes": internal_node_count,
        "data_nodes": data_node_count,
        "tree_depth": tree_depth,
        "shared_nodes": len(rmi.shared_nodes) if index_name == "FLEX" else None
    }


def _count_internal_nodes(node, data_node_class, visited=None):
    if visited is None:
        visited = set()

    if id(node) in visited:
        return 0  # DAG 중복 방지

    visited.add(id(node))

    if isinstance(node, data_node_class):
        return 0

    count = 1
    for child in node.children:
        count += _count_internal_nodes(child, data_node_class, visited)

    return count


# -----------------------------
# Tree Depth Helper
# -----------------------------
def get_tree_depth(node, data_node_class, current_depth=0, visited=None):
    if visited is None:
        visited = set()

    if id(node) in visited:
        return current_depth

    visited.add(id(node))

    if isinstance(node, data_node_class):
        return current_depth

    if not node.children:
        return current_depth

    return max(get_tree_depth(child, data_node_class, current_depth + 1, visited) for child in node.children)


# -----------------------------
# 실험 2: Search Cost CDF (TotalCost)
# -----------------------------
def benchmark_search_cost_cdf(index_name, rmi, search_keys):
    """
    전체 search key에 대해 TotalCost 수집 후 CDF 그래프 출력
    """
    total_costs = []

    missed_keys = []
    for key in search_keys:
        result = rmi.search_with_cost(key)
        total_costs.append(result["TotalCost"])
        if result["miss"]:
            missed_keys.append(key)

    total_costs = np.array(total_costs)
    total_costs.sort()

    cdf = np.arange(len(total_costs)) / len(total_costs)

    plt.figure(figsize=(10, 6))
    plt.plot(total_costs, cdf, marker='o', linestyle='-')
    plt.title(f"{index_name} Search TotalCost CDF")
    plt.xlabel("Total Search Cost")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()

    print(f"[{index_name}] 평균 TraverseToLeafCost : {np.mean(total_costs):.4f}, 최대 TraverseToLeafCost : {np.max(total_costs)}")
    print(f"[RESULT] Missed Keys: {len(missed_keys)} / {len(search_keys)} ({len(missed_keys) / len(search_keys) * 100:.2f}%)")

    return total_costs, cdf


# -----------------------------
# 실험 3: Split 발생 CDF (시간에 따른 Split 누적 수)
# -----------------------------
def benchmark_insert_with_split_tracking(index_name, rmi, insert_keys, interval=10):
    """
    insert 수행 시 split 발생 시점 추적 및 CDF 그래프 출력
    """
    split_log = defaultdict(list)  # {시간: 누적 split 횟수}
    split_count = 0
    time_steps = []
    splits_over_time = []

    for i, key in enumerate(insert_keys, 1):
        is_split = rmi.insert_with_split_logging(key)

        if is_split:
            split_count += 1

        # 일정 주기로 누적 split 횟수를 기록 (시간 축 시뮬레이션)
        if i % interval == 0 or i == len(insert_keys):
            time_steps.append(i)
            splits_over_time.append(split_count)

    splits_over_time = np.array(splits_over_time)
    cdf = np.arange(len(splits_over_time)) / len(splits_over_time)

    plt.figure(figsize=(10, 6))
    plt.plot(time_steps, splits_over_time, marker='o', linestyle='-')
    plt.title(f"{index_name} Cumulative Split Over Time")
    plt.xlabel("Insertion Count")
    plt.ylabel("Cumulative Split Count")
    plt.grid(True)
    plt.show()

    # Split Count CDF (원하는 경우)
    plt.figure(figsize=(10, 6))
    plt.plot(splits_over_time, cdf, marker='x', linestyle='-')
    plt.title(f"{index_name} Split Count CDF")
    plt.xlabel("Cumulative Split Count")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.show()

    print(f"[{index_name}] 총 Split 발생 수: {split_count}")

    return time_steps, splits_over_time, cdf
