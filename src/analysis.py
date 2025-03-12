import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# -----------------------------
# 실험 1: 트리 구조 통계 출력
# -----------------------------
def report_tree_structure(index_name, rmi, data_node_class):
    """
    트리 구조 출력 - Internal / Data 노드 개수
    data_node_class: ALEX/FLEX의 DataNode 클래스
    """
    internal_node_count = _count_internal_nodes(rmi.root, data_node_class)
    data_node_count = len(rmi.data_nodes)

    print(f"\n[{index_name}] --- RMI Tree Structure ---")
    print(f"Internal Nodes : {internal_node_count}")
    print(f"Data Nodes     : {data_node_count}")
    print("----------------------------------------")

    return {
        "internal_nodes": internal_node_count,
        "data_nodes": data_node_count
    }


def _count_internal_nodes(node, data_node_class):
    """
    재귀적으로 InternalNode 개수 카운트
    """
    if isinstance(node, data_node_class):
        return 0
    count = 1  # 현재 InternalNode 카운트
    for child in node.children:
        count += _count_internal_nodes(child, data_node_class)
    return count

# -----------------------------
# 실험 2: Search Cost CDF (TotalCost)
# -----------------------------
def benchmark_search_cost_cdf(index_name, rmi, search_keys):
    """
    전체 search key에 대해 TotalCost 수집 후 CDF 그래프 출력
    """
    total_costs = []

    for key in search_keys:
        result = rmi.search_with_cost(key)
        total_costs.append(result["TotalCost"])

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

    print(f"[{index_name}] 평균 TotalCost : {np.mean(total_costs):.4f}, 최대 TotalCost : {np.max(total_costs)}")

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
