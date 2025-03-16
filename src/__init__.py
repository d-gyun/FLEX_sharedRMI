from ALEX import ALEX_RMI, DataNode as ALEX_DataNode
from FLEX import FLEX_RMI, DataNode as FLEX_DataNode
from analysis import report_tree_structure, benchmark_search_cost_cdf, benchmark_insert_with_split_tracking
import numpy as np

def load_dataset(file_path):
    return np.loadtxt(file_path, delimiter=",")

def main():
    # 데이터 로드
    data_file = "../datasets/dataset_LOGNORMAL_100000.csv"
    insert_file = "../datasets/dataset_LOGNORMAL_10000.csv"

    initial_data = load_dataset(data_file)
    insert_data = load_dataset(insert_file)

    print(f"초기 데이터 {len(initial_data)}개 로드 완료.")
    print(f"삽입 데이터 {len(insert_data)}개 로드 완료.")

    # ALEX 인덱스 빌드 및 분석
    alex = ALEX_RMI()
    alex.build(initial_data.tolist())

    # FLEX 인덱스 빌드 및 분석
    flex = FLEX_RMI()
    flex.build(initial_data.tolist())

    # 실험 1 - 트리 구조 분석 (DataNode 클래스를 넘김)
    report_tree_structure("ALEX", alex, ALEX_DataNode)
    report_tree_structure("FLEX", flex, FLEX_DataNode)


    # 실험 2 - Search Cost CDF (전체 데이터 1/10 샘플)
    search_keys = initial_data[np.random.choice(len(initial_data), len(initial_data)//10, replace=False)]
    benchmark_search_cost_cdf("ALEX", alex, search_keys)
    # benchmark_search_cost_cdf("FLEX", flex, search_keys)

    # 실험 3 - Insert + Split 추적
    # benchmark_insert_with_split_tracking("ALEX", alex, insert_data)
    # benchmark_insert_with_split_tracking("FLEX", flex, insert_data)

if __name__ == "__main__":
    main()
