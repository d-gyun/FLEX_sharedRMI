import numpy as np
from ALEX_RMI import ALEX
import time

def load_dataset(file_path):
    """
    CSV 파일에서 key 값만 로드하여 반환
    """
    return np.loadtxt(file_path, delimiter=",")

def main():
    # 데이터 파일 경로
    data_file = "../datasets/dataset_LONGLAT_10000.csv"   # 초기 데이터 (검색용)
    insert_file = "../datasets/dataset_LONGLAT_1000.csv"  # 삽입할 추가 데이터

    # 데이터 로드
    initial_data = load_dataset(data_file)
    insert_data = load_dataset(insert_file)

    # RMI 인덱스 초기화 및 빌드
    ALEX_RMI = ALEX_RMI()
    print("RMI 인덱스 생성 중...")
    ALEX_RMI.build(initial_data.tolist())  # 리스트 형태로 변환 후 빌드
    print("RMI 인덱스 초기화 완료.")

    # Search 테스트
    search_keys = initial_data[:10]  # 일부 키 선택하여 검색 테스트
    print("\n Search 테스트 시작:")
    for key in search_keys:
        result = ALEX_RMI.search(key)
        if result is not None:
            print(f"Key {key} 검색 성공: {result}")
        else:
            print(f"Key {key} 검색 실패")

    # Insert 테스트
    print("\n Insert 테스트 시작:")
    start_time = time.time()
    for key in insert_data:
        ALEX_RMI.insert(key)
    end_time = time.time()
    print(f"{len(insert_data)}개 키 삽입 완료. 삽입 시간: {end_time - start_time:.4f}초")


if __name__ == "__main__":
    main()
