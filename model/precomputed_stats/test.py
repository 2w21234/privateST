import pandas as pd
import numpy as np
import pickle
import os

# 파일 목록 및 예상 타입 정의
files_info = {
    'gene.pkl': 'pickle',
    'image_gene_stats.csv': 'csv',
    'mean_expression.npy': 'numpy',
    'subtype.pkl': 'pickle'
}

print("=== 현재 디렉토리 파일 내용 확인 시작 ===\n")

for filename, ftype in files_info.items():
    print(f"[{filename}] 로딩 중...")
    
    if not os.path.exists(filename):
        print(f"⚠️ 파일이 존재하지 않습니다: {filename}\n")
        continue

    try:
        # 1. CSV 파일 (image_gene_stats.csv)
        if ftype == 'csv':
            data = pd.read_csv(filename)
            print(f"▶ 데이터 타입: DataFrame")
            print(f"▶ Shape: {data.shape}")
            print(f"▶ 컬럼 목록: {list(data.columns)}")
            print("▶ 내용 미리보기 (Head):")
            print(data.head(3))

        # 2. NumPy 파일 (mean_expression.npy)
        elif ftype == 'numpy':
            data = np.load(filename)
            print(f"▶ 데이터 타입: NumPy Array")
            print(f"▶ Shape: {data.shape}")
            print(f"▶ Dtype: {data.dtype}")
            print(f"▶ 값 미리보기 (Flattened top 5): {data.flatten()[:5]}")

        # 3. Pickle 파일 (gene.pkl, subtype.pkl)
        elif ftype == 'pickle':
            # Pandas 객체일 경우와 일반 Python 객체일 경우를 모두 고려
            try:
                data = pd.read_pickle(filename)
                is_pandas = True
            except:
                with open(filename, 'rb') as f:
                    data = pickle.load(f)
                is_pandas = False
            
            print(f"▶ 데이터 타입: {type(data)}")
            
            if is_pandas:
                if isinstance(data, (pd.DataFrame, pd.Series)):
                    print(f"▶ Shape: {data.shape}")
                    print("▶ 내용 미리보기:")
                    print(data.head(3) if isinstance(data, pd.DataFrame) else data[:5])
            else:
                # 리스트나 딕셔너리인 경우 길이와 샘플 출력
                length = len(data) if hasattr(data, '__len__') else 'N/A'
                print(f"▶ 길이(Length): {length}")
                print(f"▶ 샘플 데이터: {str(data)[:200]} ...") # 너무 길면 자름

    except Exception as e:
        print(f"❌ 읽기 오류 발생: {e}")
    
    print("-" * 50 + "\n")

print("=== 확인 완료 ===")
