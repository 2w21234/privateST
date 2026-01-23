import pickle

filename = 'gene.pkl'

with open(filename, 'rb') as f:
    gene_list = pickle.load(f)

print(f"=== {filename} 상세 내용 ===")
print(f"▶ 총 유전자 개수: {len(gene_list)}")
print(f"▶ 데이터 예시 (첫 5개): {gene_list[:5]}")

# 리스트 내부 요소 타입 확인
if len(gene_list) > 0:
    print(f"▶ 요소 타입: {type(gene_list[0])}")

print("\n--- 처음 20개 유전자 ---")
print(gene_list[:20])

print("\n--- 마지막 20개 유전자 ---")
print(gene_list[-20:])

# 중복 여부 체크 (참고용)
if len(gene_list) != len(set(gene_list)):
    print(f"\n⚠️ 주의: 중복된 유전자 이름이 있습니다. (Unique: {len(set(gene_list))}개)")
else:
    print("\n✅ 중복된 유전자는 없습니다.")
