import json
from pathlib import Path
from tqdm import tqdm
import generation.legacy_but_important.version_2.atomic_amr_short as aasc

# 모델 로딩
stog, gtos = aasc.load_models('/workspace/SCU_MAVERICK')

# 입력 및 출력 경로
input_path = Path("data/aggre_fact_xsum_sota.json")
output_path = Path("atomic_result/output_xsum.json")

# 중간 저장용
temp_results = []

# 데이터 로드
with input_path.open("r", encoding="utf-8") as f:
    entries = json.load(f)

# 처리
with output_path.open("a", encoding="utf-8") as f_a:
    for entry in tqdm(entries):
        claim = entry["claim"]
        atomic_facts = aasc.process_summary(claim, stog, gtos)
        result_entry = {
            "claim": claim,
            "atomic_facts": atomic_facts
        }
        temp_results.append(result_entry)
        f_a.write(json.dumps(result_entry, ensure_ascii=False) + "\n")

# 전체 결과 덮어쓰기
with output_path.open("w", encoding="utf-8") as f_w:
    json.dump(temp_results, f_w, indent=2, ensure_ascii=False)