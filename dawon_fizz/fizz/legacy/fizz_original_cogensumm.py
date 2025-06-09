import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from atomic_fact_decomposition import AtomicFactDecomposer
from atomic_fact_filtering import AtomicFactFilterer
from atomic_fact_scoring import AtomicFactScorer
from tqdm import tqdm
import argparse
import nltk
nltk.download('punkt')
parser = argparse.ArgumentParser(description='fairy')

parser.add_argument('--input_path', type=str, default='data/cogensumm.json')
parser.add_argument('--output_path', type=str, default='results/fizz_original_cogensumm.json')
parser.add_argument('--doc_label', type=str, default='document')
parser.add_argument('--summary_label', type=str, default='claim')
parser.add_argument('--label_label', type=str, default='label')
parser.add_argument('--score_column', type=str, default='FIZZ_score')
parser.add_argument('--model_name', type=str, default='orca2')
parser.add_argument('--granularity', type=str, default='3G')

args = parser.parse_args()

def main():
    # 입력 파일 로드 (JSON 형식)
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    decomposer = AtomicFactDecomposer(model_name=args.model_name)
    filterer = AtomicFactFilterer()
    scorer = AtomicFactScorer(granularity=args.granularity)

    output_data = []

    for entry in tqdm(data, desc="Processing", mininterval=0.01):
        doc = entry[args.doc_label]
        summary = entry[args.summary_label]
        label = entry[args.label_label]

        # Atomic Fact 추출 및 필터링, 점수 계산
        atomic_facts = decomposer.atomic_facts_decompose(summary)
        filtered_atomic_facts = filterer.atomic_facts_filtering(summary, atomic_facts)
        score = scorer.atomic_facts_scoring(doc, filtered_atomic_facts)

        # 결과 데이터 구성
        result = {
            "document": doc,
            "summary": summary,
            "label": label,
            "score": score
        }
        output_data.append(result)

    # JSON 파일로 저장
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()
