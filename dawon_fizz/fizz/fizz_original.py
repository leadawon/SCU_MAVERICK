import os
import json


import argparse


# Argument Parser 설정
parser = argparse.ArgumentParser(description='fairy')

# 필수 인자들 (기본값 제거)
parser.add_argument('--input_path', type=str, required=True,
                    help='입력 JSON 파일 경로. 예: --input_path data/xsumfaith.json')
parser.add_argument('--output_path', type=str, required=True,
                    help='출력 JSON 파일 경로. 예: --output_path results/fizz_rouge_sbert_xsumfaith.json')
parser.add_argument('--granularity', type=str, required=True,
                    help='점수 계산 시 사용되는 granularity 설정. 예: --granularity 3G')
parser.add_argument('--cuda_device', type=str, required=True,
                    help='사용할 CUDA 디바이스 번호. 예: --cuda_device 0')
parser.add_argument('--mode', type=str, required=True,
                    help='운영 모드 (예: original,rouge_sbert 등). 예: --mode test')
parser.add_argument('--weight_rouge', type=float, required=True,
                    help='ROUGE 점수 가중치. 예: --weight_rouge 0.3')
parser.add_argument('--weight_bert', type=float, required=True,
                    help='BERTScore 가중치. 예: --weight_bert 0.7')
parser.add_argument('--base', type=float, required=True,
                    help='softmin의 밑. 예: --base 2.0')
parser.add_argument('--exponent_delta', type=float, required=True,
                    help='softmin x축 이동량 예: --exponent_delta 0.2')
# parser.add_argument('--base_flex', type=float, required=True,
#                     help='summary chunk개수에 따른 지수 밑 변동량 예: --base_flex 0.2')
parser.add_argument('--weight_min', type=float, required=True,
                    help='min 가중치. 예: --weight_min 0.3')
parser.add_argument('--weight_mean', type=float, required=True,
                    help='mean 가중치. 예: --weight_mean 0.7')
# 선택 인자들 (기본값 유지)
parser.add_argument('--doc_label', type=str, default='document')
parser.add_argument('--summary_label', type=str, default='claim')
parser.add_argument('--label_label', type=str, default='label')
parser.add_argument('--score_column', type=str, default='FIZZ_score')
parser.add_argument('--model_name', type=str, default='orca2')



args = parser.parse_args()
# CUDA 디바이스 환경 설정
os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_device
from atomic_fact_decomposition import AtomicFactDecomposer
from atomic_fact_filtering import AtomicFactFilterer
from atomic_fact_scoring import AtomicFactScorer
from tqdm import tqdm
import nltk
from rouge_score import rouge_scorer
from bert_score import BERTScorer
nltk.download('punkt')





def main_original():
    #assert False, "This function is not implemented yet. Please use the fizz_coarse.py script instead."
    print(f"Running in mode: {args.mode}")
    print(f"Using CUDA device: {args.cuda_device}")

    # 입력 파일 로드
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
        cut = entry.get("cut", None)

        # Atomic Fact 추출 및 처리
        atomic_facts = decomposer.atomic_facts_decompose(summary)
        filtered_atomic_facts = filterer.atomic_facts_filtering(summary, atomic_facts)
        score = scorer.atomic_facts_scoring(doc, filtered_atomic_facts,weight_min= args.weight_min, weight_mean=args.weight_mean)

        # 결과 저장
        result = {
            "document": doc,
            "summary": summary,
            "label": label,
            "score": score,
            "cut": cut
        }
        output_data.append(result)

    # 출력 저장
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)



def main_rouge_sbert():
    assert False, "This function is not implemented yet. Please use the fizz_coarse.py script instead."
    # 입력 파일 로드 (JSON 형식)
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    decomposer = AtomicFactDecomposer(model_name=args.model_name)
    filterer = AtomicFactFilterer()
    scorer = AtomicFactScorer(granularity=args.granularity)
    rouge_scorer_obj = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    output_data = []

    for entry in tqdm(data, desc="Processing", mininterval=0.01):
        doc = entry[args.doc_label]
        summary = entry[args.summary_label]
        label = entry[args.label_label]
        cut = entry.get("cut", None)

        # 1. Atomic Fact 추출 및 필터링
        atomic_facts = decomposer.atomic_facts_decompose(summary)
        filtered_atomic_facts = filterer.atomic_facts_filtering(summary, atomic_facts)

        # 2. Filtered Atomic Facts를 Sentence로 분할
        summary_sentences = decomposer.split_sentences(filtered_atomic_facts)
        doc_sentences = decomposer.split_sentences(doc)

        # 3. 가장 유사한 문장 선택
        selected_sentences = set()
        for fact in summary_sentences:
            best_score = 0
            best_sentence = None

            for doc_sentence in doc_sentences:
                # ROUGE 점수 계산
                rouge_score = rouge_scorer_obj.score(fact, doc_sentence)['rougeL'].fmeasure

                # BERTScore 계산
                P, R, F1 = bert_scorer.score([doc_sentence], [fact])
                bert_score = F1.item()

                # 가중 평균 계산
                combined_score = (args.weight_rouge * rouge_score) + (args.weight_bert * bert_score)

                if combined_score > best_score:
                    best_score = combined_score
                    best_sentence = doc_sentence

            if best_sentence:
                selected_sentences.add(best_sentence)

        # 4. 선택된 문장에 대해 Atomic Fact 분해 및 필터링
        modified_sentences = []
        for sentence in doc_sentences:
            modified_sentences.append(sentence)  # 선택된 문장도 먼저 추가
            if sentence in selected_sentences:
                atomic_facts = decomposer.atomic_facts_decompose(sentence)
                filtered_facts = filterer.atomic_facts_filtering(sentence, atomic_facts)
                fact_sentences = decomposer.split_sentences(filtered_facts)
                modified_sentences.extend(fact_sentences)

        # 5. 순서 보존된 문서 재구성
        final_doc = ' '.join(modified_sentences)

        # 6. Score 계산
        score = scorer.atomic_facts_scoring(final_doc, filtered_atomic_facts)

        # 7. 결과 저장
        result = {
            "document": final_doc,
            "summary": summary,
            "label": label,
            "score": score,
            "cut": cut
        }
        output_data.append(result)

    # output_path 수정 (no_discards 추가)
    output_path = args.output_path.replace(".json", "_no_discards.json")
    # JSON 파일로 저장
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=4)




if __name__ == "__main__":
    if args.mode == "original":
        main_original()
    elif args.mode == "rouge_sbert":
        main_rouge_sbert()
