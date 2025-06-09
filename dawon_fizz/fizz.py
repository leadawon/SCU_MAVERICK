import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import nltk
nltk.download('punkt')

from atomic_fact_decomposition import AtomicFactDecomposer
from atomic_fact_filtering import AtomicFactFilterer

# Summary 문장 리스트
summaries = [
    "A police chief resigned from a police post to appear on BBC Question Time.",
    "The company delayed the product launch due to technical issues.",
    "She studied hard to pass the final exam.",
    "They canceled the concert after the storm hit the city.",
    "The scientist published her findings before attending the conference.",
    "I love NLP."
]

# 필요한 객체 초기화
decomposer = AtomicFactDecomposer(model_name="orca2")
filterer = AtomicFactFilterer()

# 각 summary에 대해 필터링된 atomic facts 출력
for summary in summaries:
    atomic_facts = decomposer.atomic_facts_decompose(summary)
    filtered_atomic_facts = filterer.atomic_facts_filtering(summary, atomic_facts)
    
    print(f"\nSummary: {summary}")
    print("Filtered Atomic Facts:")

    # 문자열로 하나 붙어있는 경우 리스트로 변환
    if isinstance(filtered_atomic_facts, str):
        facts = [s.strip() for s in filtered_atomic_facts.strip().split(".") if s.strip()]
    else:
        facts = filtered_atomic_facts

    for fact in facts:
        print("-", fact.strip() + ".")
