# import sys
# sys.path.append('/workspace/SCU-text-evaluation')
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = ""
# import random
# import numpy as np
# import torch
# from transformers import set_seed as hf_set_seed, set_seed
# # ===== 시드 고정 =====
# def fix_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
#     set_seed(seed)
#     hf_set_seed(seed)

# fix_seed(42)

# import re
# import spacy
# import penman
# from penman.models import amr
# from penman.graph import Graph
# from amrlibfd.amrlib.graph_processing.amr_fix import maybe_fix_unlinked_in_subgraph
# #import amrlibfd.amrlib as amrlib
# import amrlib
# import spacy

# from collections import OrderedDict
# from allennlp.predictors.predictor import Predictor

# import re
# # AllenNLP coref 모델 로드
# coref_predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

# spacy_o = spacy.load("en_core_web_sm")


# GENERIC_TERMS = [
#     "person", "people", "man", "woman", "individual",
#     "thing", "the thing", "something", "object", "item", "entity",
#     "event", "situation", "place", "location", "area",
#     "group", "organization","he", "she", "his", "her", "they", "them", "their"
# ]

# def resolve_coref_allennlp(original_text: str, atomic_sentence: str) -> str:
#     resolved = atomic_sentence

#     try:
#         # full resolved 문장
#         coref_result = coref_predictor.predict(document=original_text)
#         resolved_text = coref_result.get("document", None)
#         clusters = coref_result.get("clusters", [])

#         if resolved_text and clusters:
#             token_list = resolved_text
#             reconstructed_text = " ".join(token_list)

#             for cluster in clusters:
#                 main_mention = cluster[0]
#                 main_text = " ".join(token_list[main_mention[0]:main_mention[1]+1])

#                 for mention in cluster[1:]:
#                     mention_text = " ".join(token_list[mention[0]:mention[1]+1])
#                     # 대명사가 atomic sentence에 있는 경우만 치환 시도
#                     if mention_text.lower() in atomic_sentence.lower():
#                         resolved = re.sub(rf"\b{re.escape(mention_text)}\b", main_text, resolved, flags=re.IGNORECASE)

#     except Exception as e:
#         print(f"[WARN] AllenNLP coref failed: {e}")

#     return resolved


# def replace_general_terms_with_specific(original_text: str, atomic_sentence: str) -> str:
#     atomic_sentence = resolve_coref_allennlp(original_text, atomic_sentence)  # ← 먼저 coref 처리

#     doc = spacy_o(original_text)
#     resolved = atomic_sentence

#     # generic term 보정
#     # candidates = set(chunk.text.lower() for chunk in doc.noun_chunks)
#     # candidates.update(token.text.lower() for token in doc if token.pos_ == "NOUN")

#     # for generic in GENERIC_TERMS:
#     #     if generic in resolved:
#     #         best_match = None
#     #         best_score = 0.0
#     #         for cand in candidates:
#     #             score = SequenceMatcher(None, generic, cand).ratio()
#     #             if score > best_score:
#     #                 best_score = score
#     #                 best_match = cand

#     #         if best_match and best_score < 1.0:
#     #             resolved = resolved.replace(generic, best_match)

#     return resolved




# # Load models and spaCy

# DIR_STOG_MODEL = '../model_parse_xfm_bart_large-v0_1_0'
# DIR_GTOS_MODEL = '../model_generate_t5wtense-v0_1_0'
# stog = amrlib.load_stog_model(DIR_STOG_MODEL)
# gtos = amrlib.load_gtos_model(DIR_GTOS_MODEL)

# # Util functions
# def gstring_to_oneline(gstring):
#     graph_lines = [line.strip() for line in gstring.splitlines() if not line.startswith('#')]
#     gstring = ' '.join(graph_lines)
#     return re.sub(' +', ' ', gstring)

# def get_concepts(g_tag):
#     tokens = g_tag.split()
#     dict_concepts = {}
#     for t in tokens:
#         if "~" in t:
#             t = t.replace("(", "").replace(")", "")
#             parts_t = t.split("~")
#             dict_concepts[parts_t[0]] = t
#     return dict_concepts

# def replace_graph_with_tags(dict_tag, graph):
#     for key, value in dict_tag.items():
#         graph = graph.replace(key, value)
#     return graph





# # Main logic
# def process_summary(summary: str):
#     sentences = [s.text for s in spacy_o(summary).sents]
    
#     graphs = stog.parse_sents(sentences, add_metadata=True)
#     print("=== Parsed Graphs ===")
#     print(graphs)
#     graph_tags = [''] * len(graphs)  # 빈 태그, 실제로 tag 사용 안함

#     all_atomic_facts = []
#     for s, g, g_tag in zip(sentences, graphs, graph_tags):
#         # root에 :purpose, :cause, :time 없는 경우 skip
#         g_penman = penman.decode(g, model=amr.model)
#         root = g_penman.triples[0][0]
#         root_triples = [t for t in g_penman.triples if t[0] == root]
#         related_roles = {":purpose", ":cause", ":time"}
#         if not any(t[1] in related_roles for t in root_triples):
#             print(f"-> Skipped (no relevant semantic role like :cause, :purpose, :time)")
#             continue

#         dict_tag = get_concepts(g_tag)
#         subgraphs = get_subgraphs_relation_aware(g)

#         # 🔍 추출된 subgraph AMR 출력
#         print("=== Extracted Subgraphs ===")
#         for i, sb in enumerate(subgraphs, 1):
#             print(f"[Subgraph {i}]")
#             print(sb.strip())

#         subgraphs_tag = []
#         for sb in subgraphs:
#             sb = maybe_fix_unlinked_in_subgraph(g, sb)
#             sb = gstring_to_oneline(sb)
#             sb = replace_graph_with_tags(dict_tag, sb)
#             subgraphs_tag.append(sb)

#         sents, _ = gtos.generate(    subgraphs_tag,
#             disable_progress=True,
#             num_beams=1,
#             do_sample=False,
#             early_stopping=True
#             )
        
#         for sent in sents:
#             postprocessed = replace_general_terms_with_specific(summary, sent)
#             all_atomic_facts.append(postprocessed)
#             # 🔍 추출된 subgraph AMR 출력
#             print("=== atomic facts before coreference resolution ===")
#             print(sent.strip())
#         #all_atomic_facts.extend(sents)

#     return all_atomic_facts






# def get_subgraphs_relation_aware(amr_graph):
#     g = penman.decode(amr_graph, model=amr.model)
#     root = g.triples[0][0]
#     root_triples = [t for t in g.triples if t[0] == root]

#     # 관심 있는 semantic role 및 predicate 정의
#     related_roles = {":purpose", ":cause", ":time"}
#     related_preds = {"cause-01", "purpose-01", "time-entity", "before", "after"}

#     subgraph_triples = list(root_triples)
#     related_targets = set()

#     # 1. Role 기반 대상 찾기 (루트 기준)
#     for t in root_triples:
#         if t[1] in related_roles:
#             related_targets.add(t[2])
#             subgraph_triples.append(t)

#     # 2. Predicate 이름 기반 대상 찾기 (모든 노드 대상)
#     predicate_nodes = set()
#     for t in g.triples:
#         subj, role, obj = t
#         if role == ":instance" and obj in related_preds:
#             predicate_nodes.add(subj)

#     related_targets.update(predicate_nodes)

#     # ✅ 3. 루트의 :ARG0이 person인 경우 → 그 주변 triple까지 확장
#     for t in root_triples:
#         if t[1] == ":ARG0":
#             arg0_var = t[2]
#             for triple in g.triples:
#                 if triple[0] == arg0_var and triple[1] == ":instance" and triple[2] == "person":
#                     for t2 in g.triples:
#                         if t2[0] == arg0_var or t2[2] == arg0_var:
#                             subgraph_triples.append(t2)

#     # 4. 관련 대상 노드와 연결된 triple 추가
#     for target in related_targets:
#         for t in g.triples:
#             if t[0] == target or t[2] == target:
#                 subgraph_triples.append(t)

#     # ✅ 순서를 유지한 중복 제거
#     subgraph_triples = list(OrderedDict.fromkeys(subgraph_triples))
#     return [penman.format(penman.configure(Graph(subgraph_triples)))]

# if __name__ == '__main__':
    
#     input_sentences = [
#         "A police chief resigned from a police post to appear on BBC Question Time.",
#         # "The company delayed the product launch due to technical issues.",
#         # "She studied hard to pass the final exam.",
#         # "They canceled the concert after the storm hit the city.",
#         # "The scientist published her findings before attending the conference.",
#         # "I love NLP."
#     ]

#     for idx, sentence in enumerate(input_sentences, 1):
#         print(f"\n\n\n=== Input Sentence {idx} ===")
#         print(sentence)
#         atomic_facts = process_summary(sentence)

#         print("=== Atomic Facts ===")
#         for i, fact in enumerate(atomic_facts, 1):
#             print(f"{i}. {fact}")