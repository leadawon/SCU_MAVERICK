import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.tokenize import sent_tokenize
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')
from rouge_score import rouge_scorer
from bert_score import BERTScorer
import os
import math
import torch.nn.functional as F
model_map = {
    "tals": {"model_ckpt": "tals/albert-xlarge-vitaminc-mnli"}
}

class AtomicFactScorer:
    def __init__(self, model_name="tals", granularity="3G", device="cuda"):
        assert granularity in ["1G", "2G", "3G", "4G"], "Wrong granularity %s" % (granularity)
        assert model_name in model_map.keys(), "Wrong model name: `%s`" % (model_name)

        self.granularity = granularity
        self.gran = int(granularity[0]) + 1
        self.model_name = model_name
        self.model_ckpt = model_map[self.model_name]["model_ckpt"]
        self.model = None
        self.device = device

        ## dawon code

        self.weight_rouge = 0.3#weight_rouge
        self.weight_bert = 0.3#weight_bert
        self.rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)

    def load_lm(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_ckpt,
                                                       use_fast=False)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_ckpt)
        self.model.to(self.device)

    # from https://github.com/tingofurro/summac/
    def split_sentences(self, text):
        sentences = sent_tokenize(text)
        sentences = [sent for sent in sentences if len(sent)>10]
        return sentences
    

    # fizz original code
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)
        
    #     max_scores = []
    #     for decomposed_sentence in decomposed_sentences_list:
    #         decomposed_sentence_scores = [[], [], []]

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer([original_sentence.strip()],
    #                                     [decomposed_sentence.strip()],
    #                                     padding=True,
    #                                     truncation=True,
    #                                     return_tensors="pt").to(self.device)
    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 scores = torch.nn.functional.softmax(logits, dim=-1)
    #                 evid_score = np.array(scores[0][0].cpu()).item() # entailment score입니다.
    #                 conts_score = np.array(scores[0][1].cpu()).item()
    #                 neuts_score = np.array(scores[0][2].cpu()).item()

    #                 decomposed_sentence_scores[0].append(evid_score)
    #                 decomposed_sentence_scores[1].append(conts_score)
    #                 decomposed_sentence_scores[2].append(neuts_score)
            
    #         max_evid_score = max(decomposed_sentence_scores[0]) # compare할때 entailment score기준으로 compare하고 가장 높은 document chunk하나를 선택합니다.
    #         max_evid_idx = decomposed_sentence_scores[0].index(max_evid_score) # entailment score기준으로 가장 높은 document chunk의 인덱스를 기억합니다.

    #         if decomposed_sentence_scores[0][max_evid_idx] > decomposed_sentence_scores[1][max_evid_idx] and decomposed_sentence_scores[0][max_evid_idx] > decomposed_sentence_scores[2][max_evid_idx]:
    #             max_scores.append(max_evid_score) #expansion 안하면 그냥 chunk하나의 점수를 저장합니다.
    #         else: # 여기서 expansion 시작입니다.
    #             temp_scores = []
    #             expanded_original_sentences = []

    #             expanded_gran_idx_list = get_combinations(list(range(len(original_sentences_list))), self.gran, max_evid_idx) # 주위 chunk들의 combination을 구합니다.
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original_sentences = ""
    #                 for gran_idx in gran_idx_list:
    #                     new_original_sentences += original_sentences_list[gran_idx] + " "
    #                 expanded_original_sentences.append(new_original_sentences)
                
    #             for expanded_original_sentence in expanded_original_sentences: #그 combination 들의 entailemtn 점수를 구합니다.
    #                 features = self.tokenizer([expanded_original_sentence.strip()],
    #                                           [decomposed_sentence.strip()],
    #                                           padding=True,
    #                                           truncation=True,
    #                                           return_tensors="pt").to(self.device)
    #                 self.model.eval()
    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     scores = torch.nn.functional.softmax(logits, dim=-1)
    #                     evid_score = np.array(scores[0][0].cpu()).item()
    #                     temp_scores.append(evid_score)
                
    #             max_temp_score = max(temp_scores) # combination들의 e 점수중에 가장 큰값을 찾고
    #             max_scores.append(max(max_evid_score, max_temp_score)) # 가장 큰값을 저장합니다.
        
    #     min_max_score = min(max_scores) # final aggregate min을 사용합니다.
    #     # min_idx = max_scores.index(min_max_score)

    #     return min_max_score


    # fizz original code + min,mean
    def atomic_facts_scoring(self, original_text, decomposed_text, weight_min, weight_mean):
        self.load_lm()
        original_sentences_list = self.split_sentences(original_text)
        decomposed_sentences_list = self.split_sentences(decomposed_text)
        
        max_scores = []
        for decomposed_sentence in decomposed_sentences_list:
            decomposed_sentence_scores = [[], [], []]

            for original_sentence in original_sentences_list:
                features = self.tokenizer([original_sentence.strip()],
                                        [decomposed_sentence.strip()],
                                        padding=True,
                                        truncation=True,
                                        return_tensors="pt").to(self.device)
                self.model.eval()
                with torch.no_grad():
                    logits = self.model(**features).logits
                    scores = torch.nn.functional.softmax(logits, dim=-1)
                    evid_score = np.array(scores[0][0].cpu()).item() # entailment score입니다.
                    conts_score = np.array(scores[0][1].cpu()).item()
                    neuts_score = np.array(scores[0][2].cpu()).item()

                    decomposed_sentence_scores[0].append(evid_score)
                    decomposed_sentence_scores[1].append(conts_score)
                    decomposed_sentence_scores[2].append(neuts_score)
            
            max_evid_score = max(decomposed_sentence_scores[0]) # compare할때 entailment score기준으로 compare하고 가장 높은 document chunk하나를 선택합니다.
            max_evid_idx = decomposed_sentence_scores[0].index(max_evid_score) # entailment score기준으로 가장 높은 document chunk의 인덱스를 기억합니다.

            if decomposed_sentence_scores[0][max_evid_idx] > decomposed_sentence_scores[1][max_evid_idx] and decomposed_sentence_scores[0][max_evid_idx] > decomposed_sentence_scores[2][max_evid_idx]:
                max_scores.append(max_evid_score) #expansion 안하면 그냥 chunk하나의 점수를 저장합니다.
            else: # 여기서 expansion 시작입니다.
                temp_scores = []
                expanded_original_sentences = []

                expanded_gran_idx_list = get_combinations(list(range(len(original_sentences_list))), self.gran, max_evid_idx) # 주위 chunk들의 combination을 구합니다.
                for gran_idx_list in expanded_gran_idx_list:
                    new_original_sentences = ""
                    for gran_idx in gran_idx_list:
                        new_original_sentences += original_sentences_list[gran_idx] + " "
                    expanded_original_sentences.append(new_original_sentences)
                
                for expanded_original_sentence in expanded_original_sentences: #그 combination 들의 entailemtn 점수를 구합니다.
                    features = self.tokenizer([expanded_original_sentence.strip()],
                                              [decomposed_sentence.strip()],
                                              padding=True,
                                              truncation=True,
                                              return_tensors="pt").to(self.device)
                    self.model.eval()
                    with torch.no_grad():
                        logits = self.model(**features).logits
                        scores = torch.nn.functional.softmax(logits, dim=-1)
                        evid_score = np.array(scores[0][0].cpu()).item()
                        temp_scores.append(evid_score)
                
                max_temp_score = max(temp_scores) # combination들의 e 점수중에 가장 큰값을 찾고
                max_scores.append(max(max_evid_score, max_temp_score)) # 가장 큰값을 저장합니다.
        
        

        min_max_score = weight_min * min(max_scores) + weight_mean * np.mean(max_scores)
        # min_idx = max_scores.index(min_max_score)

        return min_max_score
    # #ecn score 구하는 코드
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment
    #                     expanded_scores.append((ecn_score, entailment))

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment)
    #             else:
    #                 max_scores.append(best_entailment_from_expansion)

    #     return min(max_scores)


    # #e + rouge + bertscore 구하는 코드
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         combined_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 복합 점수 계산 (entailment + ROUGE + BERT)
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()

    #             rouge_score = self.rouge_scorer.score(decomposed_sentence, original_sentence)['rougeL'].fmeasure
    #             P, R, F1 = self.bert_scorer.score([original_sentence], [decomposed_sentence])
    #             bert_score = F1.item()

    #             combined = entailment + self.weight_rouge * rouge_score + self.weight_bert * bert_score

    #             combined_scores.append(combined)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         # 2. 복합 점수 기준 최고 선택
    #         best_idx = int(np.argmax(combined_scores))
    #         best_entailment = entailment_scores[best_idx]
    #         best_contradiction = contradiction_scores[best_idx]
    #         best_neutral = neutral_scores[best_idx]

    #         # 3. dominant entailment 판단
    #         if best_entailment > best_contradiction and best_entailment > best_neutral:
    #             max_scores.append(best_entailment)
    #         else:
    #             # 문맥 확장 (granularity 기반)
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 best_idx
    #             )

    #             expanded_combined_scores = []
    #             expanded_entailments = []

    #             for gran_idx_list in expanded_gran_idx_list:
    #                 expanded_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()

    #                 rouge_score = self.rouge_scorer.score(decomposed_sentence, expanded_context)['rougeL'].fmeasure
    #                 P, R, F1 = self.bert_scorer.score([expanded_context], [decomposed_sentence])
    #                 bert_score = F1.item()

    #                 combined = entailment + self.weight_rouge * rouge_score + self.weight_bert * bert_score

    #                 expanded_combined_scores.append(combined)
    #                 expanded_entailments.append(entailment)

    #             if expanded_combined_scores:
    #                 best_expansion_idx = int(np.argmax(expanded_combined_scores))
    #                 best_expansion_entailment = expanded_entailments[best_expansion_idx]
    #                 max_scores.append(max(best_entailment, best_expansion_entailment))
    #             else:
    #                 max_scores.append(best_entailment)

    #     return min(max_scores)
    


    # e+contradiction 구하는 코드 인데 if조건 조금 바꾼 코드
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         combined_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 복합 점수 계산 (entailment + ROUGE + BERT)
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()

    #             #rouge_score = self.rouge_scorer.score(decomposed_sentence, original_sentence)['rougeL'].fmeasure
    #             #P, R, F1 = self.bert_scorer.score([original_sentence], [decomposed_sentence])
    #             #bert_score = F1.item()

    #             #combined = entailment + self.weight_rouge * rouge_score + self.weight_bert * bert_score
    #             combined = entailment + contradiction

    #             combined_scores.append(combined)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         # 2. 복합 점수 기준 최고 선택
    #         best_idx = int(np.argmax(combined_scores))
    #         best_entailment = entailment_scores[best_idx]
    #         best_contradiction = contradiction_scores[best_idx]
    #         best_neutral = neutral_scores[best_idx]

    #         # 3. dominant entailment 판단
    #         if best_entailment > best_neutral or best_contradiction > best_neutral:
    #             max_scores.append(best_entailment)
    #         else:
    #             # 문맥 확장 (granularity 기반)
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 best_idx
    #             )

    #             expanded_combined_scores = []
    #             expanded_entailments = []

    #             for gran_idx_list in expanded_gran_idx_list:
    #                 expanded_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()

                    
    #                 #rouge_score = self.rouge_scorer.score(decomposed_sentence, expanded_context)['rougeL'].fmeasure
    #                 #P, R, F1 = self.bert_scorer.score([expanded_context], [decomposed_sentence])
    #                 #bert_score = F1.item()

    #                 #combined = entailment + self.weight_rouge * rouge_score + self.weight_bert * bert_score
    #                 combined = entailment + contradiction

    #                 expanded_combined_scores.append(combined)
    #                 expanded_entailments.append(entailment)

    #             if expanded_combined_scores:
    #                 best_expansion_idx = int(np.argmax(expanded_combined_scores))
    #                 best_expansion_entailment = expanded_entailments[best_expansion_idx]
    #                 max_scores.append(max(best_entailment, best_expansion_entailment))
    #             else:
    #                 max_scores.append(best_entailment)

    #     return min(max_scores)




    # 로그찍는 코드

    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     log_path = "./log/xsumfaith_ecscore.log"
    #     os.makedirs(os.path.dirname(log_path), exist_ok=True)

    #     # 파일 존재 여부 확인 후 모드 결정
    #     mode = "a" if os.path.exists(log_path) else "w"

    #     with open(log_path, mode, encoding="utf-8") as log_file:
    #         if mode == "w":
    #             log_file.write("===== Entailment+Contradiction Selection Log =====\n\n")

    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for idx, decomposed_sentence in enumerate(decomposed_sentences_list):
    #         combined_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         with open(log_path, "a", encoding="utf-8") as log_file:
    #             log_file.write(f"[Decomposed Sentence {idx+1}] {decomposed_sentence.strip()}\n")

    #         for original_idx, original_sentence in enumerate(original_sentences_list):
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()

    #             combined = entailment - contradiction

    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  [Original {original_idx+1}] {original_sentence.strip()}\n")
    #                 log_file.write(f"    Entailment: {entailment:.4f}, Contradiction: {contradiction:.4f}, Neutral: {neutral:.4f}, Combined: {combined:.4f}\n")

    #             combined_scores.append(combined)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         best_idx = int(np.argmax(combined_scores))
    #         best_entailment = entailment_scores[best_idx]
    #         best_contradiction = contradiction_scores[best_idx]
    #         best_neutral = neutral_scores[best_idx]

    #         #if best_entailment > best_neutral or best_contradiction > best_neutral:
    #         if best_entailment > best_contradiction and best_entailment > best_neutral:
    #             selected_entailment = best_entailment
    #             selected_combined = combined_scores[best_idx]
    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  → Selected (No Expansion): Original {best_idx+1}, Combined: {selected_combined:.4f}, Entailment: {selected_entailment:.4f}\n\n")
    #             max_scores.append(selected_entailment)
    #         else:
    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write("  >> Granularity Expansion Triggered <<\n")

    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 best_idx
    #             )

    #             expanded_combined_scores = []
    #             expanded_entailments = []

    #             for combo_idx, gran_idx_list in enumerate(expanded_gran_idx_list):
    #                 expanded_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()

    #                     combined = entailment - contradiction

    #                 with open(log_path, "a", encoding="utf-8") as log_file:
    #                     log_file.write(f"    [Expanded {combo_idx+1}] {expanded_context.strip()}\n")
    #                     log_file.write(f"      Entailment: {entailment:.4f}, Contradiction: {contradiction:.4f}, Neutral: {neutral:.4f}, Combined: {combined:.4f}\n")

    #                 expanded_combined_scores.append(combined)
    #                 expanded_entailments.append(entailment)

    #             if expanded_combined_scores:
    #                 best_expansion_idx = int(np.argmax(expanded_combined_scores))
    #                 best_expansion_entailment = expanded_entailments[best_expansion_idx]
    #                 best_expansion_combined = expanded_combined_scores[best_expansion_idx]

    #                 final_entailment = max(best_entailment, best_expansion_entailment)
    #                 final_combined = max(combined_scores[best_idx], best_expansion_combined)
    #             else:
    #                 final_entailment = best_entailment
    #                 final_combined = combined_scores[best_idx]

    #             max_scores.append(final_entailment)

    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  → Selected (With Expansion): Combined: {final_combined:.4f}, Entailment: {final_entailment:.4f}\n\n")

    #     return min(max_scores)




    # #expansion 후 마지막에 e < c or e < n이면 e+c점수로 선택하는 전략
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     log_path = "./log/xsumfaith_ecscore.log"
    #     os.makedirs(os.path.dirname(log_path), exist_ok=True)

    #     # 파일 존재 여부 확인 후 모드 결정
    #     mode = "a" if os.path.exists(log_path) else "w"

    #     with open(log_path, mode, encoding="utf-8") as log_file:
    #         if mode == "w":
    #             log_file.write("===== Entailment+Contradiction Selection Log =====\n\n")

    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for idx, decomposed_sentence in enumerate(decomposed_sentences_list):
    #         combined_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         with open(log_path, "a", encoding="utf-8") as log_file:
    #             log_file.write(f"[Decomposed Sentence {idx+1}] {decomposed_sentence.strip()}\n")

    #         for original_idx, original_sentence in enumerate(original_sentences_list):
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()

    #             combined = entailment #- contradiction

    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  [Original {original_idx+1}] {original_sentence.strip()}\n")
    #                 log_file.write(f"    Entailment: {entailment:.4f}, Contradiction: {contradiction:.4f}, Neutral: {neutral:.4f}, Combined: {combined:.4f}\n")

    #             combined_scores.append(combined)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         best_idx = int(np.argmax(combined_scores))
    #         best_entailment = entailment_scores[best_idx]
    #         best_contradiction = contradiction_scores[best_idx]
    #         best_neutral = neutral_scores[best_idx]

    #         if best_entailment > best_contradiction and best_entailment > best_neutral:
    #             selected_entailment = best_entailment
    #             selected_combined = combined_scores[best_idx]
    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  → Selected (No Expansion): Original {best_idx+1}, Combined: {selected_combined:.4f}, Entailment: {selected_entailment:.4f}\n\n")
    #             max_scores.append(selected_entailment)
    #         else:
    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write("  >> Granularity Expansion Triggered <<\n")

    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 best_idx
    #             )

    #             expanded_combined_scores = []
    #             expanded_entailments = []
    #             expanded_contradictions = []
    #             expanded_neutrals = []
    #             expanded_second_combined = []

    #             for combo_idx, gran_idx_list in enumerate(expanded_gran_idx_list):
    #                 expanded_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()

    #                     combined = entailment #- contradiction
    #                     second_combined = contradiction #+ entailment

    #                 with open(log_path, "a", encoding="utf-8") as log_file:
    #                     log_file.write(f"    [Expanded {combo_idx+1}] {expanded_context.strip()}\n")
    #                     log_file.write(f"      Entailment: {entailment:.4f}, Contradiction: {contradiction:.4f}, Neutral: {neutral:.4f}, Combined: {combined:.4f}, SecondCombined: {second_combined:.4f}\n")

    #                 expanded_combined_scores.append(combined)
    #                 expanded_entailments.append(entailment)
    #                 expanded_contradictions.append(contradiction)
    #                 expanded_neutrals.append(neutral)
    #                 expanded_second_combined.append(second_combined)

    #             if expanded_combined_scores:
    #                 best_expansion_idx = int(np.argmax(expanded_combined_scores))
    #                 best_expansion_entailment = expanded_entailments[best_expansion_idx]
    #                 best_expansion_contradiction = expanded_contradictions[best_expansion_idx]
    #                 best_expansion_neutral = expanded_neutrals[best_expansion_idx]
    #                 best_second_combined = expanded_second_combined[best_expansion_idx]

    #                 if best_expansion_entailment > best_expansion_contradiction and best_expansion_entailment > best_expansion_neutral:
    #                     final_entailment = best_expansion_entailment
    #                     final_combined = expanded_combined_scores[best_expansion_idx]
    #                 else:
    #                     # ✅ 수정: second_combined에서 가장 큰 인덱스로 entailment 선택
    #                     best_second_idx = int(np.argmax(expanded_second_combined))
    #                     final_entailment = expanded_entailments[best_second_idx]
    #                     final_combined = expanded_second_combined[best_second_idx]
    #             else:
    #                 final_entailment = best_entailment
    #                 final_combined = combined_scores[best_idx]


    #             max_scores.append(final_entailment)

    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  → Selected (With Expansion): Combined: {final_combined:.4f}, Entailment: {final_entailment:.4f}\n\n")

    #     return min(max_scores)








    # confidence기반임. (e+c) *(e-c) # 이거 괜춘한데.... 

    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     log_path = "./log/xsumfaith_ecscore.log"
    #     os.makedirs(os.path.dirname(log_path), exist_ok=True)

    #     # 파일 존재 여부 확인 후 모드 결정
    #     mode = "a" if os.path.exists(log_path) else "w"

    #     with open(log_path, mode, encoding="utf-8") as log_file:
    #         if mode == "w":
    #             log_file.write("===== Entailment+Contradiction Selection Log =====\n\n")

    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for idx, decomposed_sentence in enumerate(decomposed_sentences_list):
    #         confidence_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         with open(log_path, "a", encoding="utf-8") as log_file:
    #             log_file.write(f"[Decomposed Sentence {idx+1}] {decomposed_sentence.strip()}\n")

    #         for original_idx, original_sentence in enumerate(original_sentences_list):
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()

    #             confidence = entailment + contradiction

    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  [Original {original_idx+1}] {original_sentence.strip()}\n")
    #                 log_file.write(f"    Entailment: {entailment:.4f}, Contradiction: {contradiction:.4f}, Neutral: {neutral:.4f}, confidence: {confidence:.4f}\n")

    #             confidence_scores.append(confidence)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         best_idx = int(np.argmax(entailment_scores))
    #         best_entailment = entailment_scores[best_idx]
    #         best_contradiction = contradiction_scores[best_idx]
    #         best_neutral = neutral_scores[best_idx]

    #         #if best_entailment > best_neutral or best_contradiction > best_neutral:
    #         if best_entailment > best_contradiction and best_entailment > best_neutral:
    #             selected_entailment_contradiction = best_entailment - best_contradiction
    #             selected_confidence = confidence_scores[best_idx]
    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  → Selected (No Expansion): Original {best_idx+1}, Confidence: {selected_confidence:.4f}, Entailment_Contradiction: {selected_entailment_contradiction:.4f}\n\n")
    #             max_scores.append(selected_entailment_contradiction * selected_confidence)
    #         else:
    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write("  >> Granularity Expansion Triggered <<\n")

    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 best_idx
    #             )

    #             expanded_confidence_scores = []
    #             expanded_entailments = []
    #             expanded_contradictions = []

    #             for combo_idx, gran_idx_list in enumerate(expanded_gran_idx_list):
    #                 expanded_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()

    #                     confidence = entailment + contradiction

    #                 with open(log_path, "a", encoding="utf-8") as log_file:
    #                     log_file.write(f"    [Expanded {combo_idx+1}] {expanded_context.strip()}\n")
    #                     log_file.write(f"      Entailment: {entailment:.4f}, Contradiction: {contradiction:.4f}, Neutral: {neutral:.4f}, Combined: {confidence:.4f}\n")

    #                 expanded_confidence_scores.append(confidence)
    #                 expanded_entailments.append(entailment)
    #                 expanded_contradictions.append(contradiction)

    #             if expanded_confidence_scores:
    #                 best_expansion_idx = int(np.argmax(expanded_entailments))


    #                 if best_entailment > expanded_entailments[best_expansion_idx]:
    #                     final_entailment_contradiction = best_entailment - best_contradiction
    #                     final_confidence = confidence_scores[best_idx]
    #                 else:
    #                     final_entailment_contradiction = expanded_entailments[best_expansion_idx] - expanded_contradictions[best_expansion_idx]
    #                     final_confidence = expanded_confidence_scores[best_expansion_idx]
                    
    #             else:
    #                 final_entailment_contradiction = best_entailment - best_contradiction
    #                 final_confidence = confidence_scores[best_idx]

    #             max_scores.append(final_entailment_contradiction * final_confidence)

    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  → Selected (With Expansion): Confidence: {final_confidence:.4f}, Entailment_Contradiction: {final_entailment_contradiction:.4f}\n\n")

    #     return np.mean(max_scores)
    


    # 초기 ecscore인데 np.mean적용 : 성능 개구림

    
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment + contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment + contradiction
    #                     expanded_scores.append((ecn_score, entailment))

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment)
    #             else:
    #                 max_scores.append(best_entailment_from_expansion)

    #     return np.mean(max_scores)








    # e로 ranking + fenice scoring (e-c) , np.min, np.mean 적용 짜증나게 np.mean한거 성능 오름.... 하............

    
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment #+ contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment #+ contradiction
    #                     expanded_scores.append((ecn_score, entailment - contradiction)) # 여기 조심해라!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_contradiction_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment - max_contradiction)
    #             else:
    #                 max_scores.append(best_entailment_contradiction_from_expansion)

    #     return np.mean(max_scores)








    # 초기 np.mean만 적용

    
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment #+ contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment #+ contradiction
    #                     expanded_scores.append((ecn_score, entailment))

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment)
    #             else:
    #                 max_scores.append(best_entailment_from_expansion)

    #     return np.mean(max_scores)



# confidence기반임. (bertscore + rougescore) *(e-c) # 이거... 구림.

    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         entailment_scores = []
    #         contradiction_scores = []
    #         confidences = []

    #         for original_sentence in original_sentences_list:
    #             # NLI
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()

    #             # ROUGE + BERTScore
    #             rouge_score = self.rouge_scorer.score(decomposed_sentence, original_sentence)['rougeL'].fmeasure
    #             P, R, F1 = self.bert_scorer.score([original_sentence], [decomposed_sentence])
    #             bert_score = F1.item()
    #             confidence = 1 + self.weight_rouge * rouge_score + self.weight_bert * bert_score 


    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             confidences.append(confidence)

    #         best_idx = int(np.argmax(entailment_scores))
    #         best_entailment = entailment_scores[best_idx]
    #         best_contradiction = contradiction_scores[best_idx]
    #         best_confidence = confidences[best_idx]

    #         if best_entailment > best_contradiction:
    #             core_score = best_entailment - best_contradiction
    #             max_scores.append(core_score * best_confidence)
    #         else:
    #             # 문맥 확장
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 best_idx
    #             )

    #             expanded_entailments = []
    #             expanded_contradictions = []
    #             expanded_confidences = []

    #             for gran_idx_list in expanded_gran_idx_list:
    #                 expanded_context = " ".join(original_sentences_list[i] for i in gran_idx_list)

    #                 # NLI
    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()

    #                 # BERT + ROUGE
    #                 rouge_score = self.rouge_scorer.score(decomposed_sentence, expanded_context)['rougeL'].fmeasure
    #                 P, R, F1 = self.bert_scorer.score([expanded_context], [decomposed_sentence])
    #                 bert_score = F1.item()
    #                 confidence = 1 + self.weight_rouge * rouge_score + self.weight_bert * bert_score 


    #                 expanded_entailments.append(entailment)
    #                 expanded_contradictions.append(contradiction)
    #                 expanded_confidences.append(confidence)

    #             if expanded_confidences:
    #                 best_exp_idx = int(np.argmax(expanded_entailments))
    #                 best_exp_entailment = expanded_entailments[best_exp_idx]
    #                 best_exp_contradiction = expanded_contradictions[best_exp_idx]
    #                 best_exp_confidence = expanded_confidences[best_exp_idx]

    #                 if best_exp_entailment > best_exp_contradiction:
    #                     core_score = best_exp_entailment - best_exp_contradiction
    #                     max_scores.append(core_score * best_exp_confidence)
    #                 else:
    #                     fallback_idx = int(np.argmax([
    #                         e - c for e, c in zip(expanded_entailments, expanded_contradictions)
    #                     ]))
    #                     core_score = expanded_entailments[fallback_idx] - expanded_contradictions[fallback_idx]
    #                     max_scores.append(core_score * expanded_confidences[fallback_idx])
    #             else:
    #                 core_score = best_entailment - best_contradiction
    #                 max_scores.append(core_score * best_confidence)

    #     return np.mean(max_scores)




    # entailment + rouge + bertscore 로 ranking,  fenice scoring, np.mean
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         combined_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 복합 점수 계산 (ROUGE + BERT)
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()

    #             rouge_score = self.rouge_scorer.score(decomposed_sentence, original_sentence)['rougeL'].fmeasure
    #             P, R, F1 = self.bert_scorer.score([original_sentence], [decomposed_sentence])
    #             bert_score = F1.item()

    #             combined = self.weight_rouge * rouge_score + self.weight_bert * bert_score #+entailment

    #             combined_scores.append(combined)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         # 2. 복합 점수 기준 최고 선택
    #         best_idx = int(np.argmax(combined_scores))
    #         best_entailment = entailment_scores[best_idx]
    #         best_contradiction = contradiction_scores[best_idx]
    #         best_neutral = neutral_scores[best_idx]

    #         # 3. dominant entailment 판단
    #         if best_entailment > best_contradiction and best_entailment > best_neutral:
    #             max_scores.append(best_entailment - best_contradiction)
    #         else:
    #             # 문맥 확장 (granularity 기반)
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 best_idx
    #             )

    #             expanded_combined_scores = []
    #             expanded_entailments_contradictions = []

    #             for gran_idx_list in expanded_gran_idx_list:
    #                 expanded_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()

    #                 rouge_score = self.rouge_scorer.score(decomposed_sentence, expanded_context)['rougeL'].fmeasure
    #                 P, R, F1 = self.bert_scorer.score([expanded_context], [decomposed_sentence])
    #                 bert_score = F1.item()

    #                 combined = self.weight_rouge * rouge_score + self.weight_bert * bert_score #+entailment

    #                 expanded_combined_scores.append(combined)
    #                 expanded_entailments_contradictions.append(entailment - contradiction)

    #             if expanded_combined_scores:
    #                 best_expansion_idx = int(np.argmax(expanded_combined_scores))
    #                 best_expansion_entailment_contradiction = expanded_entailments_contradictions[best_expansion_idx]
    #                 max_scores.append(max(best_entailment - best_contradiction, best_expansion_entailment_contradiction))
    #             else:
    #                 max_scores.append(best_entailment - best_contradiction)

    #     return np.mean(max_scores)




# e로 ranking + fenice scoring (e-c) , minus softmin

    
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment #+ contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment #+ contradiction
    #                     expanded_scores.append((ecn_score, entailment - contradiction)) 

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_contradiction_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment - max_contradiction)
    #             else:
    #                 max_scores.append(best_entailment_contradiction_from_expansion)

    #     return np.mean([-math.pow(16, -s) for s in max_scores])






    # e로 ranking + confidence KL divergence (e-c) , np.mean 

    # def kl_confidence(self,entailment, contradiction, neutral, eps=1e-8):
    #     # 모델 출력 확률분포 (P)
    #     P = np.array([entailment, contradiction, neutral]) + eps
    #     P /= P.sum()  # 정규화

    #     # 이상적인 분포 (Q): 전적으로 entailment에만 집중
    #     Q = np.array([1.0, 0.0, 0.0]) + eps
    #     Q /= Q.sum()

    #     # KL(Q || P)
    #     kl_div = np.sum(Q * np.log(Q / P))

    #     # confidence는 KL divergence의 음수 (작을수록 좋다 → 클수록 confidence 높음)
    #     confidence = np.exp(-kl_div)
    #     return confidence

    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     log_path = "./log/xsumfaith_ecscore.log"
    #     os.makedirs(os.path.dirname(log_path), exist_ok=True)

    #     # 파일 존재 여부 확인 후 모드 결정
    #     mode = "a" if os.path.exists(log_path) else "w"

    #     with open(log_path, mode, encoding="utf-8") as log_file:
    #         if mode == "w":
    #             log_file.write("===== Entailment+Contradiction Selection Log =====\n\n")

    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for idx, decomposed_sentence in enumerate(decomposed_sentences_list):
    #         confidence_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         with open(log_path, "a", encoding="utf-8") as log_file:
    #             log_file.write(f"[Decomposed Sentence {idx+1}] {decomposed_sentence.strip()}\n")

    #         for original_idx, original_sentence in enumerate(original_sentences_list):
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()

    #             confidence = self.kl_confidence(entailment, contradiction, neutral)

    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  [Original {original_idx+1}] {original_sentence.strip()}\n")
    #                 log_file.write(f"    Entailment: {entailment:.4f}, Contradiction: {contradiction:.4f}, Neutral: {neutral:.4f}, confidence: {confidence:.4f}\n")

    #             confidence_scores.append(confidence)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         best_idx = int(np.argmax(entailment_scores))
    #         best_entailment = entailment_scores[best_idx]
    #         best_contradiction = contradiction_scores[best_idx]
    #         best_neutral = neutral_scores[best_idx]

    #         #if best_entailment > best_neutral or best_contradiction > best_neutral:
    #         if best_entailment > best_contradiction and best_entailment > best_neutral:
    #             selected_entailment_contradiction = best_entailment - best_contradiction
    #             selected_confidence = confidence_scores[best_idx]
    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  → Selected (No Expansion): Original {best_idx+1}, Confidence: {selected_confidence:.4f}, Entailment_Contradiction: {selected_entailment_contradiction:.4f}\n\n")
    #             max_scores.append(selected_entailment_contradiction * selected_confidence)
    #         else:
    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write("  >> Granularity Expansion Triggered <<\n")

    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 best_idx
    #             )

    #             expanded_confidence_scores = []
    #             expanded_entailments = []
    #             expanded_contradictions = []

    #             for combo_idx, gran_idx_list in enumerate(expanded_gran_idx_list):
    #                 expanded_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()

    #                     confidence = self.kl_confidence(entailment, contradiction, neutral)

    #                 with open(log_path, "a", encoding="utf-8") as log_file:
    #                     log_file.write(f"    [Expanded {combo_idx+1}] {expanded_context.strip()}\n")
    #                     log_file.write(f"      Entailment: {entailment:.4f}, Contradiction: {contradiction:.4f}, Neutral: {neutral:.4f}, Combined: {confidence:.4f}\n")

    #                 expanded_confidence_scores.append(confidence)
    #                 expanded_entailments.append(entailment)
    #                 expanded_contradictions.append(contradiction)

    #             if expanded_confidence_scores:
    #                 best_expansion_idx = int(np.argmax(expanded_entailments))


    #                 if best_entailment > expanded_entailments[best_expansion_idx]:
    #                     final_entailment_contradiction = best_entailment - best_contradiction
    #                     final_confidence = confidence_scores[best_idx]
    #                 else:
    #                     final_entailment_contradiction = expanded_entailments[best_expansion_idx] - expanded_contradictions[best_expansion_idx]
    #                     final_confidence = expanded_confidence_scores[best_expansion_idx]
                    
    #             else:
    #                 final_entailment_contradiction = best_entailment - best_contradiction
    #                 final_confidence = confidence_scores[best_idx]

    #             max_scores.append(final_entailment_contradiction * final_confidence)

    #             with open(log_path, "a", encoding="utf-8") as log_file:
    #                 log_file.write(f"  → Selected (With Expansion): Confidence: {final_confidence:.4f}, Entailment_Contradiction: {final_entailment_contradiction:.4f}\n\n")

    #     return np.mean(max_scores)












    # e로 ranking + infuse expansion + fenice scoring (e-c) , np.mean 
    
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 ecn_score = entailment  # or entailment + contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # dominant 판단
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             # ecn_ranking 기준 정렬된 인덱스
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #             concat_sentences = []
    #             prev_neutral = None
    #             selected_entailment = None
    #             selected_contradiction = None

    #             for i, idx in enumerate(ranked_indices):
    #                 concat_sentences.append(original_sentences_list[idx])
    #                 expanded_context = " ".join(concat_sentences)

    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()

    #                 if i == 0:
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction
    #                 else:
    #                     if neutral > prev_neutral:
    #                         break
    #                     else:
    #                         prev_neutral = neutral
    #                         selected_entailment = entailment
    #                         selected_contradiction = contradiction

    #             max_scores.append(selected_entailment - selected_contradiction)

    #     return np.mean(max_scores)



    # # ranking + infuse expansion(종료조건 엔트로피) + fenice scoring (e-c) , np.mean 
    # def atomic_facts_scoring_with_entropy(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_probs = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().tolist()
    #                 entailment_probs.append(probs)
    #                 ecn_ranking_scores.append(probs[0])  # entailment

    #         ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #         concat_sentences = []
    #         selected_entailment = None
    #         selected_contradiction = None
    #         prev_entropy = None

    #         for i, idx in enumerate(ranked_indices):
    #             concat_sentences.append(original_sentences_list[idx])
    #             expanded_context = " ".join(concat_sentences)

    #             features = self.tokenizer(
    #                 [expanded_context.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().tolist()
    #                 entropy = -sum([x * math.log(x + 1e-10) for x in probs])
    #                 entailment = probs[0]
    #                 contradiction = probs[1]

    #             if i == 0:
    #                 prev_entropy = entropy
    #                 selected_entailment = entailment
    #                 selected_contradiction = contradiction
    #             else:
    #                 if entropy > prev_entropy:
    #                     break
    #                 else:
    #                     prev_entropy = entropy
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction

    #         max_scores.append(selected_entailment - selected_contradiction)

    #     return np.mean(max_scores)


    # e ranking + infuse expansion(종료조건 bert+rouge) + fenice scoring (e-c) , np.mean 
    # def atomic_facts_scoring_with_similarity(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_probs = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().tolist()
    #                 entailment_probs.append(probs)
    #                 ecn_ranking_scores.append(probs[0])  # entailment

    #         ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #         concat_sentences = []
    #         selected_entailment = None
    #         selected_contradiction = None
    #         prev_similarity = None

    #         for i, idx in enumerate(ranked_indices):
    #             concat_sentences.append(original_sentences_list[idx])
    #             expanded_context = " ".join(concat_sentences)

    #             features = self.tokenizer(
    #                 [expanded_context.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1).squeeze().cpu().tolist()
    #                 entailment = probs[0]
    #                 contradiction = probs[1]

    #             rouge_score = self.rouge_scorer.score(decomposed_sentence, expanded_context)['rougeL'].fmeasure
    #             _, _, F1 = self.bert_scorer.score([expanded_context], [decomposed_sentence])
    #             bert_score = F1.item()
    #             similarity = self.weight_rouge * rouge_score + self.weight_bert * bert_score

    #             if i == 0:
    #                 prev_similarity = similarity
    #                 selected_entailment = entailment
    #                 selected_contradiction = contradiction
    #             else:
    #                 if similarity < prev_similarity:
    #                     break
    #                 else:
    #                     prev_similarity = similarity
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction

    #         max_scores.append(selected_entailment - selected_contradiction)

    #     return np.mean(max_scores)


# ranking + fizz/infuse expansion + fenice scoring (e-c) , np.mean 이거 모든 측면에서 best는 아니지만 이걸 내 sota로 설정정

    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # Step 1: NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # Step 2: dominant 판단
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ### 방법 1: gran 기반 context 확장 ###
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             gran_expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].item()
    #                     contradiction = probs[0][1].item()
                    
    #                 gran_expanded_scores.append(entailment - contradiction)

    #             best_gran_score = max(gran_expanded_scores) if gran_expanded_scores else -1

    #             ### 방법 2: ecn 내림차순 정렬 후 progressive 확장 (neutral 증가 시 종료) ###
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
    #             concat_sentences = []
    #             prev_neutral = None
    #             selected_entailment = None
    #             selected_contradiction = None

    #             for i, idx in enumerate(ranked_indices):
    #                 concat_sentences.append(original_sentences_list[idx])
    #                 expanded_context = " ".join(concat_sentences)

    #                 features = self.tokenizer(
    #                     [expanded_context.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].item()
    #                     contradiction = probs[0][1].item()
    #                     neutral = probs[0][2].item()

    #                 if i == 0:
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction
    #                 else:
    #                     if neutral > prev_neutral:
    #                         break
    #                     prev_neutral = neutral
    #                     selected_entailment = entailment
    #                     selected_contradiction = contradiction

    #             ranked_expansion_score = selected_entailment - selected_contradiction

    #             ### 최종 선택: 두 확장 방식 중 더 좋은 score ###
    #             final_score = max(best_gran_score, ranked_expansion_score)
    #             max_scores.append(final_score)

    #     return np.mean(max_scores)






# # ranking + fizz/infuse expansion + fenice scoring (e-c) , np.mean 이거 모든 측면에서 best는 아니지만 이걸 내 sota로 설정정

#     def atomic_facts_scoring(self, original_text, decomposed_text):
#         self.load_lm()
#         original_sentences_list = self.split_sentences(original_text)
#         decomposed_sentences_list = self.split_sentences(decomposed_text)

#         max_scores = []

#         for decomposed_sentence in decomposed_sentences_list:
#             ecn_ranking_scores = []
#             entailment_scores = []
#             contradiction_scores = []
#             neutral_scores = []

#             # Step 1: NLI 계산
#             for original_sentence in original_sentences_list:
#                 features = self.tokenizer(
#                     [original_sentence.strip()],
#                     [decomposed_sentence.strip()],
#                     padding=True,
#                     truncation=True,
#                     return_tensors="pt"
#                 ).to(self.device)

#                 self.model.eval()
#                 with torch.no_grad():
#                     logits = self.model(**features).logits
#                     probs = torch.nn.functional.softmax(logits, dim=-1)
#                     entailment = probs[0][0].item()
#                     contradiction = probs[0][1].item()
#                     neutral = probs[0][2].item()
#                     ecn_score = entailment

#                     ecn_ranking_scores.append(ecn_score)
#                     entailment_scores.append(entailment)
#                     contradiction_scores.append(contradiction)
#                     neutral_scores.append(neutral)

#             # Step 2: dominant 판단
#             max_ecn_idx = int(np.argmax(ecn_ranking_scores))
#             max_entailment = entailment_scores[max_ecn_idx]
#             max_contradiction = contradiction_scores[max_ecn_idx]
#             max_neutral = neutral_scores[max_ecn_idx]

#             if max_entailment > max_contradiction and max_entailment > max_neutral:
#                 max_scores.append(max_entailment - max_contradiction)
#             else:
#                 ### 방법 1: gran 기반 context 확장 ###
#                 expanded_gran_idx_list = get_combinations(
#                     list(range(len(original_sentences_list))),
#                     self.gran,
#                     max_ecn_idx
#                 )

#                 gran_expanded_scores = []
#                 for gran_idx_list in expanded_gran_idx_list:
#                     new_context = " ".join(original_sentences_list[i] for i in gran_idx_list)
#                     features = self.tokenizer(
#                         [new_context.strip()],
#                         [decomposed_sentence.strip()],
#                         padding=True,
#                         truncation=True,
#                         return_tensors="pt"
#                     ).to(self.device)

#                     with torch.no_grad():
#                         logits = self.model(**features).logits
#                         probs = torch.nn.functional.softmax(logits, dim=-1)
#                         entailment = probs[0][0].item()
#                         contradiction = probs[0][1].item()
                    
#                     gran_expanded_scores.append(entailment - contradiction)

#                 best_gran_score = max(gran_expanded_scores) if gran_expanded_scores else -1

#                 ### 방법 2: ecn 내림차순 정렬 후 progressive 확장 (neutral 증가 시 종료) ###
#                 ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)
#                 concat_sentences = []
#                 prev_neutral = None
#                 selected_entailment = None
#                 selected_contradiction = None

#                 for i, idx in enumerate(ranked_indices):
#                     concat_sentences.append(original_sentences_list[idx])
#                     expanded_context = " ".join(concat_sentences)

#                     features = self.tokenizer(
#                         [expanded_context.strip()],
#                         [decomposed_sentence.strip()],
#                         padding=True,
#                         truncation=True,
#                         return_tensors="pt"
#                     ).to(self.device)

#                     with torch.no_grad():
#                         logits = self.model(**features).logits
#                         probs = torch.nn.functional.softmax(logits, dim=-1)
#                         entailment = probs[0][0].item()
#                         contradiction = probs[0][1].item()
#                         neutral = probs[0][2].item()

#                     if i == 0:
#                         prev_neutral = neutral
#                         selected_entailment = entailment
#                         selected_contradiction = contradiction
#                     else:
#                         if neutral > prev_neutral:
#                             break
#                         prev_neutral = neutral
#                         selected_entailment = entailment
#                         selected_contradiction = contradiction

#                 ranked_expansion_score = selected_entailment - selected_contradiction

#                 ### 최종 선택: 두 확장 방식 중 더 좋은 score ###
#                 final_score = max(best_gran_score, ranked_expansion_score)
#                 max_scores.append(final_score)

#         return np.mean(max_scores)



# e ranking dawon fifuse expansion ,e-c scoring, 낫베드이고 노벨티도 있음. 중요한거는 ablation study인데...
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ranked_indices = sorted(range(len(entailment_scores)), key=lambda i: entailment_scores[i], reverse=True)

    #             best_combo_score = -float('inf')
    #             best_combo = []
    #             prev_neutral = None

    #             for i, idx in enumerate(ranked_indices):
    #                 gran_combos = get_combinations(list(range(len(original_sentences_list))), self.gran, idx)

    #                 best_gran_score = -float('inf')
    #                 best_gran_combo = []
    #                 best_gran_neutral = None

    #                 for combo in gran_combos:
    #                     context = " ".join(original_sentences_list[j] for j in combo)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()
    #                         neutral = probs[0][2].item()

    #                     score = entailment - contradiction
    #                     if score > best_gran_score:
    #                         best_gran_score = score
    #                         best_gran_combo = combo
    #                         best_gran_neutral = neutral

    #                 if i == 0:
    #                     prev_neutral = best_gran_neutral
    #                     best_combo = best_gran_combo
    #                     best_combo_score = best_gran_score
    #                 else:
    #                     if best_gran_neutral > prev_neutral:
    #                         break
    #                     combined = sorted(set(best_combo + best_gran_combo))
    #                     context = " ".join(original_sentences_list[j] for j in combined)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()

    #                     score = entailment - contradiction
    #                     if score > best_combo_score:
    #                         best_combo = combined
    #                         best_combo_score = score
    #                     prev_neutral = best_gran_neutral

    #             max_scores.append(best_combo_score)

    #     return np.mean(max_scores)



# e ranking dawon fifuse expansion ,e-c scoring, 노벨티 좀더 추가한 버전. expansion phase원래 optional이었는데 madatory하게 바꿈
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral and False:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ranked_indices = sorted(range(len(entailment_scores)), key=lambda i: entailment_scores[i], reverse=True)

    #             best_combo_score = -float('inf')
    #             best_combo = []
    #             prev_neutral = None

    #             for i, idx in enumerate(ranked_indices):
    #                 gran_combos = get_combinations(list(range(len(original_sentences_list))), self.gran, idx)

    #                 best_gran_score = -float('inf')
    #                 best_gran_combo = []
    #                 best_gran_neutral = None

    #                 for combo in gran_combos:
    #                     context = " ".join(original_sentences_list[j] for j in combo)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()
    #                         neutral = probs[0][2].item()

    #                     score = entailment - contradiction
    #                     if score > best_gran_score:
    #                         best_gran_score = score
    #                         best_gran_combo = combo
    #                         best_gran_neutral = neutral

    #                 if i == 0:
    #                     prev_neutral = best_gran_neutral
    #                     best_combo = best_gran_combo
    #                     best_combo_score = best_gran_score
    #                 else:
    #                     if best_gran_neutral > prev_neutral:
    #                         break
    #                     combined = sorted(set(best_combo + best_gran_combo))
    #                     context = " ".join(original_sentences_list[j] for j in combined)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()

    #                     score = entailment - contradiction
    #                     if score > best_combo_score:
    #                         best_combo = combined
    #                         best_combo_score = score
    #                     prev_neutral = best_gran_neutral

    #             max_scores.append(best_combo_score)

    #     return np.mean(max_scores)




# e-c ranking dawon fifuse expansion ,e - c scoring, np.mean 이거 ranking e-c로 하면 점수 많이 떨어짐
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()
    #                 ecn_score = entailment - contradiction

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: ecn_ranking_scores[i], reverse=True)

    #             best_combo_score = -float('inf')
    #             best_combo = []
    #             prev_neutral = None

    #             for i, idx in enumerate(ranked_indices):
    #                 gran_combos = get_combinations(list(range(len(original_sentences_list))), self.gran, idx)

    #                 best_gran_score = -float('inf')
    #                 best_gran_combo = []
    #                 best_gran_neutral = None

    #                 for combo in gran_combos:
    #                     context = " ".join(original_sentences_list[j] for j in combo)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()
    #                         neutral = probs[0][2].item()

    #                     score = entailment - contradiction
    #                     if score > best_gran_score:
    #                         best_gran_score = score
    #                         best_gran_combo = combo
    #                         best_gran_neutral = neutral

    #                 if i == 0:
    #                     prev_neutral = best_gran_neutral
    #                     best_combo = best_gran_combo
    #                     best_combo_score = best_gran_score
    #                 else:
    #                     if best_gran_neutral > prev_neutral:
    #                         break
    #                     combined = sorted(set(best_combo + best_gran_combo))
    #                     context = " ".join(original_sentences_list[j] for j in combined)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()

    #                     score = entailment - contradiction
    #                     if score > best_combo_score:
    #                         best_combo = combined
    #                         best_combo_score = score
    #                     prev_neutral = best_gran_neutral

    #             max_scores.append(best_combo_score)

    #     return np.mean(max_scores)
    


# e ranking dawon fifuse expansion(rouge, bertscore) ,e-c scoring, np.mean
    # def atomic_facts_scoring(self, original_text, decomposed_text):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         rouge_scores = []
    #         bert_scores = []

    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].item()
    #                 contradiction = probs[0][1].item()
    #                 neutral = probs[0][2].item()

    #             # rouge_score = self.rouge_scorer.score(decomposed_sentence, original_sentence)['rougeL'].fmeasure
    #             _, _, F1 = self.bert_scorer.score([original_sentence], [decomposed_sentence])
    #             bert_score = F1.item()

    #             ecn_score = entailment + self.weight_bert * bert_score

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             ranked_indices = sorted(range(len(ecn_ranking_scores)), key=lambda i: entailment_scores[i], reverse=True)
    #             best_combo_score = -float('inf')
    #             best_combo = []
    #             prev_neutral = None

    #             for i, idx in enumerate(ranked_indices):
    #                 gran_combos = get_combinations(list(range(len(original_sentences_list))), self.gran, idx)

    #                 best_gran_score = -float('inf')
    #                 best_gran_combo = []
    #                 best_gran_neutral = None

    #                 for combo in gran_combos:
    #                     context = " ".join(original_sentences_list[j] for j in combo)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()
    #                         neutral = probs[0][2].item()
    #                     # rouge_score = self.rouge_scorer.score(decomposed_sentence, context)['rougeL'].fmeasure
    #                     _, _, F1 = self.bert_scorer.score([context], [decomposed_sentence])
    #                     bert_score = F1.item()
    #                     score = entailment + self.weight_bert * bert_score

    #                     if score > best_gran_score:
    #                         best_gran_score = score
    #                         best_gran_combo = combo
    #                         best_gran_neutral = neutral

    #                 if i == 0:
    #                     prev_neutral = best_gran_neutral
    #                     best_combo = best_gran_combo
    #                     best_combo_score = best_gran_score
    #                 else:
    #                     if best_gran_neutral > prev_neutral:
    #                         break
    #                     combined = sorted(set(best_combo + best_gran_combo))
    #                     context = " ".join(original_sentences_list[j] for j in combined)
    #                     features = self.tokenizer(
    #                         [context.strip()],
    #                         [decomposed_sentence.strip()],
    #                         padding=True,
    #                         truncation=True,
    #                         return_tensors="pt"
    #                     ).to(self.device)

    #                     with torch.no_grad():
    #                         logits = self.model(**features).logits
    #                         probs = torch.nn.functional.softmax(logits, dim=-1)
    #                         entailment = probs[0][0].item()
    #                         contradiction = probs[0][1].item()

    #                     rouge_score = self.rouge_scorer.score(decomposed_sentence, context)['rougeL'].fmeasure
    #                     _, _, F1 = self.bert_scorer.score([context], [decomposed_sentence])
    #                     bert_score = F1.item()
    #                     score = entailment + self.weight_rouge * rouge_score + self.weight_bert * bert_score

    #                     if score > best_combo_score:
    #                         best_combo = combined
    #                         best_combo_score = score
    #                     prev_neutral = best_gran_neutral

    #             max_scores.append(best_combo_score)

    #     return np.mean(max_scores)





# e로 ranking + fenice scoring (e-c) , minus softmin-0.2

    
    # def atomic_facts_scoring(self, original_text, decomposed_text, base, exponent_delta):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment #+ contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment #+ contradiction
    #                     expanded_scores.append((ecn_score, entailment - contradiction)) 

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_contradiction_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment - max_contradiction)
    #             else:
    #                 max_scores.append(best_entailment_contradiction_from_expansion)

    #     transformed_scores = [-math.pow(base, -(s-exponent_delta)) for s in max_scores]
    #     final_score = sum(transformed_scores) / len(transformed_scores)

    #     # 로그 작성
    #     # log_path = "./log/scoring_last.log"
    #     # os.makedirs(os.path.dirname(log_path), exist_ok=True)


    #     # with open(log_path, "a", encoding="utf-8") as f:
    #     #     f.write("[Max Scores]\n")
    #     #     for score in max_scores:
    #     #         f.write(f"{score:.6f}\n")

    #     #     mean_raw_score = np.mean(max_scores)
    #     #     f.write(f"\n[Mean Raw Score]\n{mean_raw_score:.6f}\n")

    #     #     f.write("\n[Transformed Scores]\n")
    #     #     for tscore in transformed_scores:
    #     #         f.write(f"{tscore:.6f}\n")

    #     #     f.write(f"\n[Final Score (Transformed Mean)]\n{final_score:.6f}\n")
    #     #     f.write(f"\n[Original Decomposed Text]\n{decomposed_text}\n")
    #     #     f.write("="*40 + "\n\n")

    #     return final_score






# e로 ranking + fenice scoring (e-c) , interactive minus softmin 쌉구림

    
    # def atomic_facts_scoring(self, original_text, decomposed_text, base, exponent_delta):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 ecn_score = entailment

    #             ecn_ranking_scores.append(ecn_score)
    #             entailment_scores.append(entailment)
    #             contradiction_scores.append(contradiction)
    #             neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     ecn_score = entailment
    #                     expanded_scores.append((ecn_score, entailment - contradiction))

    #             best_ecn_from_expansion, best_entailment_contradiction_from_expansion = max(
    #                 expanded_scores, key=lambda x: x[0])

    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment - max_contradiction)
    #             else:
    #                 max_scores.append(best_entailment_contradiction_from_expansion)

    #     # 지수 함수 적용을 위한 base 조정
    #     num_sentences = len(decomposed_sentences_list)
    #     base_adjusted = base + (num_sentences - 3) * base_flex

    #     transformed_scores = [-math.pow(base_adjusted, -(s - exponent_delta)) for s in max_scores]
    #     final_score = sum(transformed_scores) / len(transformed_scores)

    #     return final_score



# e로 ranking + fenice scoring (e-c) , minus softmin-0.2

    
    # def atomic_facts_scoring(self, original_text, decomposed_text, base, exponent_delta):
    #     self.load_lm()
    #     original_sentences_list = self.split_sentences(original_text)
    #     decomposed_sentences_list = self.split_sentences(decomposed_text)

    #     max_scores = []

    #     for decomposed_sentence in decomposed_sentences_list:
    #         ecn_ranking_scores = []
    #         entailment_scores = []
    #         contradiction_scores = []
    #         neutral_scores = []

    #         # 1. 모든 original 문장에 대해 NLI 계산
    #         for original_sentence in original_sentences_list:
    #             features = self.tokenizer(
    #                 [original_sentence.strip()],
    #                 [decomposed_sentence.strip()],
    #                 padding=True,
    #                 truncation=True,
    #                 return_tensors="pt"
    #             ).to(self.device)

    #             self.model.eval()
    #             with torch.no_grad():
    #                 logits = self.model(**features).logits
    #                 probs = torch.nn.functional.softmax(logits, dim=-1)
    #                 entailment = probs[0][0].cpu().item()
    #                 contradiction = probs[0][1].cpu().item()
    #                 neutral = probs[0][2].cpu().item()
    #                 #ecn_score = entailment + contradiction - neutral
    #                 ecn_score = entailment #+ contradiction

    #                 ecn_ranking_scores.append(ecn_score)
    #                 entailment_scores.append(entailment)
    #                 contradiction_scores.append(contradiction)
    #                 neutral_scores.append(neutral)

    #         # 2. ECN 기준으로 최고 문장 선택 (단, entailment 점수는 따로 보관)
    #         max_ecn_idx = int(np.argmax(ecn_ranking_scores))
    #         max_entailment = entailment_scores[max_ecn_idx]
    #         max_contradiction = contradiction_scores[max_ecn_idx]
    #         max_neutral = neutral_scores[max_ecn_idx]

    #         # 3. dominant 판단: entailment가 가장 클 경우
    #         if max_entailment > max_contradiction and max_entailment > max_neutral:
    #             max_scores.append(max_entailment - max_contradiction)
    #         else:
    #             # 문맥 확장 조합 생성
    #             expanded_gran_idx_list = get_combinations(
    #                 list(range(len(original_sentences_list))),
    #                 self.gran,
    #                 max_ecn_idx
    #             )

    #             expanded_scores = []
    #             for gran_idx_list in expanded_gran_idx_list:
    #                 new_original = " ".join(original_sentences_list[i] for i in gran_idx_list)
    #                 features = self.tokenizer(
    #                     [new_original.strip()],
    #                     [decomposed_sentence.strip()],
    #                     padding=True,
    #                     truncation=True,
    #                     return_tensors="pt"
    #                 ).to(self.device)

    #                 with torch.no_grad():
    #                     logits = self.model(**features).logits
    #                     probs = torch.nn.functional.softmax(logits, dim=-1)
    #                     entailment = probs[0][0].cpu().item()
    #                     contradiction = probs[0][1].cpu().item()
    #                     neutral = probs[0][2].cpu().item()
    #                     #ecn_score = entailment + contradiction - neutral
    #                     ecn_score = entailment #+ contradiction
    #                     expanded_scores.append((ecn_score, entailment - contradiction)) 

    #             # 확장 문맥들 중 최고 ECN
    #             best_ecn_from_expansion, best_entailment_contradiction_from_expansion = max(expanded_scores, key=lambda x: x[0])

    #             # 최종 비교도 ECN 기준
    #             if ecn_ranking_scores[max_ecn_idx] >= best_ecn_from_expansion:
    #                 max_scores.append(max_entailment - max_contradiction)
    #             else:
    #                 max_scores.append(best_entailment_contradiction_from_expansion)

    #     #transformed_scores = [-math.pow(base, -(s-exponent_delta)) for s in max_scores]
    #     #final_score = sum(transformed_scores) / len(transformed_scores)
    #     final_score = 0.5 * np.mean(max_scores) + 0.5 * np.min(max_scores)


    #     # 로그 작성
    #     # log_path = "./log/scoring_last.log"
    #     # os.makedirs(os.path.dirname(log_path), exist_ok=True)


    #     # with open(log_path, "a", encoding="utf-8") as f:
    #     #     f.write("[Max Scores]\n")
    #     #     for score in max_scores:
    #     #         f.write(f"{score:.6f}\n")

    #     #     mean_raw_score = np.mean(max_scores)
    #     #     f.write(f"\n[Mean Raw Score]\n{mean_raw_score:.6f}\n")

    #     #     f.write("\n[Transformed Scores]\n")
    #     #     for tscore in transformed_scores:
    #     #         f.write(f"{tscore:.6f}\n")

    #     #     f.write(f"\n[Final Score (Transformed Mean)]\n{final_score:.6f}\n")
    #     #     f.write(f"\n[Original Decomposed Text]\n{decomposed_text}\n")
    #     #     f.write("="*40 + "\n\n")

    #     return final_score









def is_consecutive_by_one(numbers):
    for i in range(1, len(numbers)):
        if abs(numbers[i] - numbers[i-1]) != 1:
            return False
    return True

def get_combinations(num_list, size, target):
    combination_list = []
    for i in range(1, size):
        combination = combinations(num_list, i)
        comb_list = list(combination)
        combination_list.extend(comb_list)
    
    possible_idx_list = []
    for combination in combination_list:
        idx_list = list(combination)
        if target in idx_list and is_consecutive_by_one(idx_list):
            possible_idx_list.append(idx_list)

    return possible_idx_list

def main():
    scorer = AtomicFactScorer(granularity="4G")
    # original = "lisa courtney, of hertfordshire, has spent most of her life collecting pokemon memorabilia."
    # atomic_facts = "Lisa Courtney is from Hertfordshire. Lisa Courtney has spent most of her life collecting Pokémon memorabilia."
    
    original = "todd phillips, a pit crew member was hit by francesco dracone, a car on sunday during the inaugural indycar grand prix of louisiana . todd phillips , a front - outside tire changer for dayle coyne racing , was injuried when todd phillips was struck by francesco dracone, the car of francesco dracone , who had come in on lap 25 for tires and fuel . francesco dracone, dracone spun while exiting francesco dracone's put box , clipping phillips ' leg . todd phillips, tire changer todd phillips , a front - outside tire changer for dayle coyne racing , was injuried when todd phillips was struck by francesco dracone, the car of francesco dracone , who had come in on lap 25 for tires and fuel phillips was taken to the infield care center for treatment where todd phillips has received stitches for a cut on todd phillips's leg and has been released . many cars struggled with traction during the race after rain had fallen for hours leading up to the green flag . francesco dracone, dracone did not finish the race and wound up 23rd . francesco dracone ( 19 ) , of italy , spins out on turn one during the indycar grand prix of louisiana auto race in avondale on sunday"
    atomic_facts = "Todd Phillips is a tire changer. Todd Phillips was injured. Todd Phillips was struck by a car. The car that struck Todd Phillips was driven by Francesco Dracone. The event where Todd Phillips was struck by the car was the inaugural IndyCar Grand Prix of Louisiana. The date of the event was Sunday. Dracone came in on lap 25. Dracone needed tires and fuel. Dracone spun while exiting the pit box. Todd Phillips was taken to the infield care center. Todd Phillips received stitches for a cut on Todd Phillips's leg. Todd Phillips has been released. "

    score = scorer.atomic_facts_scoring(original, atomic_facts)
    print(score)

if __name__ == "__main__":
    main()