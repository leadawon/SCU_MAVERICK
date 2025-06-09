import re
import nltk
import torch
from nltk import sent_tokenize
from maverick import Maverick
from spacy import load as spacy_load
nltk.download('punkt')

MAX_TOKENS = 512

# penalized_pronouns = set([
#     "i", "he", "she", "you", "me", "him", "myself", "yourself",
#     "himself", "herself", "yourselves", "my", "mine", "her",
#     "hers", "his", "your", "yours", "it", "its", "we", "us",
#     "they", "them", "their", "theirs", "itself", "themselves",
#     "this", "that", "these", "those", "who", "whom", "whose",
#     "which", "what", "anyone", "someone", "everyone", "no one",
#     "anybody", "somebody", "everybody", "nobody", "each other",
#     "one another", "each", "either", "neither", "both", "all",
#     "some", "any", "none", "few", "many", "several", "much", "person", "people"
# ])

def contains_penalized_word(mention_text, penalized_set):
    tokens = mention_text.lower().split()
    return any(tok in penalized_set for tok in tokens)

penalized_pronouns = set([
    "person", "people"
])

def custom_tokenize(text):
    raw_tokens = text.strip().split()
    tokens = []
    for token in raw_tokens:
        split_tokens = re.findall(r"\w+|[^\w\s]", token, re.UNICODE)
        tokens.extend(split_tokens)
    return tokens

def custom_detokenize(tokens):
    text = ""
    open_quotes = {'\"', "``", "‘", "“"}
    close_quotes = {'\"', "''", "’", "”"}

    for i, tok in enumerate(tokens):
        if i == 0:
            text += tok
        elif tok == "'" and i + 1 < len(tokens) and tokens[i + 1].lower() in {"s", "t", "re", "ve", "ll", "d", "m"}:
            text += "'"
        elif tok in {".", ",", "!", "?", ";", ":", ")", "]", "}"}:
            text += tok
        elif tok in {"(", "[", "{"}:
            text += " " + tok
        elif tok in close_quotes:
            text += tok
        elif tok in open_quotes:
            text += " " + tok
        else:
            text += " " + tok
    return text.strip()

def choose_best_mention(cluster, nlp, original_text):
    scores = []
    fallback = cluster[0]
    original_text_lower = original_text.lower()

    for mention in cluster:
        if mention.lower() not in original_text_lower:
            continue  # ❗ original_text에 없는 건 건너뛴다

        doc = nlp(mention)
        tokens_lower = [tok.text.lower() for tok in doc]
        if all(tok in penalized_pronouns for tok in tokens_lower):
            total_score = -100
        else:
            ner_score = sum(1 for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "LOC"})
            propn_score = sum(1 for tok in doc if tok.pos_ == "PROPN")
            total_score = ner_score * 2 + propn_score

        scores.append((total_score, mention))

    if scores:
        scores.sort(reverse=True)
        return scores[0][1]
    else:
        return fallback
    
def replace_mentions_with_representative(tokens, clusters, nlp, original_text):
    token_map = {i: tok for i, tok in enumerate(tokens)}

    for cluster in clusters:
        mentions = [(start, end, " ".join(tokens[start:end + 1])) for start, end in cluster]
        mentions_sorted = sorted(mentions, key=lambda x: x[0])

        # ✅ penalized 아닌 첫 mention을 대표로
        representative = None
        for _, _, mention in mentions_sorted:
            if not contains_penalized_word(mention, penalized_pronouns):
                representative = mention
                break
        if not representative:
            representative = mentions_sorted[0][2]

        # ✅ penalized 포함 mention은 모두 치환
        for start, end, mention_text in mentions_sorted[1:]:
            if contains_penalized_word(mention_text, penalized_pronouns):
                token_map[start] = representative
                for i in range(start + 1, end + 1):
                    token_map[i] = ""

    return [token_map[i] for i in range(len(tokens)) if token_map[i] != ""]


# ✅ 단일 텍스트 입력 + 내부에서 모델 로드 및 해제
def resolve_coreferences(text: str, device="cuda") -> str:
    

    model = Maverick("sapienzanlp/maverick-mes-ontonotes", device=device)
    nlp = spacy_load("en_core_web_lg")

    sentences_list = sent_tokenize(text)
    resolved_sentences = []
    chunk, chunk_length = [], 0

    for sentence in sentences_list:
        tokens = custom_tokenize(sentence)
        token_count = len(tokens)

        if chunk_length + token_count > MAX_TOKENS:
            ontonotes_format = [custom_tokenize(sent) for sent in chunk if sent.strip()]
            if ontonotes_format:
                try:
                    result = model.predict(ontonotes_format)
                    
                    tokens = result.get("tokens", [])
                    clusters = result.get("clusters_token_offsets", [])
                    resolved_tokens = replace_mentions_with_representative(tokens, clusters, nlp)
                    resolved_sentences.append(custom_detokenize(resolved_tokens))
                except IndexError:
                    print("[DEBUG] model.predict() executed, about to assert False")
                    assert False
            chunk, chunk_length = [sentence], token_count
        else:
            chunk.append(sentence)
            chunk_length += token_count

    if chunk:
        ontonotes_format = [custom_tokenize(sent) for sent in chunk if sent.strip()]
        try:
            result = model.predict(ontonotes_format)
            
            
            tokens = result.get("tokens", [])
            clusters = result.get("clusters_token_offsets", [])
            resolved_tokens = replace_mentions_with_representative(tokens, clusters, nlp, original_text=text)
            resolved_sentences.append(custom_detokenize(resolved_tokens))
        except IndexError:
            print("[DEBUG] model.predict() executed, about to assert False")
            assert False

    # ✅ 모델 메모리 정리
    del model, nlp
    torch.cuda.empty_cache()

    return " ".join(resolved_sentences)


# === 직접 실행 확인 (옵션)
if __name__ == "__main__":
    test = "John loves NLP. He is an AI researcher."
    resolved = resolve_coreferences(test)
    print("Original:", test)
    print("Resolved:", resolved)
