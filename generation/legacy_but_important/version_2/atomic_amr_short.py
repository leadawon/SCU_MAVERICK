import sys
import os
import random
import numpy as np
import torch
import re
import spacy
import penman
from penman.models import amr
from penman.graph import Graph
from collections import OrderedDict
from transformers import set_seed as hf_set_seed, set_seed
from amrlibfd.amrlib.graph_processing.amr_fix import maybe_fix_unlinked_in_subgraph
import amrlib
from do_maverick import resolve_coreferences

# ===== 시드 고정 =====
def fix_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)
    hf_set_seed(seed)

fix_seed(42)

# ===== 전역 설정 =====
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
spacy_o = spacy.load("en_core_web_sm")

GENERIC_TERMS = [
    "person", "people", "man", "woman", "individual",
    "thing", "the thing", "something", "object", "item", "entity",
    "event", "situation", "place", "location", "area",
    "group", "organization", "he", "she", "his", "her", "they", "them", "their"
]

RELATED_ROLES = {":cause", ":purpose", ":time", ":reason", ":because-of", ":result"}
RELATED_PREDS = {
    "cause-01", "result-01", "purpose-01", "goal-01", "aim-01", "reason-01",
    "time-entity", "before", "after", "explain-01", "lead-01", "intend-01"
}

# ===== 모델 로드 =====
DIR_STOG_MODEL = '../model_parse_xfm_bart_large-v0_1_0'
DIR_GTOS_MODEL = '../model_generate_t5wtense-v0_1_0'
stog = amrlib.load_stog_model(DIR_STOG_MODEL)
gtos = amrlib.load_gtos_model(DIR_GTOS_MODEL)

# ===== 유틸 함수 =====
def gstring_to_oneline(gstring):
    graph_lines = [line.strip() for line in gstring.splitlines() if not line.startswith('#')]
    gstring = ' '.join(graph_lines)
    return re.sub(' +', ' ', gstring)

def get_concepts(g_tag):
    tokens = g_tag.split()
    dict_concepts = {}
    for t in tokens:
        if "~" in t:
            t = t.replace("(", "").replace(")", "")
            parts_t = t.split("~")
            dict_concepts[parts_t[0]] = t
    return dict_concepts

def replace_graph_with_tags(dict_tag, graph):
    for key, value in dict_tag.items():
        graph = graph.replace(key, value)
    return graph

def replace_general_terms_with_specific(original_text: str, atomic_sentence: str) -> str:
    SEP = " | | | "
    combined_text = f"{original_text.strip()}{SEP}{atomic_sentence.strip()}"
    resolved_combined = resolve_coreferences(combined_text)
    if SEP in resolved_combined:
        return resolved_combined.split(SEP)[-1].strip()
    else:
        return resolved_combined.strip()

def get_subgraphs_relation_aware(amr_graph):
    g = penman.decode(amr_graph, model=amr.model)
    root = g.triples[0][0]
    root_triples = [t for t in g.triples if t[0] == root]

    subgraph_triples = list(root_triples)
    related_targets = set()

    for t in root_triples:
        if t[1] in RELATED_ROLES:
            related_targets.add(t[2])
            subgraph_triples.append(t)

    predicate_nodes = set()
    for t in g.triples:
        subj, role, obj = t
        if role == ":instance" and obj in RELATED_PREDS:
            predicate_nodes.add(subj)

    related_targets.update(predicate_nodes)

    for t in root_triples:
        if t[1] == ":ARG0":
            arg0_var = t[2]
            for triple in g.triples:
                if triple[0] == arg0_var and triple[1] == ":instance" and triple[2] == "person":
                    for t2 in g.triples:
                        if t2[0] == arg0_var or t2[2] == arg0_var:
                            subgraph_triples.append(t2)

    for target in related_targets:
        for t in g.triples:
            if t[0] == target or t[2] == target:
                subgraph_triples.append(t)

    subgraph_triples = list(OrderedDict.fromkeys(subgraph_triples))
    return [penman.format(penman.configure(Graph(subgraph_triples)))]

# ===== 메인 처리 함수 =====
def process_summary(summary: str):
    sentences = [s.text for s in spacy_o(summary).sents]
    graphs = stog.parse_sents(sentences, add_metadata=True)
    graph_tags = [''] * len(graphs)

    all_atomic_facts = []
    for s, g, g_tag in zip(sentences, graphs, graph_tags):
        g_penman = penman.decode(g, model=amr.model)
        root = g_penman.triples[0][0]
        root_triples = [t for t in g_penman.triples if t[0] == root]
        contains_predicate = any(
            t[1] == ":instance" and t[2] in RELATED_PREDS for t in g_penman.triples
        )
        if not any(t[1] in RELATED_ROLES for t in root_triples) and not contains_predicate:
            continue

        dict_tag = get_concepts(g_tag)
        subgraphs = get_subgraphs_relation_aware(g)

        subgraphs_tag = []
        for sb in subgraphs:
            sb = maybe_fix_unlinked_in_subgraph(g, sb)
            sb = gstring_to_oneline(sb)
            sb = replace_graph_with_tags(dict_tag, sb)
            subgraphs_tag.append(sb)

        sents, _ = gtos.generate(subgraphs_tag, disable_progress=True)

        for sent in sents:
            postprocessed = replace_general_terms_with_specific(summary, sent)
            all_atomic_facts.append(postprocessed)

    return all_atomic_facts
