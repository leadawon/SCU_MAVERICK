# assert False, "this py file is old code"    
# import sys
# sys.path.append('/workspace/SCU-text-evaluation')  # 또는 정확한 절대경로
# import re
# import spacy
# import penman
# from penman.models import amr
# from penman.graph import Graph
# from amrlibfd.amrlib.graph_processing.amr_fix import maybe_fix_unlinked_in_subgraph
# import amrlibfd.amrlib as amrlib

# # Ensure proper module resolution
# sys.path.append('/workspace/SCU-text-evaluation')

# # Load spaCy and AMR models
# spacy_o = spacy.load("en_core_web_sm")
# DIR_STOG_MODEL = '../model_parse_xfm_bart_large-v0_1_0'
# DIR_GTOS_MODEL = '../model_generate_t5wtense-v0_1_0'
# stog = amrlib.load_stog_model(DIR_STOG_MODEL)
# gtos = amrlib.load_gtos_model(DIR_GTOS_MODEL)


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

# def inner_decompose(summary):
#     # Sentence split
#     page_doc = spacy_o(summary)  # Do NOT disable tagger to avoid lemmatizer warnings
#     sentences = [se.text for se in page_doc.sents]

#     # AMR parse
#     parse_result = stog.parse_sents(sentences)

#     # 출력 추가
#     print("\n--- AMR Graphs ---")
#     if isinstance(parse_result, tuple) and len(parse_result) == 2:
#         graphs, graphs_tags = parse_result
#     else:
#         graphs = parse_result
#         graphs_tags = [""] * len(graphs)

#     for i, g in enumerate(graphs):
#         print(f"[Sentence {i+1}]")
#         print(g.strip())
#         print()


#     all_atomic_facts = []
#     for g, g_tag in zip(graphs, graphs_tags):
#         dict_tag = get_concepts(g_tag)

#         subgraphs = get_subgraphs(g)

#         subgraphs_tag = []
#         for sb in subgraphs:
#             sb = maybe_fix_unlinked_in_subgraph(g, sb)
#             sb = gstring_to_oneline(sb)
#             sb = replace_graph_with_tags(dict_tag, sb)
#             subgraphs_tag.append(sb)

#         # Generate natural language from subgraphs
#         #sents, _ = gtos.generate_taged(subgraphs_tag, disable_progress=True)
#         sents, _ = gtos.generate(subgraphs_tag, disable_progress=True)
#         all_atomic_facts.extend(sents)

#     return all_atomic_facts

# def get_subgraphs(amr_graph):
#     g = penman.decode(amr_graph, model=amr.model)
#     t = penman.configure(g)
#     dict_variables = {}
#     root_node = t.node[0]
#     subgraphs = []
#     for path, branch in t.walk():
#         val_node = path_var(path, t.node)
#         if val_node not in dict_variables:
#             dict_variables[val_node] = branch
#         if val_node != root_node:
#             continue
#         role, target = branch
#         if isinstance(target, tuple):
#             linearized_graph = penman.format((val_node, [dict_variables[val_node], (role, target)]))
#             subgraphs.append(linearized_graph)
#     return subgraphs

# def path_var(path, node):
#     var, branches = node
#     for step in path[:-1]:
#         var, branches = branches[step][1]
#     return var

# if __name__ == '__main__':
#     summary = "A police chief resigned from a police chief post to appear on BBC Question Time."
#     atomic_facts = inner_decompose(summary)
#     print("\n--- Atomic Facts ---")
#     for fact in atomic_facts:
#         print(f"- {fact}")
