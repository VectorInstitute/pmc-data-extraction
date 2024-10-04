import pandas as pd
import numpy as np
import os
import json

pmcpatients_root_dir = "/projects/multimodal/datasets/pmc_patients/"

# load patient entries
# data = pd.read_csv(os.path.join(pmcpatients_root_dir, "PMC-Patients/PMC-Patients.csv"))
# print(data)

# load test queries
queries_file = os.path.join(pmcpatients_root_dir, "PMC-Patients-ReCDS/queries/test_queries.jsonl")
with open(queries_file, encoding="utf-8") as file:
    queries = [json.loads(line) for line in file.readlines()]
queries = pd.DataFrame.from_records(queries)

# load ppr corpus
corpus_file = os.path.join(pmcpatients_root_dir, "PMC-Patients-ReCDS/PPR/corpus.jsonl")
with open(corpus_file, encoding="utf-8") as file:
    corpus = [json.loads(line) for line in file.readlines()]
corpus = pd.DataFrame.from_records(corpus)
# print(corpus)

# load ppr test qrels
qrels_file = os.path.join(pmcpatients_root_dir, "PMC-Patients-ReCDS/PPR/qrels_test.tsv")
qrels = pd.read_csv(qrels_file, sep="\t")

# get idx'th sample
idx = 0
query_id = qrels.iloc[idx]["query-id"]
corpus_id = qrels.iloc[idx]["corpus-id"]
print(f"query_id: {query_id}")
print(f"corpus_id: {corpus_id}")

query_text = queries["text"].loc[queries["_id"] == query_id].values[0]
target_text = corpus["text"].loc[corpus["_id"] == corpus_id].values[0]

print(query_text)
print(target_text)

print(type(query_text))
print(type(target_text))

print(len(qrels.index))

# get length statistics
lengths = []
for query in queries["text"].values:
    lengths.append(len(query.split(" ")))
lengths = np.array(lengths)
print(f"num queries: {len(queries)}, average text length: {np.mean(lengths)}, max text length: {np.max(lengths)}")