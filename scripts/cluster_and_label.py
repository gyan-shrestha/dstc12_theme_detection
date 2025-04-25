import json, copy
import numpy as np
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
from langchain_huggingface import HuggingFaceLLM
from scripts.prompts import LABEL_CLUSTERS_PROMPT

class DotAllRegexParser:
    def __init__(self, regex):
        import re
        self.pattern = re.compile(regex, re.DOTALL)

    def invoke(self, text):
        match = self.pattern.search(text)
        return match.group(1).strip() if match else "unknown"

def main():
    path = "data/finance_train.jsonl"
    output_path = "output/finance_predictions.jsonl"
    model_path = "models/fine_tuned_embedding_model"
    n_clusters = 10

    with open(path) as f:
        data = [json.loads(line) for line in f]
    utts = list({turn['utterance'] for d in data for turn in d['turns'] if 'theme_label' in turn})

    model = SentenceTransformer(model_path)
    embs = model.encode(utts)

    kmeans = KMeans(n_clusters=n_clusters).fit(embs)
    cluster_utts = [[] for _ in range(n_clusters)]
    for u, label in zip(utts, kmeans.labels_):
        cluster_utts[label].append(u)

    llm = HuggingFaceLLM(model_id="mistralai/Mistral-7B-Instruct-v0.3")
    parser = DotAllRegexParser(r"<theme_label>(.*?)</theme_label>")
    cluster_labels = {}

    for cluster in cluster_utts:
        prompt = LABEL_CLUSTERS_PROMPT.format(utterances="\n".join(cluster))
        result = llm.invoke(prompt)
        label = parser.invoke(result)
        for u in cluster:
            cluster_labels[u] = label

    # Assign predictions
    new_data = copy.deepcopy(data)
    for d in new_data:
        for turn in d["turns"]:
            if "theme_label" in turn:
                turn["theme_label_predicted"] = cluster_labels.get(turn["utterance"], "")

    with open(output_path, "w") as f:
        for d in new_data:
            print(json.dumps(d), file=f)

if __name__ == "__main__":
    main()
