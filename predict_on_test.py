import json, os, copy
from sklearn.cluster import KMeans
import numpy as np
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
    test_file = "data/appen_travel_test.jsonl"
    output_file = "all-to-predict.jsonl"
    embedding_model_path = "models/fine_tuned_embedding_model"
    n_clusters = 10  # you can tune this

    with open(test_file) as f:
        data = [json.loads(line) for line in f]
    utterances = list({turn['utterance'] for d in data for turn in d['turns']})

    model = SentenceTransformer(embedding_model_path)
    embeddings = model.encode(utterances)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clustered_utterances = [[] for _ in range(n_clusters)]
    for utt, label in zip(utterances, labels):
        clustered_utterances[label].append(utt)

    llm = HuggingFaceLLM(model_id="mistralai/Mistral-7B-Instruct-v0.3")
    parser = DotAllRegexParser(r"<theme_label>(.*?)</theme_label>")

    label_map = {}
    for cluster in clustered_utterances:
        joined = "\n".join(cluster)
        prompt = LABEL_CLUSTERS_PROMPT.format(utterances=joined)
        result = llm.invoke(prompt)
        label = parser.invoke(result)
        for u in cluster:
            label_map[u] = label

    # Assign labels back to test data
    output_data = copy.deepcopy(data)
    for d in output_data:
        for turn in d["turns"]:
            turn["theme_label_predicted"] = label_map.get(turn["utterance"], "")

    with open(output_file, "w") as f:
        for d in output_data:
            print(json.dumps(d), file=f)

if __name__ == "__main__":
    main()
