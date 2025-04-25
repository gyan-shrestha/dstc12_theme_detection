from sentence_transformers import SentenceTransformer, InputExample, losses, models
from torch.utils.data import DataLoader
import json

def create_utterance_lookup(data_path):
    lookup = {}
    with open(data_path) as f:
        for line in f:
            dialogue = json.loads(line)
            for turn in dialogue["turns"]:
                if "turn_id" in turn:
                    lookup[turn["turn_id"]] = turn["utterance"]
    return lookup

def load_pairs(preference_file, utt_lookup):
    with open(preference_file) as f:
        prefs = json.load(f)
    examples = []
    for a, b in prefs["should_link"]:
        examples.append(InputExample(texts=[utt_lookup[a], utt_lookup[b]], label=1.0))
    for a, b in prefs["cannot_link"]:
        examples.append(InputExample(texts=[utt_lookup[a], utt_lookup[b]], label=0.0))
    return examples

def main():
    model_name = "sentence-transformers/all-mpnet-base-v2"
    output_dir = "/Users/gyanendra/Spring_2025/Amazon_challenge/dstc12_theme_detection/models/fine_tuned_embedding_model"
    data_path = "/Users/gyanendra/Spring_2025/Amazon_challenge/dstc12_theme_detection/data/all.jsonl"
    prefs = "/Users/gyanendra/Spring_2025/Amazon_challenge/dstc12_theme_detection/data/preference_pairs.json"
    print("-------starting----------------")
    utt_lookup = create_utterance_lookup(data_path)
    print("-----------finisdhed utt_lookup-----------")
    train_data = load_pairs(prefs, utt_lookup)

    word_model = models.Transformer(model_name)
    pooling = models.Pooling(word_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_model, pooling])

    dataloader = DataLoader(train_data, shuffle=True, batch_size=16)
    loss = losses.CosineSimilarityLoss(model)

    model.fit(train_objectives=[(dataloader, loss)], epochs=1, output_path=output_dir)

if __name__ == "__main__":
    main()
