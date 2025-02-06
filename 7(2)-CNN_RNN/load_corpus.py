# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    dataset = load_dataset("google-research-datasets/poem_sentiment")
    corpus = [entry["verse_text"] for entry in dataset["train"] if entry["verse_text"].strip()]
   
    return corpus