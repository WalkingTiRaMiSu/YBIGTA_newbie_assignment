# 구현하세요!
from datasets import load_dataset

def load_corpus() -> list[str]:
    corpus: list[str] = []
    # 구현하세요!
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train[:100%]")
    corpus = [item["verse_text"] for item in dataset]
    corpus = [line for line in corpus if len(line.strip()) >= 0]

    return corpus