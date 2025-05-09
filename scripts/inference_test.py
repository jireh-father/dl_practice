from src.sentiment import get_sentiment

if __name__ == "__main__":
    texts = [
        "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end.",
        "I really did not like this movie. It was boring and too long.",
        "It was okay, not the best but not the worst either."
    ]
    results = get_sentiment(texts)
    for text, result in zip(texts, results):
        print(f"입력: {text}")
        print(f"결과: {result}\n")

        