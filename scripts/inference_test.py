from transformers import pipeline

if __name__ == "__main__":
    # 사용할 사전학습 모델 지정 (예: distilbert-base-uncased-finetuned-sst-2-english)
    model_name = "distilbert-base-uncased-finetuned-sst-2-english"
    classifier = pipeline("sentiment-analysis", model=model_name)

    # 테스트할 문장
    texts = [
        "This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end.",
        "I really did not like this movie. It was boring and too long.",
        "It was okay, not the best but not the worst either."
    ]

    for text in texts:
        result = classifier(text)
        print(f"입력: {text}")
        print(f"결과: {result}\n")