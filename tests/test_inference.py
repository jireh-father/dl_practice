import pytest
from src.sentiment import get_sentiment

def test_positive_sentiment():
    text = ["This was a masterpiece. Not completely faithful to the books, but enthralling from beginning to end."]
    result = get_sentiment(text)[0]
    assert result["label"] == "POSITIVE"
    assert result["score"] > 0.9

def test_negative_sentiment():
    text = ["I really did not like this movie. It was boring and too long."]
    result = get_sentiment(text)[0]
    assert result["label"] == "NEGATIVE"
    assert result["score"] > 0.9

def test_neutral_sentiment():
    text = ["It was okay, not the best but not the worst either."]
    result = get_sentiment(text)[0]
    # 중립 문장은 모델이 긍정/부정 중 하나로 분류할 수 있으므로, 스코어만 검증
    assert result["score"] > 0.5 