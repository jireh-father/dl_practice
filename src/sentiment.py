from transformers import pipeline
from typing import List, Dict

def get_sentiment(texts: List[str], model_name: str = "distilbert-base-uncased-finetuned-sst-2-english") -> List[Dict]:
    """
    입력된 텍스트 리스트에 대해 감정 분류 결과를 반환합니다.
    """
    classifier = pipeline("sentiment-analysis", model=model_name)
    return [classifier(text)[0] for text in texts] 