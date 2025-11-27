# api_service.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.base import BaseEstimator, TransformerMixin
import joblib
import numpy as np
import jieba
import re
import pandas as pd
from pathlib import Path
import sys

# 预初始化jieba，避免运行时下载
jieba.initialize()


# 首先定义 NewsTitlePreprocessor 类
class NewsTitlePreprocessor(BaseEstimator, TransformerMixin):
    """新闻标题预处理器"""

    def __init__(self):
        self.stop_words = {
            "公司",
            "股份",
            "有限",
            "责任",
            "集团",
            "关于",
            "公告",
            "事项",
        }

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return [self._preprocess_text(text) for text in X]

    def _preprocess_text(self, text):
        if pd.isna(text):
            return ""

        cleaned_text = re.sub(r"^[^:]*:", "", str(text))
        company_patterns = [
            r"^[A-Za-z0-9\u4e00-\u9fa5]{2,8}?:",
            r"^[A-Za-z0-9\u4e00-\u9fa5]{2,8}?公告:",
            r"^[A-Za-z0-9\u4e00-\u9fa5]{2,8}?关于",
        ]

        for pattern in company_patterns:
            cleaned_text = re.sub(pattern, "", cleaned_text)

        cleaned_text = re.sub(r"\d{4}年|\d{1,2}月|\d{1,2}日|\d+", " ", cleaned_text)
        cleaned_text = re.sub(r"[^\u4e00-\u9fa5]", " ", cleaned_text)

        words = jieba.cut(cleaned_text)
        result = []
        for word in words:
            if len(word) > 1 and word not in self.stop_words:
                result.append(word)

        return " ".join(result)


# Ensure unpickling works even if original model was trained in __main__
sys.modules.setdefault("__main__", sys.modules[__name__])
sys.modules["__main__"].NewsTitlePreprocessor = NewsTitlePreprocessor


# 加载模型
BASE_DIR = Path(__file__).resolve().parent


def load_model():
    try:
        model_path = BASE_DIR / "news_topic_model.pkl"
        model_data = joblib.load(model_path)
        return model_data
    except Exception as e:
        raise RuntimeError(f"无法加载模型: {str(e)}")


# 初始化模型
model_data = load_model()
pipeline = model_data["pipeline"]
topic_labels = model_data["topic_labels"]

# 在模型加载后添加检查信息
print("=== API服务模型信息 ===")
print(f"模型加载时间: 现在")
print(f"主题数量: {len(topic_labels)}")
print(f"支持的分类: {topic_labels}")

# 测试一个简单预测
test_result = pipeline.predict(["上市保荐书"])[0]
test_prob = pipeline.predict_proba(["上市保荐书"])[0].max()
print(f"测试预测: '上市保荐书' -> {test_result} (概率: {test_prob:.4f})")
print("=" * 50)

app = FastAPI(
    title="新闻主题分类API",
    description="基于机器学习的新闻主题分类服务",
    version="1.0.0",
)

TOPIC_TO_ID = {
    "上市保荐书": "1",
    "保荐/核查意见": "2",
    "公司章程": "3",
    "公司章程修订": "4",
    "关联交易": "5",
    "分配方案决议公告": "6",
    "分配方案实施": "7",
    "分配预案": "8",
    "半年度报告全文": "9",
    "发行保荐书": "10",
    "年度报告全文": "11",
    "年度报告摘要": "12",
    "独立董事候选人声明": "13",
    "独立董事提名人声明": "14",
    "独立董事述职报告": "15",
    "股东大会决议公告": "16",
    "诉讼仲裁": "17",
    "高管人员任职变动": "18",
}


class TopicResponse(BaseModel):
    topic: str
    probability: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    topics: list


# 修复Pydantic警告
HealthResponse.model_config["protected_namespaces"] = ()


# 定义请求模型
class PredictionRequest(BaseModel):
    text: str


@app.get("/")
async def root():
    return {"message": "新闻主题分类API服务已启动"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipeline is not None,
        "topics": topic_labels.tolist(),
    }


# 预测端点 - 使用请求体模型
@app.post("/predict_topic", response_model=TopicResponse)
async def predict(request: PredictionRequest):
    try:
        prediction = pipeline.predict([request.text])[0]
        probabilities = pipeline.predict_proba([request.text])[0]
        confidence = float(max(probabilities))

        topic_id = TOPIC_TO_ID.get(prediction, "未知")

        return TopicResponse(topic=topic_id, probability=f"{confidence:.2f}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.get("/topics")
async def get_topics():
    mapping = []
    for topic_name, topic_id in TOPIC_TO_ID.items():
        mapping.append({"id": topic_id, "name": topic_name})
    return {"topic_mapping": mapping}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=5724)
