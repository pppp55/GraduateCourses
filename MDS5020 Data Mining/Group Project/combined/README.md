Combined service for Task2 Subtask1 (sentiment) and Subtask2 (topic).

Structure:

- `subtask1/` 原始情感分析代码及模型
- `subtask2/` 原始主题分类代码及模型
- `app.py` 统一的 FastAPI 入口
- `Dockerfile` + `requirements.txt` + `README.md`

构建 & 运行（在仓库根目录执行）：

```bash
docker build -f combined/Dockerfile -t combined-news-api:latest .
docker run --rm -p 5724:5724 combined-news-api:latest
```

如只保留 `combined/` 目录，也可以在该目录内运行：

```bash
docker build -t combined-news-api .
docker run --rm -p 5724:5724 combined-news-api
```

健康检查：

```bash
curl http://localhost:5724/health
```

情感预测：

```bash
curl -X POST http://localhost:5724/predict_sentiment \
	-H "Content-Type: application/json" \
	-d '{"news_text":"Global markets rally as tech stocks surge ahead of earnings."}'
```

主题预测：

```bash
curl -X POST http://localhost:5724/predict_topic \
	-H "Content-Type: application/json" \
	-d '{"text":"上市保荐书"}'
```