FROM python:3.9-slim

WORKDIR /app

# 复制依赖文件
COPY requirements.txt .

# 安装依赖并预下载jieba词典
RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
    python -c "import jieba; jieba.initialize()"

# 复制应用代码
COPY . .

EXPOSE 5724

CMD ["uvicorn", "api_service:app", "--host", "0.0.0.0", "--port", "5724"]