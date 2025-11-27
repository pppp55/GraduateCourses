# Combined Dockerfile - contains both Subtask1 (sentiment) and Subtask2 (topic)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy all code so imports like subtask1.* and subtask2.* work
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r combined/requirements.txt \
    && python -m nltk.downloader stopwords wordnet

EXPOSE 5724

# Run combined app
CMD ["uvicorn", "combined.app:app", "--host", "0.0.0.0", "--port", "5724"]
