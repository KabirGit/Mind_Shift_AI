FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence-transformers \
    TRANSFORMERS_CACHE=/app/.cache/huggingface/transformers

# System deps kept minimal; faiss-cpu and spaCy wheels are self-contained.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    && python -c "from transformers import pipeline; pipeline('text-classification', model='SamLowe/roberta-base-go_emotions', top_k=None)"

COPY . .

EXPOSE 8501

# NOTE: The LLM call uses an external free-tier API. Pass the key at runtime:
#   docker run -e MISTRAL_API_KEY=<token> -p 8501:8501 <image>
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
