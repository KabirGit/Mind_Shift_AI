FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; faiss-cpu and spaCy wheels are self-contained.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m spacy download en_core_web_sm

COPY . .

EXPOSE 8501

# NOTE: The LLM call uses an external free-tier API. Pass the key at runtime:
#   docker run -e MISTRAL_API_KEY=<token> -p 8501:8501 <image>
CMD ["sh", "-c", "streamlit run app.py --server.address=0.0.0.0 --server.port=${PORT:-8501}"]
