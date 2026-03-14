# Minimal image for the LLM Query API
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# System deps (optional: build essentials; add more if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    build-essential \ 
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
ENV API_KEY=
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]


# --- Optional for local Hugging Face models ---
# COPY requirements-local.txt ./
# RUN pip install --no-cache-dir -r requirements-local.txt
# ENV HF_HOME=/root/.cache/huggingface
# VOLUME ["/root/.cache/huggingface"]
