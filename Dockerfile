FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv

# ponytail: CPU-only torch first (~200MB vs ~4GB of CUDA wheels).
# Later resolve sees torch satisfied. Swap to the cu121 index if you ever add a GPU.
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
