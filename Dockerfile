FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv

# CPU-only torch first: ~200MB instead of ~4GB of CUDA wheels
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

COPY pyproject.toml uv.lock ./
RUN uv pip install --system --no-cache -r pyproject.toml

COPY . .

# -m gives a writable HOME, which streamlit needs
RUN useradd -m -u 1000 app && chown -R app /app
USER app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
