# Serves the precomputed comparison app. Deliberately does NOT ship torch or the
# base model: this image renders results that were produced on a machine with a
# GPU/MPS, which is why it is ~200 MB rather than ~3 GB.
FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --retries 5 --timeout 120 -r requirements.txt
COPY app/ app/
COPY reports/ reports/
EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit","run","app/streamlit_app.py","--server.port=8501","--server.address=0.0.0.0","--server.headless=true"]
