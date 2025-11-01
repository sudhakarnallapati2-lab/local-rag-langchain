FROM python:3.10-slim

# system deps for faiss and tesseract/pdf if needed
RUN apt-get update && apt-get install -y --no-install-recommends         build-essential git wget ffmpeg libsndfile1 &&         rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# copy project files
COPY . /app

# create folders for persistence
RUN mkdir -p /app/data /app/vectorstore

ENV PORT=8501
EXPOSE 8501

# default command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.runOnSave=false", "--server.headless=true"]
