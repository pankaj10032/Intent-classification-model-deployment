FROM python:3.9
WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Use the recommended HF_HOME instead of deprecated TRANSFORMERS_CACHE
ENV HF_HOME=/code/cache/huggingface

# Create the directory for the Transformers cache and set permissions
RUN mkdir -p /code/cache/huggingface && chmod -R 777 /code/cache/huggingface

COPY . /app

EXPOSE 7860
# Increase Gunicorn timeout to prevent worker timeout during long initializations
CMD ["gunicorn", "-b", "0.0.0.0:7862", "main:app", "--timeout", "120",]

# CMD ["gunicorn", "-b", "0.0.0.0:7862", "main:app", "--timeout", "120", "--workers", "2", "--threads", "2"]

