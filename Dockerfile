FROM python:3.7-slim
WORKDIR /app
COPY requirements.txt .

RUN pip install -U pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install gunicorn

COPY app/ .

# Expose the port
EXPOSE 80

CMD python -u app.py