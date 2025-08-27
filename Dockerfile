
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN python bootstrap.py || true

CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
