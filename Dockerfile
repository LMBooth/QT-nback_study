FROM python:3.11-bullseye

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libx11-xcb1 \
    libxext6 \
    libxrender1 \
    libxrandr2 \
    libxfixes3 \
    libxi6 \
    libxkbcommon-x11-0 \
    libsm6 \
    libxdamage1 \
    libxcomposite1 \
    libxcursor1 \
    libxtst6 \
    libnss3 \
    libasound2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app

CMD ["python", "n-back/mainExperiment.py"]
