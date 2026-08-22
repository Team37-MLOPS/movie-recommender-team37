FROM apache/airflow:2.10.4-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends openjdk-17-jre-headless libgomp1 \
    && rm -rf /var/lib/apt/lists/*

USER airflow
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt
