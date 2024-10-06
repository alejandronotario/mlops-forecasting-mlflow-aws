FROM apache/airflow:2.5.0-python3.8
USER root
RUN apt-get update
USER airflow
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt