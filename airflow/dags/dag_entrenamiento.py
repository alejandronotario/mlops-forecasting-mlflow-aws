""" Dag para entrenar el modelo."""
import sys
sys.path.append("/opt/airflow/")
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
import logging
from mlflow.tracking import MlflowClient
from src.data.data import read_first_record_from_postgres, read_last_record_from_postgres, load_interval_data
from src.data.data import load_incremental_data, get_data
from src.model.model_train import train, register_best_model

logger = logging.getLogger(__name__)

HOST_NAME = 'postgres'
DATABASE = 'airflow'
USER_NAME = 'airflow'
PASSWORD = 'airflow'

dag = DAG(
    dag_id='dag_entrenamiento',
    start_date=airflow.utils.dates.days_ago(0),
    schedule=None #'0 0 * * 0' para este caso un domingo a las 00:00
)

read_last_record = PythonOperator(
    task_id="read_last_record",
    python_callable=read_last_record_from_postgres,
    dag=dag
)

get_gas_data = PythonOperator(
        task_id="get_gas_data", python_callable=get_data, dag=dag
    )

load_incremental_gas_data = PythonOperator( 
        task_id="load_incremental_gas_data",
        op_kwargs={
        'db_host': HOST_NAME,
        'db_name': DATABASE,
        'db_user': USER_NAME,
        'db_pswd': PASSWORD,
        'key': 'gas_data',
        'task_id': 'get_gas_data',
        'key_2': 'last_record',
        'task_id_2': 'read_last_record',
        'days': 7
    }, python_callable=load_incremental_data, dag=dag
)

get_first_record = PythonOperator(
    task_id="get_first_record",
  python_callable=read_first_record_from_postgres, dag=dag
)

get_last_record = PythonOperator(
    task_id="get_last_record",
    python_callable=read_last_record_from_postgres, dag=dag
)

load_train_data = PythonOperator(
    task_id="load_interval_data",
    op_kwargs={
        'db_host': HOST_NAME,
        'db_name': DATABASE,
        'db_user': USER_NAME,
        'db_pswd': PASSWORD,
        'key': 'first_record',
        'task_id': 'get_first_record',
        'key_2': 'last_record',
        'task_id_2': 'get_last_record',
    }, python_callable=load_interval_data, dag=dag
)

model_train = PythonOperator(
    task_id="model_train",
    op_kwargs={
        'key': 'gas_data',
        'task_id': 'load_interval_data',
    },
    python_callable=train, dag=dag
)

register_model = PythonOperator(
    task_id="register_model",
    op_kwargs={
        'top_n': 4
    },
    python_callable=register_best_model, dag=dag
)


read_last_record >> get_gas_data >> load_incremental_gas_data >> [get_first_record, get_last_record] >> load_train_data >> model_train >> register_model
