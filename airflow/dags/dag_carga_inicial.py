""" Dag para cargar los datos iniciales. """
import sys
sys.path.append("/opt/airflow/")
import airflow
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.postgres_operator import PostgresOperator
import logging
from src.data.data import load_data, get_data


logger = logging.getLogger(__name__)

HOST_NAME = 'postgres'
DATABASE = 'airflow'
USER_NAME = 'airflow'
PASSWORD = 'airflow'

dag = DAG(
    dag_id='dag_carga_inicial',
    start_date=airflow.utils.dates.days_ago(0),
    schedule_interval=None
)

create_table = PostgresOperator(task_id='create_table',postgres_conn_id='postgres' ,sql= """CREATE TABLE IF NOT EXISTS public.gas_supply(
	ds varchar(255) NULL,
	y float8 NULL);""")
    

get_gas_data = PythonOperator(
        task_id="get_gas_data", python_callable=get_data, dag=dag
    )

load_gas_data = PythonOperator( 
        task_id="load_gas_data",
        op_kwargs={
        'db_host': HOST_NAME,
        'db_name': DATABASE,
        'db_user': USER_NAME,
        'db_pswd': PASSWORD,
        'key': 'gas_data',
        'task_id': 'get_gas_data',
        'fecha_ini': '2018-01-01',
        'fecha_fin': '2023-06-30'
    }, python_callable=load_data, dag=dag
    )

create_table >> get_gas_data >> load_gas_data