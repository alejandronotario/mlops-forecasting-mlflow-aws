import json
import requests
from datetime import datetime, timedelta
from sqlalchemy import create_engine
import pandas as pd
import logging
from airflow.hooks.postgres_hook import PostgresHook

logger = logging.getLogger(__name__)


def get_data(ti):
    """
    Respuesta de la API
    """
    url = 'https://www.gasnetworks.ie/corporate/open-data/Daily-Supply-Q2-2024.json'
    res = requests.get(url).json()
    logger.info(res)
    ti.xcom_push(key="gas_data", value=res)


def load_data(**kwargs):
    """
    Carga en Postgres los datos de entrada
    """
    ti = kwargs['ti']
    gas_data = ti.xcom_pull(key=kwargs['key'], task_ids=kwargs['task_id'])
    logger.info(gas_data)
    records_list = []

    for r in gas_data:
        rows = {
            #'ds': r['Date'],
            'ds': datetime.strptime(r['Date'], "%d/%m/%Y").strftime('%Y-%m-%d'),
            'y': r['Corrib production']
        }
        records_list.append(rows)
    df = pd.DataFrame(records_list)
    df = df[(df['ds'] >= kwargs['fecha_ini']) & (df['ds'] <= kwargs['fecha_fin'])]
    #df['ds'] = df['ds'].astype(str)
    logger.info("CONTEO: ", df.count())
    """
    En la configuración del motor es imprescindible que los argumentos sean:
    db_user: usuario incluido en docker compose
    df_pswd: pasword incluido en docker compose
    db_host: nombre del servicio sobre el que está la imagen postgres
    db_name: nombre de la base de datos en docker compose
    """
    engine = create_engine(f"postgresql+psycopg2://{kwargs['db_user']}:{kwargs['db_pswd']}@{kwargs['db_host']}/{kwargs['db_name']}")
    df.to_sql('gas_supply', con=engine, schema='public', if_exists='append', index=False)
    logger.info(f"Exito: Cargados {len(df)} registros en {kwargs['db_name']}.")


def load_incremental_data(**kwargs):
    """
    Carga en Postgres los datos de entrada de forma incremental
    """
    ti = kwargs['ti']
    gas_data = ti.xcom_pull(key=kwargs['key'], task_ids=kwargs['task_id'])
    last_record = ti.xcom_pull(key=kwargs['key_2'], task_ids=kwargs['task_id_2'])
    last_record = datetime.strptime(last_record, "%Y-%m-%d").strftime('%Y-%m-%d')
    last_record_7_days = (datetime.strptime(last_record, "%Y-%m-%d") + timedelta(
        days=kwargs['days'])).strftime('%Y-%m-%d')
    logger.info(type(last_record_7_days), last_record_7_days)
    records_list = []

    for r in gas_data:
        rows = {
            'ds': datetime.strptime(r['Date'], "%d/%m/%Y").strftime('%Y-%m-%d'),
            'y': r['Corrib production']
        }
        records_list.append(rows)
    df = pd.DataFrame(records_list)
    df = df[(df['ds'] >= last_record) & (df['ds'] <= last_record_7_days)]
    logger.info("CONTEO: ", df.count())
    engine = create_engine(f"postgresql+psycopg2://{kwargs['db_user']}:{kwargs['db_pswd']}@{kwargs['db_host']}/{kwargs['db_name']}")
   
    df.to_sql('gas_supply', con=engine, schema='public', if_exists='append', index=False)
  


def read_data_from_postgres(fecha_foto: str):
    """
    Lectura de Postgres los datos de entrada
    """
    fecha_foto = fecha_foto
    logger.info(fecha_foto)
    registros = []
    request = "SELECT * FROM public.gas_supply where ds > '{}'".format(fecha_foto)
    logger.info(request)
    pg_hook = PostgresHook(postgres_conn_id='postgres')
    connection = pg_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute(request)
    sources = cursor.fetchall()
    for source in sources:
        registros.append(source)
    logger.info(source)
    return registros


    
def read_last_record_from_postgres(ti):
    """
    Lectura de último registro de Postgres        
    """
    request = "SELECT max(ds) FROM public.gas_supply"
    logger.info(request)
    pg_hook = PostgresHook(postgres_conn_id='postgres')
    connection = pg_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute(request)
    res = cursor.fetchone()[0]
    logger.info("resultado", type(res), res)
    ti.xcom_push(key="last_record", value=res)

def read_first_record_from_postgres(ti):
    """
    Lectura de último registro de Postgres        
    """
    request = "SELECT min(ds) FROM public.gas_supply"
    logger.info(request)
    pg_hook = PostgresHook(postgres_conn_id='postgres')
    connection = pg_hook.get_conn()
    cursor = connection.cursor()
    cursor.execute(request)
    res = cursor.fetchone()[0]
    logger.info("resultado", type(res), res)
    ti.xcom_push(key="first_record", value=res)

def load_interval_data(**kwargs):
    """
    Lectura de conjunto de datos según fechas      
    """
    ti = kwargs['ti']
    first_record = ti.xcom_pull(key=kwargs['key'], task_ids=kwargs['task_id'])
    last_record = ti.xcom_pull(key=kwargs['key_2'], task_ids=kwargs['task_id_2'])
    hook = PostgresHook(postgres_conn_id="postgres")
    df = hook.get_pandas_df(
        sql="SELECT * FROM public.gas_supply where ds between '{}' and '{}'".format(
            first_record, last_record 
        ))
    res = df.to_json(orient="records")
    logger.info(df.to_json(orient="records"))
    ti.xcom_push(key="gas_data", value=res)