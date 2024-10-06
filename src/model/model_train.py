import os
import json
from datetime import datetime
from sqlalchemy import create_engine
import logging
import pandas as pd
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
import itertools
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import logging
from airflow.hooks.postgres_hook import PostgresHook
import requests

logger = logging.getLogger(__name__)
ALERTMANAGER_URL = "http://172.17.0.1:9093/api/v1/alerts"



# param_grid = {  
#     'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
#     'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
#     'holidays_prior_scale': [0.01, 0.1, 1.0, 10.0],
#     'seasonality_mode': ['additive', 'multiplicative']
#     }

param_grid = {  
    'changepoint_prior_scale': [0.01],
    'seasonality_prior_scale': [0.1],
    'holidays_prior_scale': [0.01],
    'seasonality_mode': ['additive']
    }


def train(**kwargs):
    ti = kwargs['ti']
    gas_data = ti.xcom_pull(key=kwargs['key'], task_ids=kwargs['task_id'])
    logger.info(type(gas_data), gas_data)
    df = pd.DataFrame(json.loads(gas_data))
    df['ds'] = pd.to_datetime(df['ds'], format="%Y-%m-%d")
    initial = int(0.7*(df.shape[0]))
    horizon = initial / 3
    period = horizon / 4 
    input_example = df[:int(0.05*(df.shape[0]))]
    mlflow.set_experiment("time series")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"\nActive run_id: {run_id}")
        all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
        rmses = []
        for params in all_params:
            m = Prophet(**params).fit(df)  
            df_cv = cross_validation(m, initial='{} days'.format(initial),
                period='{} days'.format(period), horizon = '{} days'.format(horizon))
            logger.info(df_cv)
            df_p = performance_metrics(df_cv, rolling_window=1)
            logger.info(df_p)
            rmses.append(df_p['rmse'].values[0])
        logger.info(rmses)
        rmse_min = int(min(rmses))
        mlflow.log_metric("rmse", rmse_min)
        tuning_results = pd.DataFrame(all_params)
        tuning_results['rmse'] = rmses
        best_params = {}
        best_params_list = list(all_params[np.argmin(rmses)].items())
        best_params[best_params_list[0][0]] = best_params_list[0][1]
        best_params[best_params_list[1][0]] = best_params_list[1][1]
        best_params[best_params_list[2][0]] = best_params_list[2][1]
        best_params[best_params_list[3][0]] = best_params_list[3][1]
        logger.info(best_params)
        mlflow.log_params(best_params)
        model = Prophet(**best_params).fit(df)
        train = model.history
        future = model.make_future_dataframe(periods=30)
        logger.info(future)
        mlflow.prophet.log_model(pr_model=model, artifact_path="prophet",
                                 input_example=input_example,
                                 registered_model_name="prophet")
        model_uri = mlflow.get_artifact_uri("prophet")
    return run_id, model_uri

def register_best_model(top_n: int) -> None:
    client = MlflowClient()
    logger.info("Registrando el mejor modelo...")
    experiment = client.get_experiment_by_name("time series")
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=top_n,
        order_by=["metrics.RMSE ASC"]
    )
    best_run = runs[0]
    best_model_uri = f"runs:/{best_run.info.run_id}/prophet"
    mlflow.register_model(model_uri=best_model_uri, name="prophet_mejorado")
    logger.info(f"Registrado el modelos {best_model_uri} como 'Mejor modelo'")

def execute():
    hook = PostgresHook(postgres_conn_id="postgres")
    df = hook.get_pandas_df(
        sql="SELECT * FROM public.gas_supply where ds between '2022-02-01' and '2022-02-20'"
        )
    model_name = "prophet"
    model_version = "latest"
    df = df[:int(0.1*(df.shape[0]))]
    df['ds'] = pd.to_datetime(df['ds'], format="%Y-%m-%d")
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    #logger.info(df1)
    forecast = model.predict(df)
    logger.info(forecast)
    y_true = df['y'].values
    logger.info(y_true)
    y_pred = forecast['yhat'].values
    logger.info(y_pred)
    