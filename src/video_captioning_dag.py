from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys
import os

# Add project path so Python can find your modules
sys.path.append(os.path.expanduser("~/Downloads/akash/real-time-video-captioning-main/src"))

from collect_softgym_episode import collect_episode
from demo_bridge_inference import run_inference

def start_data_collection():
    collect_episode()

def start_inference():
    run_inference()

with DAG(
    dag_id="video_captioning_dag",
    start_date=datetime(2025, 7, 31),
    schedule_interval=None,
    catchup=False,
    tags=["video_captioning"]
) as dag:

    collect_data_task = PythonOperator(
        task_id="collect_data",
        python_callable=start_data_collection
    )

    inference_task = PythonOperator(
        task_id="run_inference",
        python_callable=start_inference
    )

    collect_data_task >> inference_task

