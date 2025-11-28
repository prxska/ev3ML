from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# Configuración básica (Dueño, reintentos, etc.)
default_args = {
    'owner': 'Jorge Garrido',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Definición del DAG (El "Mapa")
with DAG(
    'proyecto_final_mlops_v3',       # Nombre que saldrá en la web
    default_args=default_args,
    description='Orquestador Maestro: Datos -> Clustering -> Integración',
    schedule_interval=None,          # No corre solo, espera tu click
    start_date=datetime(2023, 1, 1),
    catchup=False,
    tags=['kedro', 'evaluacion3'],
) as dag:

    # --- TAREA 1: Procesar Datos ---
    # Llama a tu pipeline 'dp' (Data Processing)
    t1_data = BashOperator(
        task_id='1_data_processing',
        bash_command='cd /opt/airflow/kedro_project && kedro run --pipeline=dp',
    )

    # --- TAREA 2: Aprendizaje No Supervisado ---
    # Llama a tu pipeline 'ul' (Unsupervised Learning)
    t2_unsupervised = BashOperator(
        task_id='2_unsupervised_learning',
        bash_command='cd /opt/airflow/kedro_project && kedro run --pipeline=ul',
    )

    # --- TAREA 3: Integración y Modelo Final ---
    # Llama a tu pipeline 'int' (Integration)
    t3_integration = BashOperator(
        task_id='3_model_integration',
        bash_command='cd /opt/airflow/kedro_project && kedro run --pipeline=int',
    )

    # --- EL ORDEN (Las flechas del grafo) ---
    # t1 -> t2 -> t3
    t1_data >> t2_unsupervised >> t3_integration