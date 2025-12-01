# Import necessary libraries and modules
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
from src.lab import load_data, data_preprocessing, build_save_model, load_model_elbow

# -----------------------------------------------------------------
# DEFAULT ARGUMENTS
# -----------------------------------------------------------------
default_args = {
    'owner': 'Shivakumar Hassan Lokesh',
    'start_date': datetime(2025, 12, 1),
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
}

# -----------------------------------------------------------------
# DAG DEFINITION
# -----------------------------------------------------------------
with DAG(
    'Airflow_Wholesale_Customers',          # NEW unique DAG name
    default_args=default_args,
    description='K-Means Clustering DAG for Wholesale Customers Dataset',
    catchup=False,
) as dag:

    # -------------------------------------------------------------
    # 1. Load Data
    # -------------------------------------------------------------
    load_data_task = PythonOperator(
        task_id='load_data_task',
        python_callable=load_data,
    )

    # -------------------------------------------------------------
    # 2. Preprocess Data
    # -------------------------------------------------------------
    data_preprocessing_task = PythonOperator(
        task_id='data_preprocessing_task',
        python_callable=data_preprocessing,
        op_args=[load_data_task.output],    # uses XCom output
    )

    # -------------------------------------------------------------
    # 3. Build + Save KMeans Model
    # -------------------------------------------------------------
    build_save_model_task = PythonOperator(
        task_id='build_save_model_task',
        python_callable=build_save_model,
        op_args=[data_preprocessing_task.output, "wholesale_model.sav"],  # model filename updated
    )

    # -------------------------------------------------------------
    # 4. Compute Elbow Method
    # -------------------------------------------------------------
    load_model_task = PythonOperator(
        task_id='load_model_task',
        python_callable=load_model_elbow,
        op_args=["wholesale_model.sav", build_save_model_task.output],
    )

    # -------------------------------------------------------------
    # TASK DEPENDENCIES
    # -------------------------------------------------------------
    load_data_task >> data_preprocessing_task >> build_save_model_task >> load_model_task


# -----------------------------------------------------------------
# For direct CLI testing (Optional)
# -----------------------------------------------------------------
if __name__ == "__main__":
    dag.test()
