# Airflow Lab 1
- End-to-End K-Means ML Workflow Orchestrated with Apache Airflow & Docker

- This project implements a complete MLOps workflow using Apache Airflow running inside Docker containers.
The pipeline ingests and preprocesses the Wholesale Customers Dataset, trains a K-Means clustering model, and determines the optimal number of clusters using the Elbow Method.

- All tasks are automated through an Airflow DAG.

## Lab Overview

### This lab demonstrates:

- Running Airflow entirely inside Docker (no local install needed)

- Building a multi-step ML pipeline orchestrated with Airflow

- Passing data across tasks using XCom (Base64 + pickle)

- Preprocessing using MinMaxScaler

- Training a K-Means model for 1–50 clusters

- Evaluating the Elbow Method using KneeLocator

- Saving the trained model

- Viewing logs and execution graph inside Airflow UI

### The DAG runs four automated tasks:

- Load Dataset — read and serialize CSV

- Preprocess Data — scale numeric fields

- Train & Save K-Means Model — compute SSE for elbow

- Elbow Method Evaluation — identify optimal cluster count

## Lab Structure

LAB_1/
│
├── config/
│   └── airflow.cfg
│
├── dags/
│   ├── data/
│   │   └── Wholesale customers data.csv       # Custom dataset
│   │
│   ├── model/
│   │   └── wholesale_model.sav               # Saved model
│   │
│   ├── src/
│   │   ├── __init__.py
│   │   └── lab.py                            # All ML logic
│   │
│   └── airflow.py                            # Airflow DAG
│
├── logs/                                     # Auto-generated
├── plugins/
├── .env
├── docker-compose.yaml
├── setup.sh
└── README.md

## Dataset Used — Wholesale Customers Data

- This dataset contains annual spending of customers across six product categories.

Columns include:

- Fresh

- Milk

- Grocery

- Frozen

- Detergents_Paper

- Delicassen

Your pipeline uses scaled numeric features to build a clustering model.

## Pipeline Steps (Task-by-Task Explanation)

* load_data()

Reads Wholesale customers data.csv

Serializes the DataFrame → pickle → Base64

Returns JSON-safe Base64 string for XCom

2️⃣ data_preprocessing()

Receives Base64 input

Deserializes into DataFrame

Drops missing values

Selects numerical features

Applies MinMaxScaler()

Returns scaled NumPy array encoded in Base64

3️⃣ build_save_model()

Iterates KMeans for k = 1 to 50

Computes SSE (inertia) for each k

Saves the final model to:
