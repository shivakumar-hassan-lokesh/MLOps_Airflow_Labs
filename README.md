# Airflow Lab 1
End-to-End K-Means ML Workflow Orchestrated with Apache Airflow & Docker

This project implements a complete MLOps workflow using Apache Airflow running inside Docker containers.
The pipeline ingests and preprocesses the Wholesale Customers Dataset, trains a K-Means clustering model, and determines the optimal number of clusters using the Elbow Method.

All tasks are automated through an Airflow DAG.

## Lab Overview

### This lab demonstrates:

Running Airflow entirely inside Docker (no local install needed)

Building a multi-step ML pipeline orchestrated with Airflow

Passing data across tasks using XCom (Base64 + pickle)

Preprocessing using MinMaxScaler

Training a K-Means model for 1–50 clusters

Evaluating the Elbow Method using KneeLocator

Saving the trained model

Viewing logs and execution graph inside Airflow UI

### The DAG runs four automated tasks:

Load Dataset — read and serialize CSV

Preprocess Data — scale numeric fields

Train & Save K-Means Model — compute SSE for elbow

Elbow Method Evaluation — identify optimal cluster count
