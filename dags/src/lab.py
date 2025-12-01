import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64

# -----------------------------------------------------------
# LOAD DATA
# -----------------------------------------------------------
def load_data():
    """
    Loads Wholesale Customers dataset and returns base64-serialized data.
    """
    print("Loading Wholesale Customers dataset...")

    # Path: dags/data/wholesale_customers.csv
    csv_path = os.path.join(os.path.dirname(__file__), "../data/wholesale customers data.csv")
    df = pd.read_csv(csv_path)

    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


# -----------------------------------------------------------
# PREPROCESSING
# -----------------------------------------------------------
def data_preprocessing(data_b64: str):
    """
    Decode → DF → drop missing → scale selected numeric features → return serialized data.
    """

    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    # clean
    df = df.dropna()

    # Wholesale Customers numerical columns
    features = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]

    clustering_data = df[features]

    # Normalize using MinMaxScaler
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    serialized_scaled = pickle.dumps(scaled_data)
    return base64.b64encode(serialized_scaled).decode("ascii")


# -----------------------------------------------------------
# K-MEANS + SAVE MODEL
# -----------------------------------------------------------
def build_save_model(data_b64: str, filename: str):
    """
    Build KMeans model for k=1..20, compute SSE, save the final KMeans model.
    """
    data_bytes = base64.b64decode(data_b64)
    X = pickle.loads(data_bytes)

    sse = []
    kmeans_kwargs = {"init": "k-means++", "n_init": 10, "max_iter": 300, "random_state": 42}

    # test K from 1 to 20
    for k in range(1, 21):
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Save the model for k=5 (reasonable for Wholesale dataset)
    final_model = KMeans(n_clusters=5, **kmeans_kwargs)
    final_model.fit(X)

    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, filename)
    with open(output_path, "wb") as f:
        pickle.dump(final_model, f)

    return sse


# -----------------------------------------------------------
# ELBOW METHOD
# -----------------------------------------------------------
def load_model_elbow(filename: str, sse: list):
    """
    Loads saved model, computes elbow, prints optimal cluster count.
    """

    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    loaded_model = pickle.load(open(model_path, "rb"))

    kl = KneeLocator(range(1, 21), sse, curve="convex", direction="decreasing")
    print(f"Optimal number of clusters (Elbow Method): {kl.elbow}")

    return kl.elbow if kl.elbow is not None else -1
