from flask import Flask, render_template, send_from_directory
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_curve, auc,
    confusion_matrix
)
import os

# Initialize Flask app
app = Flask(__name__)

# Directory to save outputs
OUTPUT_DIR = r"D:\VNIT_Mtech\Subjects\8.ML_Ops_Model_Deployment\Code"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Train a simple ML model and save outputs
def train_model_and_save_outputs():
    # Load dataset
    iris = load_iris()
    X = iris.data
    y = (iris.target != 0).astype(int)  # Binary classification

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Train a classifier
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    conf_matrix = confusion_matrix(y_test, y_pred)

    # Save ROC curve plot
    roc_path = os.path.join(OUTPUT_DIR, "roc_curve.png")
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(roc_path)
    plt.close()

    # Save confusion matrix plot (using Matplotlib instead of Seaborn)
    conf_matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["0", "1"])
    plt.yticks(tick_marks, ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    for i in range(2):
        for j in range(2):
            plt.text(j, i, f"{conf_matrix[i, j]}", horizontalalignment="center", color="white" if conf_matrix[i, j] > conf_matrix.max() / 2 else "black")
    plt.savefig(conf_matrix_path)
    plt.close()

    # Save metrics to a text file
    metrics_path = os.path.join(OUTPUT_DIR, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"Accuracy: {accuracy:.2f}\n")
        f.write(f"Precision: {precision:.2f}\n")
        f.write(f"Recall: {recall:.2f}\n")
        f.write(f"ROC AUC: {roc_auc:.2f}\n")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "roc_path": roc_path,
        "conf_matrix_path": conf_matrix_path,
        "metrics_path": metrics_path
    }

@app.route("/")
def index():
    # Train the model and save outputs
    results = train_model_and_save_outputs()

    # Generate a simple HTML response
    return f"""
    <h1>ML Model Evaluation</h1>
    <p>Accuracy: {results['accuracy']:.2f}</p>
    <p>Precision: {results['precision']:.2f}</p>
    <p>Recall: {results['recall']:.2f}</p>
    <h2>ROC Curve</h2>
    <img src="/{results['roc_path']}" alt="ROC Curve">
    <h2>Confusion Matrix</h2>
    <img src="/{results['conf_matrix_path']}" alt="Confusion Matrix">
    <h2>Metrics File</h2>
    <p><a href="/static/outputs/metrics.txt" download>Download Metrics File</a></p>
    """

if __name__ == "__main__":
    app.run(port=5003)
