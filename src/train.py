import os
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

# Point to a known folder for artifacts
mlflow.set_tracking_uri("file:///tmp/mlruns")
mlflow.set_experiment("Iris_Classifier")

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    mlflow.log_metric("accuracy", acc)

    # Save model
    mlflow.sklearn.log_model(clf, "model")

    print(f"✅ Model saved in run {mlflow.active_run().info.run_id}")
