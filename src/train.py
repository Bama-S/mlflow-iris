import os
import mlflow
import mlflow.sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ✅ Safe for GitHub Actions
mlflow.set_tracking_uri("file:" + os.path.abspath("mlruns"))
mlflow.set_experiment("Iris_Classifier")

# Load Iris data
iris = datasets.load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.3, random_state=42
)

with mlflow.start_run():
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    acc = accuracy_score(y_test, predictions)

    mlflow.sklearn.log_model(clf, "model")
    mlflow.log_metric("accuracy", acc)

    print(f"✅ Accuracy: {acc}")


