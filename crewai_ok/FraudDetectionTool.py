from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

class FraudDetectionTool:
    name = "FraudDetectionTool"

    def __init__(self, algorithm, data):
        self.algorithm = algorithm
        self.data = data

    def detect_fraud(self):
        X = self.data.drop(['Address', 'FLAG'], axis=1)
        y = self.data['FLAG']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if self.algorithm == "A":
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elif self.algorithm == "B":
            model = RandomForestClassifier(n_estimators=500, max_depth=20, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
        elif self.algorithm == "C":
            contamination = 0.1  # Adjust this value based on your dataset
            model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
            model.fit(X_train)
            y_pred = model.predict(X_test)
            # Convert predictions to match the format of other algorithms
            y_pred = np.where(y_pred == 1, 0, 1)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        return {
            "algorithm": self.algorithm,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }