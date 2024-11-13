import numpy as np
from sklearn.metrics import confusion_matrix

class EthicsCheckerTool:
    name = "EthicsCheckerTool"
    def __init__(self, data):
        self.data = data

    def check_fairness(self, predictions):
        true_labels = self.data['FLAG']
        cm = confusion_matrix(true_labels, predictions)

        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (fn + tp)

        fairness_score = 1 - abs(fpr - fnr)
        return fairness_score

    def check_bias(self, feature_importances):
        top_features = sorted(zip(feature_importances, self.data.columns[1:-1]), reverse=True)[:5]
        bias_report = "Top 5 influential features:\n"
        for importance, feature in top_features:
            bias_report += f"{feature}: {importance:.4f}\n"
        return bias_report