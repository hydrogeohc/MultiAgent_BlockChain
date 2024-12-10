import json
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from pydantic import Field

class EthicsCheckerTool:
    name: str = "EthicsCheckerTool"
    description: str = "Evaluate fairness, bias, and performance of fraud detection results. Input must be a JSON string."
    data: Any = Field(default=None, exclude=True)

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.le = LabelEncoder()
        self.data['FLAG'] = self.le.fit_transform(self.data['FLAG'])
        self.fraud_criteria = self.define_fraud_criteria()

    class Config:
        arbitrary_types_allowed = True

    def define_fraud_criteria(self):
        return {
            'high_value_received': self.data['max value received'].quantile(0.95),
            'high_transaction_count': self.data['total transactions (including tnx to create contract'].quantile(0.95),
            'suspicious_time_pattern': self.data['Time Diff between first and last (Mins)'].quantile(0.05)
        }

    def _run(self, results: str):
        try:
            # Parse the input JSON string
            results_dict = json.loads(results)
            predictions = np.array(results_dict.get('predictions', []))
            feature_importances = results_dict.get('feature_importances', [])

            # Perform evaluations
            fairness_metrics = self.check_fairness(predictions)
            bias_report = self.check_bias(feature_importances)
            performance_metrics = self.check_performance(predictions)
            transparency_report = self.generate_transparency_report()

            # Format the output
            return self.format_output(fairness_metrics, bias_report, performance_metrics, transparency_report)
        except json.JSONDecodeError:
            return "Error: Invalid input format. Please provide a valid JSON string."
        except ValueError as e:
            return f"Error: {str(e)}"

    @property
    def args_schema(self):
        return {
            "results": {
                "description": "JSON string containing predictions and feature importances",
                "type": "str"
            }
        }

    def check_fairness(self, predictions):
        true_labels = self.data['FLAG']
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        fairness_metrics = {
            'Equal Opportunity': 1 - abs(fpr - fnr),
            'Predictive Parity': 1 - abs(ppv - npv)
        }
        return fairness_metrics

    def check_bias(self, feature_importances):
        top_features = sorted(zip(feature_importances, self.data.columns[1:-1]), reverse=True)[:10]
        return {
            'top_features': top_features,
            'feature_correlation': self.calculate_feature_correlation()
        }

    def calculate_feature_correlation(self):
        corr_matrix = self.data.corr()
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if abs(corr_matrix.iloc[i, j]) > 0.8:
                    high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j]))
        return high_corr_pairs

    def check_performance(self, predictions):
        true_labels = self.data['FLAG']
        auc_roc = roc_auc_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        return {
            'AUC-ROC': auc_roc,
            'Accuracy': (tp + tn) / (tp + tn + fp + fn),
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'False Positive Rate': fpr,
            'False Negative Rate': fnr
        }

    def generate_transparency_report(self):
        return {
            'fraud_criteria': self.fraud_criteria,
            'data_distribution': self.data.describe().to_dict(),
            'missing_values': self.data.isnull().sum().to_dict()
        }

    def format_output(self, fairness_metrics, bias_report, performance_metrics, transparency_report):
        output = "## Fairness Metrics\n"
        for metric, value in fairness_metrics.items():
            output += f"- {metric}: {value:.4f}\n"
        output += "\n## Bias Report\n"
        output += "**Top 10 influential features:**\n"
        for importance, feature in bias_report['top_features']:
            output += f"- {feature}: {importance:.4f}\n"
        output += "\n**High correlation feature pairs:**\n"
        for feat1, feat2, corr in bias_report['feature_correlation']:
            output += f"- {feat1} & {feat2}: {corr:.4f}\n"
        output += "\n## Performance Metrics\n"
        for metric, value in performance_metrics.items():
            output += f"- {metric}: {value:.4f}\n"
        output += "\n## Transparency Report\n"
        output += "**Fraud Detection Criteria:**\n"
        for criterion, value in transparency_report['fraud_criteria'].items():
            output += f"- {criterion}: {value:.4f}\n"
        output += "\n**Data Distribution Summary:**\n"
        for column, stats in transparency_report['data_distribution'].items():
            output += f"- {column}:\n"
            for stat, value in stats.items():
                output += f" - {stat}: {value:.4f}\n"
        output += "\n**Missing Values:**\n"
        for column, count in transparency_report['missing_values'].items():
            output += f"- {column}: {count}\n"
        return output