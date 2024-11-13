import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
file_path = 'address_data_features_combined.csv'
data = pd.read_csv(file_path)

# Separating features and labels
X = data.drop(columns=['Address', 'FLAG'])
y = data['FLAG']

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Implementing thresholding-based model
threshold = 0.1  # Example threshold

# Predicting fraud based on threshold
y_pred = (X_test['avg val received'] > threshold).astype(int)

# Evaluating the model
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))