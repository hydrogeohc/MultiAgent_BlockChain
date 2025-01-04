import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE

# Load the dataset
file_path = './forta_hacked_address_features.csv'
data = pd.read_csv(file_path)

# Create the 'avg val received' column
data['avg val received'] = data.apply(
    lambda row: row['Total Ether Received'] / row['Received Transactions'] if row['Received Transactions'] > 0 else 0,
    axis=1
)

# Separating features and labels
X = data.drop(columns=['Address', 'FLAG'])
y = data['FLAG']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Splitting the balanced data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced)

# Range of thresholds to test
thresholds = np.arange(0.1, 1.1, 0.1)

# Evaluate the model for each threshold
for threshold in thresholds:
    print(f"Evaluating model with threshold: {threshold:.1f}")
    
    # Predicting fraud based on threshold
    y_pred = (X_test['avg val received'] > threshold).astype(int)
    
    # Confusion Matrix and Classification Report
    print('Confusion Matrix:')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print('-' * 50)  # Separator for better readability
