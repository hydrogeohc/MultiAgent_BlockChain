import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

# Load the dataset
file_path = './forta_hacked_address_features.csv'
data = pd.read_csv(file_path)

# Preprocessing data
X = data.drop(columns=['Address', 'FLAG'])  # Features
y = data['FLAG']  # Target label

# Standardize numerical features to improve model performance
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Compute class weights to handle imbalance
classes = y.unique()  # Unique classes (0 and 1)
class_weights = compute_class_weight(class_weight='balanced', classes=classes, y=y)
class_weight_dict = {cls: weight for cls, weight in zip(classes, class_weights)}
print("Class Weights:", class_weight_dict)

# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Initialize Random Forest Classifier with class weights
rf_classifier = RandomForestClassifier(random_state=42, class_weight=class_weight_dict)

# Fit the model
rf_classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=1)
recall = recall_score(y_test, y_pred, zero_division=1)
f1 = f1_score(y_test, y_pred, zero_division=1)

# Print evaluation metrics
print(f'Accuracy: {accuracy:.2f}')
print(f'Precision: {precision:.2f}')
print(f'Recall: {recall:.2f}')
print(f'F1 Score: {f1:.2f}')
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=1))

# Optional: Feature importance
feature_importance = pd.DataFrame({
    'feature': data.drop(columns=['Address', 'FLAG']).columns,
    'importance': rf_classifier.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))
