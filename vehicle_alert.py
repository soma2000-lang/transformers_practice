import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the data (you'll need to replace this with your actual data loading method)
data = pd.read_csv('alerts_data.csv')

# Preprocess the data
le = LabelEncoder()
data['severity'] = le.fit_transform(data['severity'])

# Define features and target
X = data.drop(['severity', 'alert_id'], axis=1)  # Adjust column names as needed
y = data['severity']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

def classify_alert(alert_data):
    """
    Classify a single alert based on its features.
    
    :param alert_data: A dictionary containing the alert features
    :return: A tuple containing the predicted severity and priority
    """
    # Convert the alert data to a DataFrame
    alert_df = pd.DataFrame([alert_data])
    
    # Make sure the columns match the training data
    alert_df = alert_df.reindex(columns=X.columns, fill_value=0)
    
    # Predict the severity
    severity_pred = model.predict(alert_df)[0]
    severity_label = le.inverse_transform([severity_pred])[0]
    
    # Assign priority based on severity
    if severity_label == 'Critical':
        priority = 1
    elif severity_label == 'High':
        priority = 2
    elif severity_label == 'Medium':
        priority = 3
    else:
        priority = 4
    
    return severity_label, priority

# Example usage
new_alert = {
    'feature1': 0.5,
    'feature2': 1,
    'feature3': 0.2,
    # Add other features as needed
}

predicted_severity, predicted_priority = classify_alert(new_alert)
print(f"Predicted Severity: {predicted_severity}")
print(f"Predicted Priority: {predicted_priority}")