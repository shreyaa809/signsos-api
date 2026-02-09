import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib

# Load data
data = pd.read_csv("gesture_data.csv", header=None)

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = SVC(kernel="rbf")
model.fit(X_train, y_train)

# Test accuracy
predictions = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, predictions))

# Save model
joblib.dump(model, "gesture_model.pkl")
