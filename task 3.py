import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# New import for reliable UCI data fetching
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# Set plotting style
sns.set_style('whitegrid')

# NOTE: You must install the ucimlrepo library if you haven't already:
# pip install ucimlrepo

# --- 1. Data Loading (Using ucimlrepo) ---
try:
    # Fetch Bank Marketing dataset (ID 222)
    bank_marketing = fetch_ucirepo(id=222)
    
    # Extract features (X) and target (y)
    X = bank_marketing.data.features
    y = bank_marketing.data.targets
    
    # Combine them back into a single DataFrame for easier cleaning and feature engineering
    df = pd.concat([X, y], axis=1)
    df.rename(columns={df.columns[-1]: 'y'}, inplace=True)

    print("Dataset loaded successfully using ucimlrepo.")
    
except Exception as e:
    print(f"Error loading dataset using ucimlrepo: {e}")
    print("Please ensure the 'ucimlrepo' library is installed and you have an active internet connection.")
    exit()

# --- 2. Data Cleaning and Preprocessing ---

print("\n--- Initial Data Head and Info ---")
print(df.head())
print("\nDataset shape:", df.shape)
# print(df.info(verbose=False))

# Identify categorical features and the target variable
categorical_features = df.select_dtypes(include='object').columns.tolist()

# Drop the 'duration' column (as per best practice for this dataset)
# 'duration' should not be used in the model, as it is only measured after the call is made.
df.drop('duration', axis=1, inplace=True) 

# Re-identify categorical features after dropping 'duration'
categorical_features = df.select_dtypes(include='object').columns.tolist()
# Remove the target variable 'y' for preprocessing features
categorical_features.remove('y') 

# A. Handle 'unknown' values in categorical columns by replacing them with the mode
for col in categorical_features:
    mode_value = df[col].mode()[0]
    df[col] = df[col].replace('unknown', mode_value)

# B. Convert binary categorical features ('yes'/'no') to 0/1 using LabelEncoder
# Note: The 'default' column in this dataset version mostly contains 'no' or 'unknown' (now mode-filled)
le = LabelEncoder()
df['default'] = le.fit_transform(df['default'])
df['housing'] = le.fit_transform(df['housing'])
df['loan'] = le.fit_transform(df['loan'])

# C. One-Hot Encoding for remaining nominal categorical features
# Drop the original categorical columns that were not binary encoded
cols_to_drop_before_ohe = ['default', 'housing', 'loan', 'y'] 
ohe_cols = [col for col in categorical_features if col not in cols_to_drop_before_ohe]
df_processed = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

# D. Encode the target variable 'y' ('yes'/'no' -> 1/0)
# Note: ucimlrepo returns the target as 'y', but it needs to be mapped to 0/1
df_processed['y'] = df_processed['y'].astype(str).str.lower().map({'yes': 1, 'no': 0})
df_processed.drop('day_of_week', axis=1, inplace=True) # Dropping as before

print("\n--- Processed Data Snapshot ---")
print(df_processed.head())
print(f"Processed dataset shape: {df_processed.shape}")

# --- 3. Model Training (Decision Tree) ---

# Define features (X) and target (y)
X = df_processed.drop('y', axis=1)
y = df_processed['y']

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Initialize and train the Decision Tree Classifier
# Using max_depth to prevent overfitting and improve interpretability
dt_classifier = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_classifier.fit(X_train, y_train)

# --- 4. Evaluation ---

# Make predictions on the test set
y_pred = dt_classifier.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n--- Model Performance Metrics ---")
print(f"Accuracy: {accuracy:.4f}")

# Print detailed classification report (Precision, Recall, F1-Score)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- 5. Visualization: Feature Importance ---

# Extract feature importances
feature_importances = pd.Series(dt_classifier.feature_importances_, index=X.columns)
# Get top 10 most important features
top_10_features = feature_importances.nlargest(10)

plt.figure(figsize=(10, 6))
sns.barplot(x=top_10_features.values, y=top_10_features.index, palette='crest')
plt.title('Top 10 Feature Importances from Decision Tree')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

print("\n--- Key Findings from Feature Importance ---")
print(f"The top 3 most important features are:")
for i, (feature, score) in enumerate(zip(top_10_features.index[:3], top_10_features.values[:3])):
    print(f"{i+1}. {feature} (Score: {score:.4f})")