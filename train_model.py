import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib

# Create the models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# Load the dataset
df = pd.read_csv('C:\\Users\\Ali\\PycharmProjects\\python\\drug_consumption.csv')

# Preprocess the data
columns = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 
           'Nscore', 'Escore', 'Oscore', 'Ascore', 'Cscore', 'Impulsive', 'SS', 
           'Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 
           'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 
           'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

df.columns = columns

# Map drug consumption levels to numerical values
drug_columns = ['Alcohol', 'Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 
                'Coke', 'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 
                'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']

drug_mapping = {
    'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6
}

for col in drug_columns:
    df[col] = df[col].map(drug_mapping)

# Convert categorical features to numerical values
categorical_columns = ['Gender', 'Education', 'Country', 'Ethnicity']
df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

# Example: Predicting Cannabis usage (1 if user, 0 otherwise)
df['Cannabis'] = df['Cannabis'].apply(lambda x: 1 if x > 0 else 0)

X = df.drop(['ID', 'Cannabis'], axis=1)
y = df['Cannabis']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train models
models = {
    'LogisticRegression': LogisticRegression(),
    'DecisionTree': DecisionTreeClassifier(),
    'RandomForest': RandomForestClassifier()
}

metrics = {
    'Model': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1 Score': []
}

for name, model in models.items():
    model.fit(X_train, y_train)
    joblib.dump(model, f'models/{name}.pkl')
    
    # Predicting and calculating metrics
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    metrics['Model'].append(name)
    metrics['Accuracy'].append(accuracy)
    metrics['Precision'].append(precision)
    metrics['Recall'].append(recall)
    metrics['F1 Score'].append(f1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

# Displaying the metrics
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Plotting the metrics
metrics_df.set_index('Model', inplace=True)
metrics_df.plot(kind='bar', figsize=(12, 8))
plt.title('Model Performance Metrics')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.legend(loc='best')
plt.tight_layout()
plt.show()

print("Models trained and saved.")
