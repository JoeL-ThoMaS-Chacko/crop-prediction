import IPython
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython import get_ipython
import warnings
warnings.filterwarnings("ignore")
data=pd.read_csv(r"E:\Crop_recommendation.csv")
data.head(5)                
crop_summary=pd.pivot_table(data,index=['label'],aggfunc='mean')
import plotly.express as px
fig = px.box(data, y='N',points="all")
fig.show()
fig = px.box(data, y='P',points="all")
fig.show()
fig = px.box(data, y='K',points="all")
fig.show()
fig = px.box(data, y='temperature',points="all")
fig.show()
fig = px.box(data, y='humidity',points="all")
fig.show()
fig = px.box(data, y='rainfall',points="all")
fig.show()
fig = px.box(data, y='ph',points="all")
fig.show()
df_boston = data.copy()

# List of columns
columns_to_process = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

print(f"Original shape: {df_boston.shape}")

# Create a mask to identify rows to keep
mask = pd.Series(True, index=df_boston.index)

for col in columns_to_process:
    print(f"\nProcessing: {col}")
    
    # Calculate quartiles
    q1 = np.percentile(df_boston[col], 25, interpolation='midpoint')
    q3 = np.percentile(df_boston[col], 75, interpolation='midpoint')
    IQR = q3 - q1
    
    # Update mask: keep only non-outliers
    mask = mask & (df_boston[col] >= (q1 - 1.5 * IQR)) & (df_boston[col] <= (q3 + 1.5 * IQR))
    
    print(f"  Q1: {q1:.2f}, Q3: {q3:.2f}")
    print(f"  Rows kept so far: {mask.sum()}")

# Apply the mask
df_boston_clean = df_boston[mask].copy()

print(f"\nFinal shape: {df_boston_clean.shape}")
print(f"Rows removed: {len(df_boston) - len(df_boston_clean)}")

# Visualize all columns after cleaning
for col in columns_to_process:
    fig = px.box(df_boston_clean, y=col, points="all", 
                 title=f'{col} - Final Cleaned Distribution')
    fig.show()
data=df_boston_clean
x=data.drop('label',axis=1)
y=data['label']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,shuffle=True,random_state=0)
import xgboost as xgb
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

# After your train-test split, add label encoding
from sklearn.preprocessing import LabelEncoder

# Encode the labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print(f"Training set: {x_train.shape}")
print(f"Testing set: {x_test.shape}")
print(f"Number of unique crops: {len(le.classes_)}")
print(f"Crop classes: {le.classes_}")

# Now train the model with ENCODED labels
model.fit(x_train, y_train_encoded)

# Make predictions
y_pred_encoded = model.predict(x_test)
y_pred = le.inverse_transform(y_pred_encoded)  # Convert back to crop names

# Evaluate
from sklearn.metrics import accuracy_score, classification_report

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2%}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
