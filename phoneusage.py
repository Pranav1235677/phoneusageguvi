import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE
import optuna
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Phone Usage Prediction", layout="wide")

# Cache Data Loading
@st.cache_data
def load_data():
    df = pd.read_csv('phone_usage_india.csv')
    return df

df = load_data()

# Data Cleaning Function
def clean_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)

    df.rename(columns={
        'Screen Time (hrs/day)': 'Screen Time',
        'Data Usage (GB/month)': 'Data Usage',
        'Calls Duration (mins/day)': 'Calls Duration',
        'Social Media Time (hrs/day)': 'Social Media Time',
        'Streaming Time (hrs/day)': 'Streaming Time',
        'Gaming Time (hrs/day)': 'Gaming Time',
        'E-commerce Spend (INR/month)': 'E-commerce Spend',
        'Monthly Recharge Cost (INR)': 'Monthly Recharge Cost'
    }, inplace=True)

    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df

df = clean_data(df)

# Target Column
if 'Target' not in df.columns:
    df['Target'] = np.where((df['Screen Time'] > 5) & (df['Data Usage'] > 2 & (df['Gaming Time'] > 3) &(df['Streaming Time'] > 3) ), 1, 0)

# SMOTE for Class Imbalance
X = df.drop(columns=['Target'])
y = df['Target']
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Feature Scaling
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Feature Selection
selector = RFE(estimator=RandomForestClassifier(), n_features_to_select=10)
X_resampled = selector.fit_transform(X_resampled, y_resampled)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Cache the trained model
@st.cache_resource
def train_model():
    def objective(trial):
        model_type = trial.suggest_categorical("model_type", ["RandomForest", "XGBoost"])
        
        if model_type == "RandomForest":
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 12)
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
        
        else:
            n_estimators = trial.suggest_int('n_estimators', 50, 300)
            max_depth = trial.suggest_int('max_depth', 3, 12)
            learning_rate = trial.suggest_float('learning_rate', 0.01, 0.3)
            model = XGBClassifier(n_estimators=n_estimators, max_depth=max_depth, learning_rate=learning_rate, random_state=42)

        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return accuracy_score(y_test, preds)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    best_params = study.best_params

    models = {
        'Random Forest': RandomForestClassifier(n_estimators=best_params.get('n_estimators', 100), 
                                                max_depth=best_params.get('max_depth', None), 
                                                random_state=42),
        'XGBoost': XGBClassifier(n_estimators=best_params.get('n_estimators', 100), 
                                 max_depth=best_params.get('max_depth', 3), 
                                 learning_rate=best_params.get('learning_rate', 0.1), 
                                 random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

models = train_model()

# EDA Section
st.title("📊 Exploratory Data Analysis (EDA)")

fig1, ax1 = plt.subplots(figsize=(8, 5))
sns.histplot(df['Screen Time'], bins=20, kde=True, ax=ax1)
ax1.set_title("Distribution of Screen Time")
st.pyplot(fig1)

fig2, ax2 = plt.subplots(figsize=(8, 5))
sns.boxplot(x=df['Data Usage'], ax=ax2)
ax2.set_title("Boxplot of Data Usage")
st.pyplot(fig2)

fig3, ax3 = plt.subplots(figsize=(8, 5))
sns.scatterplot(x=df['Screen Time'], y=df['Data Usage'], hue=df['Target'], ax=ax3)
ax3.set_title("Screen Time vs Data Usage")
st.pyplot(fig3)

fig4, ax4 = plt.subplots(figsize=(15, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax4)
ax4.set_title("Correlation Matrix")
st.pyplot(fig4)

fig5, ax5 = plt.subplots(figsize=(8, 5))
sns.barplot(x=df['Target'].value_counts().index, y=df['Target'].value_counts().values, ax=ax5)
ax5.set_title("Class Distribution of Target Variable")
st.pyplot(fig5)

# Sidebar for Instant Prediction
st.sidebar.subheader("🔮 Make a Prediction")

# Dictionary for categorical mappings
gender_map = {0: "Male", 1: "Female"}
phone_brand_map = {1: "Apple", 2: "Samsung", 3: "OnePlus", 4: "Xiaomi", 5: "Realme"}

# Initialize session state to store user inputs and predictions
if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = {col: df[col].median() for col in X.columns}

# Get user inputs
for col in X.columns:
    if col == "Gender":
        st.session_state.user_inputs[col] = st.sidebar.selectbox(f"Select {col}", options=[0, 1], 
                                                                  format_func=lambda x: gender_map[x], 
                                                                  index=int(df[col].median()))
    elif col == "Phone Brand":
        st.session_state.user_inputs[col] = st.sidebar.selectbox(f"Select {col}", options=[1, 2, 3, 4, 5], 
                                                                  format_func=lambda x: phone_brand_map[x], 
                                                                  index=int(df[col].median()))
    else:
        st.session_state.user_inputs[col] = st.sidebar.number_input(f"Enter {col}", value=float(df[col].median()))

# Prediction Section (Instant)
input_df = pd.DataFrame([st.session_state.user_inputs])
input_scaled = scaler.transform(input_df)
input_selected = selector.transform(input_scaled)

# Store the model choice in session state
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Random Forest"

st.session_state.selected_model = st.sidebar.radio("Select Model:", list(models.keys()), index=0)

# Instant Prediction
prediction = models[st.session_state.selected_model].predict(input_selected)[0]
confidence = np.max(models[st.session_state.selected_model].predict_proba(input_selected)) * 100 if hasattr(models[st.session_state.selected_model], "predict_proba") else 100

# Display Prediction Instantly
st.sidebar.write(f"🔮 *Prediction:* {'High Usage' if prediction == 1 else 'Low Usage'}")
st.sidebar.write(f"🎯 *Confidence:* {confidence:.2f}%")