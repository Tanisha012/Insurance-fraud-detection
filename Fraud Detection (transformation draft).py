#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install streamlit


# In[2]:


import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix


# In[3]:


# Title
st.title("Insurance Fraud Detection - Automated Machine Learning")


# In[4]:


# Sidebar Navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose an option:", ["Upload Data", "EDA", "Preprocessing", "Feature Engineering", "Modeling"])


# In[5]:


# Global Dataset
if "dataset" not in st.session_state:
    st.session_state["dataset"] = None


# In[6]:


# Upload Data
if options == "Upload Data":
    uploaded_file = st.file_uploader("insurance_claimsv2", type="csv")
    if uploaded_file:
        st.session_state["dataset"] = pd.read_csv(uploaded_file)
        st.write("Uploaded Dataset:")
        st.dataframe(st.session_state["dataset"].head())


# In[15]:


# EDA
if options == "EDA" and st.session_state["dataset"] is not None:
    dataset = st.session_state["dataset"]
    st.write("Exploratory Data Analysis")
    
    # Summary Statistics
    st.write("Basic Information")
    st.write(dataset.describe())
    
     # Correlation Heatmap
    if st.button("Correlation Heatmap"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(dataset.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        st.pyplot(fig)
        
    # Target Distribution
    target_col = st.selectbox("Select Target Variable for Distribution:", dataset.columns)
    if target_col:
        fig, ax = plt.subplots()
        sns.countplot(data=dataset, x=target_col, ax=ax)
        st.pyplot(fig)


# In[16]:


# Preprocessing
if options == "Preprocessing" and st.session_state["dataset"] is not None:
    dataset = st.session_state["dataset"]

    # Handle Missing Values
    st.write("Handle Missing Values")
    method = st.radio("Select Imputation Method:", ["Drop Rows", "Fill with Mean", "Fill with Median"])
    if method == "Drop Rows":
        dataset.dropna(inplace=True)
    elif method == "Fill with Mean":
        dataset.fillna(dataset.mean(), inplace=True)
    elif method == "Fill with Median":
        dataset.fillna(dataset.median(), inplace=True)

    # Scale Numerical Variables
    if st.checkbox("Scale Numerical Variables"):
        num_cols = st.multiselect("Select Numerical Columns to Scale:", dataset.select_dtypes(include=["float64", "int64"]).columns)
        scaler = StandardScaler()
        dataset[num_cols] = scaler.fit_transform(dataset[num_cols])

    st.session_state["dataset"] = dataset
    st.write("Preprocessed Data")
    st.dataframe(dataset.head())


# In[17]:


# Feature Engineering
if options == "Feature Engineering" and st.session_state["dataset"] is not None:
    dataset = st.session_state["dataset"]

    # Create Trend Variables
    st.write("Create Trend Variables")
    dataset["claim_avg"] = dataset[["injury_claim", "property_claim", "vehicle_claim"]].mean(axis=1)
    dataset["claim_max"] = dataset[["injury_claim", "property_claim", "vehicle_claim"]].max(axis=1)
    dataset["claim_min"] = dataset[["injury_claim", "property_claim", "vehicle_claim"]].min(axis=1)

    # WOE Encoding
    st.write("WOE Encoding for Categorical Variables")
    target_col = st.selectbox("Select Target Variable for WOE Encoding:", dataset.columns)
    cat_cols = st.multiselect("Select Categorical Columns for WOE Encoding:", dataset.select_dtypes(include=["object"]).columns)

    def calculate_woe(data, feature, target):
        grouped = data.groupby(feature)[target].agg(['sum', 'count'])
        grouped['non_event'] = grouped['count'] - grouped['sum']
        grouped['event_rate'] = grouped['sum'] / grouped['count']
        grouped['non_event_rate'] = grouped['non_event'] / grouped['count']
        grouped['WOE'] = np.log(grouped['event_rate'] / grouped['non_event_rate'])
        return grouped['WOE']

    for col in cat_cols:
        dataset[f"{col}_woe"] = dataset[col].map(calculate_woe(dataset, col, target_col))

    st.session_state["dataset"] = dataset
    st.write("Feature Engineered Data")
    st.dataframe(dataset.head())


# In[18]:


# Modeling
if options == "Modeling" and st.session_state["dataset"] is not None:
    dataset = st.session_state["dataset"]

    # Train-Test Split
    target_col = st.selectbox("Select Target Variable for Modeling:", dataset.columns)
    X = dataset.drop(columns=[target_col])
    y = dataset[target_col]
    test_size = st.slider("Select Test Size (in %):", 10, 50, 20) / 100
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # Model Selection
    model_type = st.selectbox("Select Model Type:", ["Random Forest"])
    if model_type == "Random Forest":
        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

    # Model Evaluation
    if st.button("Train and Evaluate Model"):
        y_pred = model.predict(X_test)
        st.write("Classification Report")
        st.text(classification_report(y_test, y_pred))

        st.write("Confusion Matrix")
        fig, ax = plt.subplots()
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
        st.pyplot(fig)


# In[ ]:




