import streamlit as st
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

@st.cache_data
def load_data():
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = iris.target
    return data, iris.target_names

data, target_names = load_data()

# Train a RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(data.iloc[:, :-1], data['species'])

# Make a prediction (for example purposes, we'll use the first sample)
sample = data.iloc[0, :-1].values.reshape(1, -1)
predicted_probabilities = clf.predict_proba(sample)[0]

# Convert probabilities to percentages
predicted_percentages = predicted_probabilities * 100

# Display the predicted species as percentages
for species, percentage in zip(target_names, predicted_percentages):
    st.write(f"{species}: {percentage:.2f}%")