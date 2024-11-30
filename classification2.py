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

# Sidebar inputs for sepal and petal dimensions
st.sidebar.header('Input Features')
sepal_length = st.sidebar.slider('Sepal Length (cm)', float(data['sepal length (cm)'].min()), float(data['sepal length (cm)'].max()), float(data['sepal length (cm)'].mean()))
sepal_width = st.sidebar.slider('Sepal Width (cm)', float(data['sepal width (cm)'].min()), float(data['sepal width (cm)'].max()), float(data['sepal width (cm)'].mean()))
petal_length = st.sidebar.slider('Petal Length (cm)', float(data['petal length (cm)'].min()), float(data['petal length (cm)'].max()), float(data['petal length (cm)'].mean()))
petal_width = st.sidebar.slider('Petal Width (cm)', float(data['petal width (cm)'].min()), float(data['petal width (cm)'].max()), float(data['petal width (cm)'].mean()))

# Train a RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(data.iloc[:, :-1], data['species'])

# Make a prediction based on user input
sample = [[sepal_length, sepal_width, petal_length, petal_width]]
predicted_probabilities = clf.predict_proba(sample)[0]

# Convert probabilities to percentages
predicted_percentages = predicted_probabilities * 100

# Display the predicted species as percentages
st.write("### Predicted Species Probabilities")
for species, percentage in zip(target_names, predicted_percentages):
    st.write(f"{species}: {percentage:.2f}%")