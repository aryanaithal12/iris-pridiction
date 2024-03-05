
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.write("""
## A Model that predicts the type of iris based on inputs

This web app aims at predicting isri type """)

st.sidebar.write("""### User input parameters""")

def inputs():
    sl = st.sidebar.slider("Sepal length", 4.3, 7.9,5.0)
    sw = st.sidebar.slider("Sepal width", 2.0, 4.4,3.0)
    pl = st.sidebar.slider("Petal length", 1.0, 6.9,2.0)
    pw = st.sidebar.slider("Petal width", 0.1, 2.5,1.0)
    data = {"sepal_length": sl,
            "sepal_width": sw,
            "petal_length": pl,
            "petal_width": pw}
    df = pd.DataFrame(data, index=[0])
    return df


features = inputs()
features.reset_index(drop=False, inplace=False)
st.subheader("User inputs: ")
st.write(features)
iris = datasets.load_iris()
X, Y = iris.data, iris.target
clf = RandomForestClassifier()
clf.fit(X, Y)
prediction = clf.predict(features)
st.subheader("Possible iris types: ")
st.write(iris.target_names)
st.subheader("Prediction for user input: ")
st.write(iris.target_names[prediction])

st.subheader("Code:")
st.write("""
~~~
import streamlit as st
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

st.write("
## A Model that predicts the type of iris based on inputs

This web app aims at predicting isri type ")

st.sidebar.write("### User input parameters")

def inputs():
    sl = st.sidebar.slider("Sepal length", 4.3, 7.9,5.0)
    sw = st.sidebar.slider("Sepal width", 2.0, 4.4,3.0)
    pl = st.sidebar.slider("Petal length", 1.0, 6.9,2.0)
    pw = st.sidebar.slider("Petal width", 0.1, 2.5,1.0)
    data = {"sepal_length": sl,
            "sepal_width": sw,
            "petal_length": pl,
            "petal_width": pw}
    df = pd.DataFrame(data, index=[0])
    return df


features = inputs()
features.reset_index(drop=False, inplace=False)
st.subheader("User inputs: ")
st.write(features)
iris = datasets.load_iris()
X, Y = iris.data, iris.target
clf = RandomForestClassifier()
clf.fit(X, Y)
prediction = clf.predict(features)
st.subheader("Possible iris types: ")
st.write(iris.target_names)
st.subheader("Prediction for user input: ")
st.write(iris.target_names[prediction])

~~~""")
