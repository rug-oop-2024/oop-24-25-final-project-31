import streamlit as st
import pandas as pd
from sklearn import datasets
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here

iris = datasets.fetch_openml("iris", version=1, parser="auto")
df = pd.DataFrame(
            iris.data,
            columns=iris.feature_names,
        )
dataset = Dataset.from_dataframe(
            name="iris",
            asset_path="iris.csv",
            data=df,
        )
automl.registry.register(dataset)


# ask for the task type:  task = st.selectbox('question', [options])
# ask for split:  split = st.slider('quesstion', 0.0, 1.0, 0.5)
# ask for set of metrics: metrics = st.multiselect('question', [all metrics]) 
# or st.selectbox('question', ['classification', 'regression'])
