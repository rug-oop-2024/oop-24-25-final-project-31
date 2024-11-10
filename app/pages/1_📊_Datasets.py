import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_openml
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


automl = AutoMLSystem.get_instance()

registered_datasets = automl.registry.list(type="dataset")

iris = fetch_openml("iris", version=1, parser="auto")
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
st.success("Iris dataset was registered")

dataset_list = [ds.name for ds in registered_datasets]
selected_dataset = st.selectbox("Select a dataset to showcase:", dataset_list)
if selected_dataset:
    st.write(f"Dataset selected: {selected_dataset}")
#task = st.selectbox('question', [options])
#split = st.slider('question', 0.0, 1.0, 0.5)
# ask for set of metrics: metrics = st.multiselect('question', [all metrics])
#st.selectbox('question', ['classification', 'regression'])
