import streamlit as st
from app.core.system import AutoMLSystem
import pickle

st.set_page_config(page_title="Deployment")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("Deployment")
write_helper_text("In this section, you can see saved pipelines.")

automl = AutoMLSystem.get_instance()

saved_pipe = automl.registry.list(type="pipeline")

if saved_pipe:
    st.write(f"Found {len(saved_pipe)} saved pipelines.")

if saved_pipe:
    for pipeline in saved_pipe:
        pipeline_data = pickle.loads(pipeline.data)
        st.write(f"Pipeline Name: {pipeline.name}")
        st.write(f"Version: {pipeline.version}")
        st.write(f"Model: {pipeline_data.get('model')}")
        st.write(f"Input Features: \
                 {', '.join(pipeline_data.get('input_features'))}")
        st.write(f"Target Feature: {pipeline_data.get('target_feature')}")
        st.write(f"Split Ratio: {pipeline_data.get('split_ratio')}")
        st.write(f"Metrics: {', '.join(pipeline_data.get('metrics'))}")
