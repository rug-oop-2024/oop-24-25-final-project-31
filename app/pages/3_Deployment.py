import streamlit as st
from app.core.system import AutoMLSystem
import pickle

st.set_page_config(page_title="Deployment")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("Deployment")
write_helper_text("In this section, you can see saved pipelines.")


class DeploymentPage:
    def __init__(self):
        """Initialize the AutoML system and retrieve saved pipelines."""
        self.automl = AutoMLSystem.get_instance()
        self.saved_pipe = self.automl.registry.list(type="pipeline")

    def run(self):
        """Show saved pipelines and their relevant information."""
        if self.saved_pipe:
            st.write(f"Found {len(self.saved_pipe)} saved pipelines.")

        if self.saved_pipe:
            for pipeline in self.saved_pipe:
                pipeline_data = pickle.loads(pipeline.data)
                st.write(f"Pipeline Name: {pipeline.name}")
                st.write(f"Version: {pipeline.version}")
                st.write(f"Model: {pipeline_data.get('model')}")
                st.write(f"Input Features: \
                        {', '.join(pipeline_data.get('input_features'))}")
                st.write(f"Target Feature: \
                         {pipeline_data.get('target_feature')}")
                st.write(f"Split Ratio: {pipeline_data.get('split_ratio')}")
                st.write(f"Metrics: {', '.join(pipeline_data.get('metrics'))}")


if __name__ == "__main__":
    page = DeploymentPage()
    page.run()
