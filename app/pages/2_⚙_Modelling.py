import streamlit as st
from sklearn.pipeline import Pipeline


from autoop.functional.feature import detect_feature_types


from autoop.core.ml.model.regression.multiple_linear_regression import (
    MultipleLinearRegression,
)
from autoop.core.ml.model.regression.lasso_regression import LassoRegression
from autoop.core.ml.model.regression.ridge_regression import RidgeRegression

from autoop.core.ml.model.classification.k_neighbours_classifier import (
    KNearestClassifier,
)
from autoop.core.ml.model.classification.support_vector import (
    SupportVectorClassifier,
)
from autoop.core.ml.model.classification.decision_tree import DTreeClassifier


from autoop.core.ml.metric import get_metric, METRICS
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset


st.set_page_config(page_title="Modelling", page_icon="📈")


def write_helper_text(text: str):
    st.write(f"<p style=\"color: #888;\">{text}</p>", unsafe_allow_html=True)


st.write("# ⚙ Modelling")
write_helper_text("In this section, you can design a machine learning pipeline\
                  to train a model on a dataset.")

automl = AutoMLSystem.get_instance()

datasets = automl.registry.list(type="dataset")

# your code here
# ST/modelling/datasets/list: Load existing datasets using the artifact
# registry. You can use a select box to achieve this.
st.header("Step 1: Load the Dataset")

dataset_list = [ds['name'] for ds in datasets]
selected_dataset = st.selectbox("Select a Dataset:", dataset_list)

if selected_dataset:
    dataset = next(ds for ds in datasets if ds['name'] == selected_dataset)
    data = Dataset(
        name=dataset['name'],
        asset_path=dataset['asset_path'],
        version=dataset['version'],
        data=dataset.get('data')
    )
    df = data.read()
    st.write(df)

    # ST/modelling/datasets/features: Detect the features and generate a
    # selection menu for selecting the input features (many)
    # and one target feature.
    # Based on the feature selections, prompt the user with the detected task
    # type (i.e., classification or regression).

    st.header("Step 2: Feature Selection and task type detection")

    # detect features
    features = detect_feature_types(data)
    numerical_features = [f.name for f in features if f.type == "numerical"]
    categorical_features = [f.name for f
                            in features if f.type == "categorical"]

    input_features = st.multiselect("Select input features",
                                    numerical_features + categorical_features)
    target_feature = st.selectbox("Select target feature:", df.columns)

    # detect task type
    if target_feature in numerical_features:
        task_type = "regression"
    else:
        task_type = "classification"

    st.write(f"Detected Task Type: {task_type}")

    # ST/modelling/models: Prompt the user to select a model
    # based on the task type.

    st.header("Step 3: Model selection")

    if task_type == "regression":
        model_type = st.selectbox("Select Regression Model: ",
                                  [LassoRegression(),
                                   RidgeRegression(),
                                   MultipleLinearRegression()])
        if model_type == "Lasso Regression":
            model = LassoRegression()
        elif model_type == "Ridge Regression":
            model = RidgeRegression()
        elif model_type == "Multiple Linear Regression":
            model = MultipleLinearRegression()
    else:
        model_type = st.selectbox("Select Classification Model: ",
                                  [KNearestClassifier(),
                                   SupportVectorClassifier(),
                                   DTreeClassifier()])
        if model_type == "K-Nearest Neighbors":
            model = KNearestClassifier()
        elif model_type == "Support Vector Classifier":
            model = SupportVectorClassifier()
        elif model_type == "Decision Tree Classifier":
            model = DTreeClassifier()

    # ST/modelling/pipeline/split: Prompt the user to select a dataset split.

    st.header("Step 4: Splitting the data")
    test_size = st.slider("Test Set Proportion", 0.1, 0.5, 0.2)

    # ST/modelling/pipeline/metrics: Prompt the user to select
    # a set of compatible metrics.

    st.header("Step 5: Select Metrics")

    if task_type == "regression":
        selected_metric = st.selectbox("Select Metric: ", METRICS[:3])
    else:
        selected_metric = st.selectbox("Select Metric: ", METRICS[3:])

    metric = get_metric(selected_metric)
    st.write(f"Selected Metric: {metric}")

    # ST/modelling/pipeline/summary: Prompt the user with a beautifuly
    # formatted pipeline summary with all the configurations.

    st.header("Step 6: Pipeline Summary")

    pipeline = Pipeline(
            metrics=[metric],
            dataset=data,
            model=model_type,
            input_features=input_features,
            target_feature=target_feature,
            split=test_size
        )

    if st.button("Show Pipeline Summary"):
        st.write(pipeline)

    # ST/modelling/pipeline/train: Train the class and report
    # the results of the pipeline.

    st.header("Step 7: Training and evaluation")
    if st.button("Train & Evaluate Model"):
        results = pipeline.execute()
        st.write("Model Trained")

        st.header("Pipeline Results")
        st.write("Test Metrics:")
        for metric_obj, value in results["test metrics"]:
            st.write(f"{metric_obj.name}: {value}")

        st.write("Training Metrics:")
        for metric_obj, value in results["training metrics"]:
            st.write(f"{metric_obj.name}: {value}")
