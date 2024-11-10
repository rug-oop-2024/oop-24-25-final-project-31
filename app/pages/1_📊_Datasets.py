from typing import List, Optional, Tuple
import streamlit as st
import pandas as pd
from sklearn.datasets import fetch_openml
from app.core.system import AutoMLSystem
from autoop.core.ml.dataset import Dataset
from autoop.core.ml.feature import Feature
from autoop.functional.feature import detect_feature_types


class DatasetManagement:
    def __init__(self):
        """Initialize AutoML system and load available datasets."""
        self.automl = AutoMLSystem.get_instance()
        self.datasets = self._list()

    def _list(self) -> List[Dataset]:
        """
        Loads the list of available datasets from the AutoML system registry.

        Returns:
            List[Dataset]: A list of datasets registered in the AutoML system.
        """
        return self.automl.registry.list(type="dataset")

    def create(self):
        """Fetch and register the Iris dataset as an example dataset."""
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
        self.automl.registry.register(dataset)
        st.success("Iris dataset was registered")

    def select_dataset(self) -> Optional[Dataset]:
        """
        Select a dataset from the AutoML system registry using
        a Streamlit selectbox.

        Returns:
            Optional[Dataset]: A dataset selected from the AutoML system
            registry, or None.
        """
        dataset_list = [ds.name for ds in self.datasets]
        selected_name = st.selectbox("Select a Dataset:", dataset_list)
        if selected_name:
            dataset = next(ds for ds
                           in self.datasets if ds.name == selected_name)
            return Dataset(
                name=dataset.name,
                asset_path=dataset.asset_path,
                version=dataset.version,
                data=dataset.data
            )
        return None

    def _features(self, dataset: Dataset) -> (Tuple[List[Feature],
                                                    Feature, str]):
        """
        Detects the features in a dataset and returns a tuple
        containing the list of input features, the target feature,
        and the task type.

        Parameters:
            dataset (Dataset): The dataset to detect features from.

        Returns:
            Tuple[List[Feature], Feature, str]: A tuple containing the
            list of input features, the target feature, and the task type.
        """
        df = dataset.read()
        features = detect_feature_types(dataset)
        numerical_features = [f.name for
                              f in features if f.type == "numerical"]
        categorical_features = [f.name for
                                f in features if f.type == "categorical"]

        input_features = []
        input_feature_names = st.multiselect("Select input features",
                                             numerical_features +
                                             categorical_features)
        for feature_name in input_feature_names:
            if feature_name in numerical_features:
                feature_type = "numerical"
            elif feature_name in categorical_features:
                feature_type = "categorical"
            else:
                raise ValueError(f"Unknown feature type for {feature_name}")
            input_features.append(
                Feature(name=feature_name, type=feature_type))

        target_feature_name = st.selectbox("Select target feature:",
                                           df.columns)
        if target_feature_name in numerical_features:
            feature_type = "numerical"
            task_type = "regression"
        else:
            feature_type = "categorical"
            task_type = "classification"
        target_feature = Feature(name=target_feature_name, type=feature_type)

        st.write(f"Detected Task Type: {task_type}")
        return input_features, target_feature, task_type

#task = st.selectbox('question', [options])
#split = st.slider('question', 0.0, 1.0, 0.5)
# ask for set of metrics: metrics = st.multiselect('question', [all metrics])
#st.selectbox('question', ['classification', 'regression'])


def main():
    page = DatasetManagement()
    page.create()

    dataset = page.select_dataset()
    if dataset:
        df = dataset.read()
        st.write(f"Dataset selected: {dataset.name}")
        st.write(df)

        st.header("Feature Selection")
        input_features, target_feature, task_type = page._features(dataset)
        st.write("Input Features:", [f.name for f in input_features])
        st.write("Target Feature:", target_feature.name)
        st.write("Task Type:", task_type)


if __name__ == "__main__":
    main()
