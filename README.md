# beth-dataset-anamoly-detection
In this project, we aimed to build an anomaly detection model using the BETH real cyber security dataset to detect anomalous activities on a fleet of Linux honey pots. We experimented with two models, Isolation
Forest and Stochastic Gradient Descent One-Class Support Vector Machine, running separate machine learning pipelines built with scikit-learn.

For Isolation Forest, we trained the model on the full dataset and achieved a ROC AUC of 0.85, consistent with the baseline results achieved by the BETH researchers. However, despite correctly capturing roughly 99% of the true anomalous activities, the model falsely identified close to 30% of normal activities as anomalies. In an effort to reduce the false positive rate while maintaining strong detection performance, we pivoted to a Linear SGD OCSVM model.

The Linear SGD OCSVM model combines the One-Class SVM formulation with Stochastic Gradient
Descent optimization. With its linear time complexity, lower memory requirements, and strong
performance in anomaly detection tasks, we trained a model that achieved a ROC AUC of 0.96. The model correctly captured roughly 95% of true anomalous activities while maintaining a false alarm rate of only around 2%. This is a substantial improvement over the Isolation Forest and a level of performance that has potential to be suitable for production deployment.


In anomaly detection, there is often a trade off between false positive and false negative rates. A high false positive rate generates excessive false alarms, while a high false negative rate increases the risk of missing true critical incidents. Given this trade-off, we selected the Linear SGD OCSVM model as our preferred and final model, as it controls false negatives better without sacrificing false positive rate.

## Repository structure

Below is a brief overview of the root-level folders and files in this project.

- `README.md` - This file. It explains the project goals, model findings, and how to run the code.
- `requirements.txt` - Python package dependencies required to run the notebooks and scripts.
- `data_import_and_exploration.ipynb` - Notebook for initial data loading, exploration, and early preprocessing experiments.
- `model_training_isolation_forest.ipynb` - Notebook focused on training and evaluating the Isolation Forest anomaly detector.
- `model_training_sgd_oneclass_svm.ipynb` - Notebook focused on training and evaluating the SGD one-class SVM anomaly detector.

### Root folders

- `datasets/` - Contains raw dataset files, split training/validation/testing data, and dataset metadata.
- `exploration/` - Exploratory notebooks and helper scripts used for data preprocessing, feature engineering, and model experimentation.
- `feature_testing/` - Scripts for testing and comparing feature sets on different anomaly detection models.
- `features_data/` - Exported feature importance results from model experiments.
- `model_class/` - Feature builder modules used to create model-ready datasets from the raw BETH data.
- `research/` - for research notes, papers, or additional analysis supporting the project.


### How to run:
1. From the root directory, pip install -r requirements.txt
2. Make sure you are running the notebook on python 3.14.2 (as that is what we are running on)
3. Follow the readme.md in `/datasets/raw` to get the needed data to run the project
4. Run the `model_training_isolation_forest.ipynb` notebook in the root directory for Isolation Forest anomaly detector.
5. Run the `model_training_sgd_oneclass_svm.ipynb` notebook in the root directory for SGD one-class SVM anomaly detector.
