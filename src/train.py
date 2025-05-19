"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import mlflow.sklearn
import mlflow.sklearn
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow
import mlflow 

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact
    
    joblib.dump(col_transf, "column_transformer.joblib")
    mlflow.log_artifact("column_transformer.joblib")

    return col_transf, X_train, X_test, y_train, y_test


def train_log_reg(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    max_iter=1000
    log_reg = LogisticRegression(max_iter=max_iter)
    log_reg.fit(X_train, y_train)

    #log parameters
    mlflow.log_param("max_iter", max_iter)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    pred_signature = mlflow.models.infer_signature(X_train, log_reg.predict(X_train))

    # Log model
    mlflow.sklearn.log_model(log_reg, artifact_path='model' , signature=pred_signature, input_example = X_train.iloc[[0]])

    ### Log the data
    mlflow.log_artifact("dataset/Churn_Modelling.csv")

    return log_reg

def train_random_forest(X_train, y_train):
    """
    Train a random forest model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        RandomForestClassifier: trained random forest model
    """
    n_estimator = 100
    random_forest = RandomForestClassifier(n_estimators=n_estimator)
    random_forest.fit(X_train, y_train)

    #log parameters
    mlflow.log_param("n_estimators", n_estimator)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    pred_signature = mlflow.models.infer_signature(X_train, random_forest.predict(X_train))

    # Log model
    mlflow.sklearn.log_model(random_forest, artifact_path='model_rf' , signature=pred_signature, input_example = X_train.iloc[[0]])

    ### Log the data
    mlflow.log_artifact("dataset/Churn_Modelling.csv")

    return random_forest


def train_decision_tree(X_train, y_train):
    """
    Train a decision tree model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        DecisionTreeClassifier: trained decision tree model
    """
    max_depth=10
    decision_tree = DecisionTreeClassifier(max_depth=max_depth)
    decision_tree.fit(X_train, y_train)

    #log parameters
    mlflow.log_param("max_depth", max_depth)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)
    pred_signature = mlflow.models.infer_signature(X_train, decision_tree.predict(X_train))

    # Log model
    mlflow.sklearn.log_model(decision_tree, artifact_path='model_dt' , signature=pred_signature, input_example = X_train.iloc[[0]])

    ### Log the data
    mlflow.log_artifact("dataset/Churn_Modelling.csv")

    return decision_tree

def log_model_with_mlflow(model, X_test, y_test, model_name, exp_id, output_dir):
        ### Log the metrics
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        mlflow.set_tag("model", model_name)

        conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
        conf_mat_disp = ConfusionMatrixDisplay(
            confusion_matrix=conf_mat, display_labels=model.classes_
        )
        conf_mat_disp.plot()
        
        # Log the image as an artifact in MLflow
        conf_mat_disp.figure_.savefig(f'{output_dir}/mat_{model_name}.png')
        mlflow.log_artifact(f'{output_dir}/mat_{model_name}.png')
        
        plt.show()


def main():
    os.environ["LOGNAME"] = "Zad"
    output_dir = "output"
    ### Set the tracking URI for MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    ### Set the experiment name
    exp_id = mlflow.set_experiment("exp_2").experiment_id

    df = pd.read_csv("dataset/Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    mlflow.end_run()
    with mlflow.start_run(experiment_id=exp_id):
        model_rf = train_random_forest(X_train, y_train)
        log_model_with_mlflow(model_rf, X_test, y_test, "RandomForest", exp_id, output_dir)

    mlflow.end_run()
    with mlflow.start_run(experiment_id=exp_id):
        model_log_reg = train_log_reg(X_train, y_train)
        log_model_with_mlflow(model_log_reg, X_test, y_test, "LogisticRegression", exp_id, output_dir)

    mlflow.end_run()
    with mlflow.start_run(experiment_id=exp_id):
        model_dt = train_decision_tree(X_train, y_train)
        log_model_with_mlflow(model_dt, X_test, y_test, "DecisionTree", exp_id, output_dir)

        

if __name__ == "__main__":
    main()
