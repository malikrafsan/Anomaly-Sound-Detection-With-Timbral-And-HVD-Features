from itertools import product
from typing import Union
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn import metrics
import sys
import os
import json
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import KFold


DROPPED_COLUMNS = [
    "filename", "machine_type", "machine_id", "label"
]

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, pd.Series):
            return obj.to_list()
        return super(NpEncoder, self).default(obj)


def filename(
    inputdir: str,
    machine_type: str,
    machine_id: str,
    label: str
):
    postfix = ""
    if label == "normal" or label == "train":
        postfix = "normal-trainval"
    elif label == "abnormal":
        postfix = "abnormal-val"
    elif label == "test":
        postfix = "test"
    else:
        raise ValueError("Invalid label")

    return f"{inputdir}/{machine_type}-{machine_id}-{postfix}.csv"


def one_class_svm(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param: dict,
):
    svm = OneClassSVM(verbose=1, **param)
    svm.fit(X_train)

    # calculate ROC AUC
    y_pred = svm.predict(X_test)
    scores = svm.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return {
        "y_pred": y_pred,
        "scores": scores,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def gaussian_mixture(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param: dict,
):
    gmm = GaussianMixture(verbose=1, **param)
    gmm.fit(X_train)

    # calculate ROC AUC
    y_pred = gmm.predict(X_test)
    scores = gmm.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return {
        "y_pred": y_pred,
        "scores": scores,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def local_outlier_factor(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param: dict,
):
    lof = LocalOutlierFactor(novelty=True, n_jobs=-1, **param)
    lof.fit(X_train)

    # calculate ROC AUC
    y_pred = lof.predict(X_test)
    scores = lof.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return {
        "y_pred": y_pred,
        "scores": scores,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def isoforest(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
    param: dict,
):
    isoforest = IsolationForest(n_jobs=-1, verbose=1, **param)
    isoforest.fit(X_train)

    # calculate ROC AUC
    y_pred = isoforest.predict(X_test)
    scores = isoforest.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return {
        "y_pred": y_pred,
        "scores": scores,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
    }


def pipeline(
    inputdir: str,
    machine_type: str,
    machine_id: str,
    model_name: str,
    scaler: Union[StandardScaler, MinMaxScaler,
                  RobustScaler, MaxAbsScaler, None] = None,
    features_selected: list[str] = None,
    param: dict = None
):
    MODELS = {
        "one_class_svm": one_class_svm,
        "gaussian_mixture": gaussian_mixture,
        "local_outlier_factor": local_outlier_factor,
        "isolation_forest": isoforest,
    }
    model = MODELS[model_name]

    df_normal = pd.read_csv(
        filename(inputdir, machine_type, machine_id, "normal"))
    df_abnormal = pd.read_csv(
        filename(inputdir, machine_type, machine_id, "abnormal"))

    df_normal = df_normal.drop(columns=DROPPED_COLUMNS)
    df_abnormal = df_abnormal.drop(columns=DROPPED_COLUMNS)

    df_normal["label"] = False
    df_abnormal["label"] = True

    # shuffle data
    df_normal = df_normal.sample(frac=1)
    df_abnormal = df_abnormal.sample(frac=1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # results = []
    # for model_name, model in MODELS.items():
    model_result = []
    idx = 0

    for train_index, test_index in kf.split(df_normal):
        df_train_normal = df_normal.iloc[train_index]
        df_test_normal = df_normal.iloc[test_index]

        df_test = pd.concat(
            [df_test_normal, df_abnormal], ignore_index=True)
        df_test = df_test.sample(frac=1)
        y_test = df_test["label"]
        y_test = [1 if x == False else -1 for x in y_test.to_numpy()]

        X_train_normal = df_train_normal.drop(columns=["label"])
        X_test = df_test.drop(columns=["label"])

        if features_selected is not None:
            X_train_normal = X_train_normal[features_selected]
            X_test = X_test[features_selected]

        if scaler is not None:
            X_train_normal = scaler.fit_transform(X_train_normal)
            X_test = scaler.transform(X_test)

        result = model(X_train_normal, X_test, y_test, param)
        result["fold"] = idx+1
        model_result.append(result)
        idx += 1

    aucs = [x["auc"] for x in model_result]
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    results = {
        "model_name": model_name,
        "results": model_result,
        "mean_auc": mean_auc,
        "std_auc": std_auc
    }

    return {
        "machine_type": machine_type,
        "machine_id": machine_id,
        "scaler": "NoScaler" if scaler is None else scaler.__class__.__name__,
        "results": results,
        "features_selected": ",".join(features_selected) if features_selected is not None else "All Features",
        "param": param
    }


def clean_filepath(filepath: str):
    return filepath.replace("..", "").replace("/", "-")


def get_features_selected(fs_dir: str, machine_type: str):
    features_selected = None

    with open(f"{fs_dir}/{machine_type}.json", "r") as f:
        data = json.load(f)
        print(data.keys())
        features_selected = data["best_result"]["cur_features"]

    return features_selected


def paramize(params: dict[str, list]):
    return [dict(zip(params.keys(), values))
            for values in product(*params.values())]


def reduce_params(params: dict[str, list]):
    return {k: [v[1]] for k, v in params.items()}


def get_params(model_name: str):
    params = {
        "gaussian_mixture": {
            "n_components": [1, 2, 3, 5],
            "covariance_type": ['full', 'tied', 'diag', 'spherical'],
            "tol": [1e-3, 1e-2, 1e-1],
            'max_iter': [100, 300, 500],
            'init_params': ['kmeans', 'k-means++', 'random', 'random_from_data'],
        },
        "isolation_forest": {
            "n_estimators": [50, 100, 150],
            "max_samples": ["auto", 0.25, 0.5, 0.75, 1.0],
            "contamination": ["auto", 0.1, 0.2, 0.3, 0.5],
            "max_features": [0.25, 0.5, 0.75, 1.0],
            "bootstrap": [True, False],
        },
        "local_outlier_factor": {
            'n_neighbors': [5, 10, 20, 30],
            'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'leaf_size': [15, 20, 30, 40],
            'metric': ['minkowski', 'euclidean', 'manhattan'],
            'contamination': ['auto', 0.01, 0.05, 00.1],
        },
        "one_class_svm": {
            'kernel': ['rbf', 'linear', 'poly', 'sigmoid'],
            'gamma': ['scale', 'auto'],
            'tol': [1e-2, 1e-3, 1e-4],
            'nu': [0.3, 0.5, 0.7],
            'shrinking': [True, False],
            'max_iter': [-1, 100, 300],
        },
    }
    
    if model_name not in params:
        raise ValueError("Invalid model name")
      
    return params[model_name]


def main():
    datatype = sys.argv[1]
    mt = sys.argv[2]
    mi = sys.argv[3]
    identifier = sys.argv[4]
    model_name = sys.argv[5]
    
    inputdir = f"../data-split/{datatype}"
    outdir = f"../out/optimization/hyperparameter-tuning/{datatype}/{identifier}/{model_name}"
    os.makedirs(outdir, exist_ok=True)

    fsdir_hvd = f"../out/optimization/feature-selection/hvd/1"  # change this if needed
    fsdir_timbre = f"../out/optimization/feature-selection/timbral/1"  # change this if needed

    features_selected1 = get_features_selected(fsdir_hvd, mt)
    features_selected2 = get_features_selected(fsdir_timbre, mt)

    features_selected = []
    if datatype.startswith("combination"):
        features_selected = list(
            set(features_selected1 + features_selected2))
    elif datatype.startswith("timbral"):
        features_selected = features_selected2
    elif datatype.startswith("hvd"):
        features_selected = features_selected1
    else:
        raise ValueError("Invalid datatype")

    param_grid = get_params(model_name)
    all_params = paramize(param_grid)
    print(len(all_params))
    print(features_selected)
    all_params = all_params[:10]

    results = []
    for idx in range(len(all_params)):
        param = all_params[idx]
        print(idx, len(all_params), param)
        result = pipeline(inputdir, mt, mi, model_name, scaler=StandardScaler(
        ), features_selected=features_selected, param=param)
        results.append(result)

    slim = []
    for x in results:
        slim.append({
            "param": x["param"],
            "mean_auc": x["results"]["mean_auc"],
            "model_name": x["results"]["model_name"],
            "machine_type": x["machine_type"],
            "machine_id": x["machine_id"],
            "scaler": x["scaler"],
            "features_selected": x["features_selected"]
        })

    best_param = max(slim, key=lambda x: x["mean_auc"])
    stored = {
        "best_param": best_param,
        "all_results": slim
    }

    with open(f"{outdir}/{mt}-{mi}.json", "w") as f:
        json.dump(stored, f, cls=NpEncoder, indent=4)


if __name__ == '__main__':
    main()
