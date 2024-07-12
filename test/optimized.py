from typing import Union
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn import metrics
from sklearn.decomposition import PCA
import sys
import os
import json
import time
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, MaxAbsScaler
from sklearn.model_selection import KFold
import pickle
import matplotlib.pyplot as plt


DROPPED_COLUMNS = [
    "filename", "machine_type", "machine_id", "label"
]

THRESHOLD_PATH = "../parameters/threshold.json"
with open(THRESHOLD_PATH, "r") as f:
    TABLE_THRESHOLD = json.load(f)
    argdatatype = sys.argv[1]
    if argdatatype == "combination":
        argdatatype = "timbral"
    
    TABLE_THRESHOLD = TABLE_THRESHOLD[argdatatype]


HYPERPARAM_PATH = "../parameters/hyperparameter.json"
with open(HYPERPARAM_PATH, "r") as f:
    HYPERPARAM = json.load(f)
    argdatatype = sys.argv[1]
    if argdatatype == "combination":
        argdatatype = "timbral"
    
    HYPERPARAM = HYPERPARAM[argdatatype]


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
    mt: str,
    mi: str,
):
    print("HYPERPARAM", HYPERPARAM)
    svm = OneClassSVM(verbose=1, **HYPERPARAM["ocsvm"])
    svm.fit(X_train)

    # calculate ROC AUC
    scores = svm.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    print(f"Threshold: {TABLE_THRESHOLD}")
    threshold = TABLE_THRESHOLD[mt]["one_class_svm"]
    f1 = metrics.f1_score(y_test, scores > threshold)

    return {
        "auc": auc,
        "f1": f1,
        "model": svm,
    }


def gaussian_mixture(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mt: str,
    mi: str,
):
    gmm = GaussianMixture(verbose=1, **HYPERPARAM["gmm"])
    gmm.fit(X_train)

    # calculate ROC AUC
    scores = gmm.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    threshold = TABLE_THRESHOLD[mt]["gaussian_mixture"]
    f1 = metrics.f1_score(y_test, scores > threshold)

    return {
        "auc": auc,
        "f1": f1,
        "model": gmm,
    }


def local_outlier_factor(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mt: str,
    mi: str,
):
    lof = LocalOutlierFactor(novelty=True, n_jobs=-1, **HYPERPARAM["lof"])
    lof.fit(X_train)

    # calculate ROC AUC
    scores = lof.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    threshold = TABLE_THRESHOLD[mt]["local_outlier_factor"]
    f1 = metrics.f1_score(y_test, scores > threshold)

    return {
        "auc": auc,
        "f1": f1,
        "model": lof,
    }


def isoforest(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
    mt: str,
    mi: str,
):
    isoforest = IsolationForest(
        n_jobs=-1, verbose=1, **HYPERPARAM["isoforest"])
    isoforest.fit(X_train)

    # calculate ROC AUC
    scores = isoforest.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    threshold = TABLE_THRESHOLD[mt]["isoforest"]
    f1 = metrics.f1_score(y_test, scores > threshold)

    return {
        "auc": auc,
        "f1": f1,
        "model": isoforest,
    }


def pipeline(
    inputdir: str,
    machine_type: str,
    machine_id: str,
    scaler: Union[StandardScaler, MinMaxScaler,
                  RobustScaler, MaxAbsScaler, None] = None,
    features_selected: list = None,
    MODEL_PATH: str = None
):
    MODELS = {
        "one_class_svm": one_class_svm,
        "gaussian_mixture": gaussian_mixture,
        "local_outlier_factor": local_outlier_factor,
        "isoforest": isoforest,
    }

    df_train = pd.read_csv(
        filename(inputdir, machine_type, machine_id, "train"))
    df_test = pd.read_csv(
        filename(inputdir, machine_type, machine_id, "test"))

    label = df_test["label"]
    y_test = label.apply(lambda x: True if x == "normal" else False)

    X_train = df_train.drop(columns=DROPPED_COLUMNS)
    X_test = df_test.drop(columns=DROPPED_COLUMNS)

    if features_selected is not None:
        X_train = X_train[features_selected]
        X_test = X_test[features_selected]

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        SCALER_PATH = f"{MODEL_PATH}/scaler"
        os.makedirs(SCALER_PATH, exist_ok=True)
        with open(f"{SCALER_PATH}/{machine_type}-{machine_id}.pkl", "wb") as f:
            pickle.dump(scaler, f)

    results = []
    for model_name, model in MODELS.items():
        result = model(X_train, X_test, y_test, machine_type, machine_id)
        trained_model = result.pop("model")
        model_path = f"{
            MODEL_PATH}/{machine_type}-{machine_id}-{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(trained_model, f)

        results.append({
            "model_name": model_name,
            **result
        })

    return {
        "machine_type": machine_type,
        "machine_id": machine_id,
        "scaler": "NoScaler" if scaler is None else scaler.__class__.__name__,
        "results": results,
        "features_selected": ",".join(features_selected) if features_selected is not None else "All Features"
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


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")

    datatype = sys.argv[1]

    machine_types = ["fan", "pump", "slider", "valve"]
    machine_ids = ["id_00", "id_02", "id_04", "id_06"]
    inputdir = f"../data-split/{datatype}"
    outdir = f"../out/test/optimized/{datatype}/{stamp}"
    
    fsdir_hvd = f"../out/optimization/feature-selection/hvd/1"  # change this if needed
    fsdir_timbre = f"../out/optimization/feature-selection/timbral/1"  # change this if needed
    os.makedirs(outdir, exist_ok=True)

    MODEL_PATH = f"../models/optimized/{datatype}"
    os.makedirs(MODEL_PATH, exist_ok=True)

    results = []
    for mt in machine_types:
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

        for mid in machine_ids:
            result = pipeline(inputdir, mt, mid, scaler=StandardScaler(
            ), features_selected=features_selected, MODEL_PATH=MODEL_PATH)
            results.append(result)

    with open(f"{outdir}/results-{clean_filepath(outdir)}.json", "w") as f:
        json.dump(results, f, cls=NpEncoder, indent=4)
    print(f"Results saved to {outdir}/results.json")

    # take all AUC score only
    aucs = []
    for result in results:
        auc = {
            "machine_type": result["machine_type"],
            "machine_id": result["machine_id"],
            "scaler": result["scaler"]
        }

        for res in result["results"]:
            auc[res["model_name"]+"_auc"] = res["auc"]
            auc[res["model_name"]+"_f1"] = res["f1"]

        aucs.append(auc)

    df = pd.DataFrame(aucs)
    df.to_csv(f"{outdir}/auc-{clean_filepath(outdir)}.csv", index=False)
    print(f"AUCs saved to {outdir}/aucs.csv")


if __name__ == '__main__':
    main()
