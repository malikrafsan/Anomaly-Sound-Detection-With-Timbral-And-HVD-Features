import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn import metrics
import sys
import os
import json
import time
import pickle


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
    postfix = "normal-trainval" if label == "normal" or label == "train" else "abnormal-val" if label == "abnormal" else "test"
    return f"{inputdir}/{machine_type}-{machine_id}-{postfix}.csv"


def one_class_svm(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    svm = OneClassSVM(verbose=1)
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
        "model": svm,
    }


def gaussian_mixture(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    gmm = GaussianMixture(verbose=1)
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
        "model": gmm,
    }


def local_outlier_factor(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    lof = LocalOutlierFactor(novelty=True, n_jobs=-1)
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
        "model": lof,
    }


def isoforest(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    isoforest = IsolationForest(n_jobs=-1, verbose=1)
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
        "model": isoforest,
    }


def pipeline(
    inputdir: str,
    machine_type: str,
    machine_id: str,
    model_base_path: str,
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
    label = label.apply(lambda x: 1 if x == "normal" else -1)

    df_train = df_train.drop(columns=DROPPED_COLUMNS)
    df_test = df_test.drop(columns=DROPPED_COLUMNS)

    results = []
    for model_name, model in MODELS.items():
        result = model(df_train, df_test, label)
        trained_model = result.pop("model")
        model_path = f"{
            model_base_path}/{machine_type}-{machine_id}-{model_name}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(trained_model, f)

        results.append({
            "model_name": model_name,
            **result
        })

    return {
        "machine_type": machine_type,
        "machine_id": machine_id,
        "results": results
    }


def clean_filepath(filepath: str):
    return filepath.replace("..", "").replace("/", "-")


def main():
    stamp = time.strftime("%Y%m%d-%H%M%S")

    datatype = sys.argv[1]

    machine_types = ["fan", "pump", "slider", "valve"]
    machine_ids = ["id_00", "id_02", "id_04", "id_06"]
    inputdir = f"../data-split/{datatype}"
    outdir = f"../out/test/baseline/{datatype}/{stamp}"
    MODEL_PATH = f"../models/baseline/{datatype}"

    os.makedirs(MODEL_PATH, exist_ok=True)
    os.makedirs(outdir, exist_ok=True)

    results = []
    for mt in machine_types:
        for mid in machine_ids:
            result = pipeline(inputdir, mt, mid, MODEL_PATH)
            results.append(result)

    with open(f"{outdir}/results-{clean_filepath(outdir)}__.json", "w") as f:
        json.dump(results, f, cls=NpEncoder, indent=4)
    print(f"Results saved to {outdir}/results.json")

    # take all AUC score only
    aucs = []
    for result in results:
        auc = {
            "machine_type": result["machine_type"],
            "machine_id": result["machine_id"],
        }

        for res in result["results"]:
            auc[res["model_name"]] = res["auc"]

        aucs.append(auc)

    df = pd.DataFrame(aucs)
    df.to_csv(f"{outdir}/auc-{clean_filepath(outdir)}.csv", index=False)
    print(f"AUCs saved to {outdir}/aucs.csv")


if __name__ == '__main__':
    main()
