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


DROPPED_COLUMNS = [
    "filename", "machine_type", "machine_id", "label"
]

HYPERPARAMETER_TUNING_FILEPATH = "../parameters/hyperparameter.json"
with open(HYPERPARAMETER_TUNING_FILEPATH, "r") as f:
    HYPERPARAMETER_TUNING = json.load(f)
    datatype = sys.argv[1]
    PARAMS = HYPERPARAMETER_TUNING[datatype]


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
    return f"{inputdir}/{machine_type}-{machine_id}-{"normal-trainval" if label == "normal" else "abnormal-val"}.csv"


def one_class_svm(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    svm = OneClassSVM(verbose=1, **PARAMS["ocsvm"])
    svm.fit(X_train)

    # calculate ROC AUC
    y_pred = svm.predict(X_test)
    scores = svm.score_samples(X_test)
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # calculate f1 score per threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred = [False if x < threshold else True for x in scores]
        f1 = metrics.f1_score(y_test, y_pred)
        f1_scores.append({
            "threshold": threshold,
            "f1": f1
        })

    return {
        "scores": scores,
        "auc": auc,
        "f1_scores": f1_scores,
        "best_f1": max(f1_scores, key=lambda x: x["f1"]),
    }


def gaussian_mixture(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    gmm = GaussianMixture(verbose=1, **PARAMS["gmm"])
    gmm.fit(X_train)

    # calculate ROC AUC
    y_pred = gmm.predict(X_test)
    scores = gmm.score_samples(X_test)
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # calculate f1 score per threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred = [False if x < threshold else True for x in scores]
        f1 = metrics.f1_score(y_test, y_pred)
        f1_scores.append({
            "threshold": threshold,
            "f1": f1
        })

    return {
        "scores": scores,
        "auc": auc,
        "f1_scores": f1_scores,
        "best_f1": max(f1_scores, key=lambda x: x["f1"]),
    }


def local_outlier_factor(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    lof = LocalOutlierFactor(novelty=True, n_jobs=-1, **PARAMS["lof"])
    lof.fit(X_train)

    # calculate ROC AUC
    y_pred = lof.predict(X_test)
    scores = lof.score_samples(X_test)
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # calculate f1 score per threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred = [False if x < threshold else True for x in scores]
        # print(y_pred, y_test)
        f1 = metrics.f1_score(y_test, y_pred)
        f1_scores.append({
            "threshold": threshold,
            "f1": f1
        })

    return {
        "scores": scores,
        "auc": auc,
        "f1_scores": f1_scores,
        "best_f1": max(f1_scores, key=lambda x: x["f1"]),
    }


def isoforest(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    isoforest = IsolationForest(n_jobs=-1, verbose=1, **PARAMS["isoforest"])
    isoforest.fit(X_train)

    # calculate ROC AUC
    y_pred = isoforest.predict(X_test)
    scores = isoforest.score_samples(X_test)
    scores = (scores - scores.min()) / (scores.max() - scores.min())

    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    # calculate f1 score per threshold
    f1_scores = []
    for threshold in thresholds:
        y_pred = [False if x < threshold else True for x in scores]
        # print(y_pred, y_test)
        f1 = metrics.f1_score(y_test, y_pred)
        f1_scores.append({
            "threshold": threshold,
            "f1": f1
        })

    return {
        "scores": scores,
        "auc": auc,
        "f1_scores": f1_scores,
        "best_f1": max(f1_scores, key=lambda x: x["f1"]),
    }


def pipeline(
    inputdir: str,
    machine_type: str,
    machine_id: str,
    scaler: Union[StandardScaler, MinMaxScaler,
                  RobustScaler, MaxAbsScaler, None] = None,
    features_selected: list[str] = None
):
    MODELS = {
        "one_class_svm": one_class_svm,
        "gaussian_mixture": gaussian_mixture,
        "local_outlier_factor": local_outlier_factor,
        "isoforest": isoforest,
    }

    df_normal = pd.read_csv(
        filename(inputdir, machine_type, machine_id, "normal"))
    df_abnormal = pd.read_csv(
        filename(inputdir, machine_type, machine_id, "abnormal"))

    df_normal = df_normal.drop(columns=DROPPED_COLUMNS)
    df_abnormal = df_abnormal.drop(columns=DROPPED_COLUMNS)

    df_normal["label"] = True
    df_abnormal["label"] = False

    # shuffle data
    df_normal = df_normal.sample(frac=1)
    df_abnormal = df_abnormal.sample(frac=1)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    results = []
    for model_name, model in MODELS.items():
        model_result = []
        idx = 0

        for train_index, test_index in kf.split(df_normal):
            df_train_normal = df_normal.iloc[train_index]
            df_test_normal = df_normal.iloc[test_index]

            df_test = pd.concat(
                [df_test_normal, df_abnormal], ignore_index=True)
            df_test = df_test.sample(frac=1)
            y_test = df_test["label"]
            # y_test = [1 if x == False else -1 for x in y_test.to_numpy()]

            X_train_normal = df_train_normal.drop(columns=["label"])
            X_test = df_test.drop(columns=["label"])

            if features_selected is not None:
                X_train_normal = X_train_normal[features_selected]
                X_test = X_test[features_selected]

            if scaler is not None:
                X_train_normal = scaler.fit_transform(X_train_normal)
                X_test = scaler.transform(X_test)

            result = model(X_train_normal, X_test, y_test)
            result["fold"] = idx+1
            model_result.append(result)
            idx += 1

        aucs = [x["auc"] for x in model_result]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)

        result = {
            "model_name": model_name,
            "results": model_result,
            "mean_auc": mean_auc,
            "std_auc": std_auc
        }
        results.append(result)

    return {
        "machine_type": machine_type,
        "machine_id": machine_id,
        "scaler": "NoScaler" if scaler is None else scaler.__class__.__name__,
        "results": results,
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
    datatype = sys.argv[1]
    identifier = sys.argv[2]

    machine_types = ["fan", "pump", "slider", "valve"]
    machine_ids = ["id_00", "id_02", "id_04", "id_06"]
    inputdir = f"../data-split/{datatype}"
    outdir = f"../out/optimization/threshold/{datatype}/{identifier}"
    os.makedirs(outdir, exist_ok=True)

    fsdir_hvd = f"../out/optimization/feature-selection/hvd/1"  # change this if needed
    fsdir_timbre = f"../out/optimization/feature-selection/timbral/1"  # change this if needed

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
            ), features_selected=features_selected)
            results.append(result)

    with open(f"{outdir}/results-{clean_filepath(outdir)}.json", "w") as f:
        json.dump(results, f, cls=NpEncoder, indent=4)
    print(f"Results saved to {outdir}/results.json")

    # get best f1 score and threshold
    f1data = []
    for elmt in results:
        newResult = []
        for model in elmt["results"]:
            newModelRes = []
            for modelRes in model["results"]:
                picked = ["auc", "best_f1", "fold"]
                newModelRes.append({k: modelRes[k] for k in picked})
            newResult.append(
                {"model_name": model["model_name"], "results": newModelRes})

        newF1Data = {k: elmt[k] for k in elmt.keys() if k != "results"}
        newF1Data["results"] = newResult
        f1data.append(newF1Data)

    with open(f"{outdir}/results-slim-{clean_filepath(outdir)}.json", "w") as f:
        json.dump(f1data, f, cls=NpEncoder, indent=4)
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
            auc[res["model_name"]] = res["mean_auc"]

        aucs.append(auc)

    df = pd.DataFrame(aucs)
    df.to_csv(f"{outdir}/auc-{clean_filepath(outdir)}.csv", index=False)
    print(f"AUCs saved to {outdir}/aucs.csv")


if __name__ == '__main__':
    main()
