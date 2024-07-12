from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.mixture import GaussianMixture
from sklearn import metrics
import sys
import os
import json
from sklearn.model_selection import KFold


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NpEncoder, self).default(obj)


MACHINE_IDS = ["id_00", "id_02", "id_04", "id_06"]


def gaussian_mixture(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    gmm = GaussianMixture(verbose=0)
    gmm.fit(X_train)

    # calculate ROC AUC
    y_pred = gmm.predict(X_test)
    scores = gmm.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return {
        # "y_pred": y_pred,
        # "scores": scores,
        "auc": auc,
        # "fpr": fpr,
        # "tpr": tpr,
        # "thresholds": thresholds,
    }


def local_outlier_factor(
    X_train: pd.DataFrame,  # all train data is normal
    X_test: pd.DataFrame,
    y_test: pd.Series,
):
    lof = LocalOutlierFactor(novelty=True, n_jobs=-1)
    lof.fit(X_train)

    # calculate ROC AUC
    scores = lof.score_samples(X_test)
    fpr, tpr, thresholds = metrics.roc_curve(y_test, scores, pos_label=1)
    auc = metrics.auc(fpr, tpr)

    return {
        # "y_pred": y_pred,
        # "scores": scores,
        "auc": auc,
        # "fpr": fpr,
        # "tpr": tpr,
        # "thresholds": thresholds,
    }


def sequential_feature_selection(model: "local_outlier_factor", df_normal: pd.DataFrame, df_anomaly: pd.DataFrame):
  kf = KFold(n_splits=5, shuffle=True, random_state=42)
  SCALER = StandardScaler()

  features = [x for x in list(df_normal.columns) if x != "label"]

  cur_features = []
  results = []
  for i in range(len(features)):
    print(f"Sequential Feature Selection: {i}/{len(features)}")
    print(f"cur features: {cur_features}")

    result = []
    for j in range(len(features)):
      # pick i features
      if features[j] in cur_features:
        continue

      ith_features = features[j]
      used_features = cur_features + [ith_features]

      print(f"try to add {ith_features} -> {used_features}")

      kfold_res = []
      idx_fold = 0
      for train_index, test_index in kf.split(df_normal):
        df_train_normal = df_normal.iloc[train_index]
        df_test_normal = df_normal.iloc[test_index]

        df_test = pd.concat([df_test_normal, df_anomaly])
        df_test = df_test.sample(frac=1).reset_index(drop=True)

        y_test = df_test["label"]
        y_test = [1 if x == False else -1 for x in y_test.to_numpy()]

        X_train_normal = df_train_normal.drop(columns=["label"])
        X_test = df_test.drop(columns=["label"])

        X_train_normal = X_train_normal[used_features]
        X_test = X_test[used_features]

        X_train_normal = SCALER.fit_transform(X_train_normal)
        X_test = SCALER.transform(X_test)

        res = model(X_train_normal, X_test, y_test)
        res["fold"] = idx_fold
        kfold_res.append(res)
        idx_fold += 1

      aucs = [x["auc"] for x in kfold_res]
      mean_auc = np.mean(aucs)
      std_auc = np.std(aucs)

      added_res = {
          "ith_feature": ith_features,
          # "results": kfold_res,
          "mean_auc": mean_auc,
          "std_auc": std_auc
      }
      result.append(added_res)

    # pick the best feature
    best_feature_idx = -1
    best_feature_auc = -1
    for j in range(len(result)):
      if result[j]["mean_auc"] > best_feature_auc:
        best_feature_auc = result[j]["mean_auc"]
        best_feature_idx = j

    best_feature = result[best_feature_idx]
    cur_features.append(best_feature["ith_feature"])
    results.append({
        "best_feature": best_feature,
        "cur_features": [x for x in cur_features],
        "result": result,
    })

  # best result
  best_result_idx = -1
  best_result_auc = -1
  for i in range(len(results)):
    if results[i]["best_feature"]["mean_auc"] > best_result_auc:
      best_result_auc = results[i]["best_feature"]["mean_auc"]
      best_result_idx = i

  return {
      "results": results,
      "best_result": results[best_result_idx]
  }


def fsown(
    mt: str,
    dataset: str
):
    lst_df_normal = []
    lst_df_anomaly = []
    for mi in MACHINE_IDS:
        df_normal = pd.read_csv(
            f"../data-split/{dataset}/{mt}-{mi}-normal-trainval.csv")
        df_abnormal = pd.read_csv(
            f"../data-split/{dataset}/{mt}-{mi}-abnormal-val.csv")

        # # reduce data for testing (faster)
        # df_normal = df_normal.sample(frac=0.02)
        # df_abnormal = df_abnormal.sample(frac=0.1)

        df_normal = df_normal.drop(
            columns=["filename", "machine_type", "machine_id", "label"])
        df_abnormal = df_abnormal.drop(
            columns=["filename", "machine_type", "machine_id", "label"])

        df_normal["label"] = False
        df_abnormal["label"] = True

        lst_df_normal.append(df_normal)
        lst_df_anomaly.append(df_abnormal)

    df_normal = pd.concat(lst_df_normal)
    df_normal = df_normal.sample(frac=1).reset_index(drop=True)

    df_anomaly = pd.concat(lst_df_anomaly)
    df_anomaly = df_anomaly.sample(frac=1).reset_index(drop=True)

    MODEL = local_outlier_factor # change this to change the model

    return sequential_feature_selection(MODEL, df_normal, df_anomaly)


MTs = ["fan", "pump", "slider", "valve"]
MIs = ["id_00", "id_02", "id_04", "id_06"]

dataset = sys.argv[1]  # "timbral"
mt = sys.argv[2]  # "fan"
identifier = sys.argv[3]  # "1"

outdir = f"../out/optimization/feature-selection/{dataset}/{identifier}"
os.makedirs(outdir, exist_ok=True)

print(dataset, mt, identifier)

selected_features = []
result = fsown(mt, dataset)
with open(f"{outdir}/{mt}.json", "w") as f:
    json.dump(result, f, indent=4, cls=NpEncoder)
