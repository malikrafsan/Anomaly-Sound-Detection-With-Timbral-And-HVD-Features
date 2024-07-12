MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
MACHINE_IDS = ["id_00", "id_02","id_04","id_06"]

import sys, os, json
import pandas as pd

RANDOM_STATE = 0

rootpath = "../data"
dataname = sys.argv[1] # ["hvd", "timbral", "combination"]
dirpath = f"{rootpath}/{dataname}"

outdir = f"../data-split/{dataname}"
os.makedirs(outdir, exist_ok=True)

for mt in MACHINE_TYPES:
  for mi in MACHINE_IDS:
    df_normal = pd.read_csv(f"{dirpath}/{mt}-{mi}-normal.csv")
    df_abnormal = pd.read_csv(f"{dirpath}/{mt}-{mi}-abnormal.csv")
    
    print(f"{mt}-{mi} normal: {df_normal.shape[0]}, abnormal: {df_abnormal.shape[0]}")
    
    anomaly_len = df_abnormal.shape[0]
    
    # 50% anomaly -> validation, test
    # same length for validation and test
    # rest for training
    
    anomaly_len_val = anomaly_len // 2
    anomaly_len_test = anomaly_len // 2
    
    df_abnormal = df_abnormal.sample(frac=1,random_state=RANDOM_STATE)
    df_normal = df_normal.sample(frac=1, random_state=RANDOM_STATE)
    
    df_abnormal_val = df_abnormal.sample(frac=0.5, random_state=RANDOM_STATE)
    df_abnormal_test = df_abnormal.drop(df_abnormal_val.index)
    
    df_normal_test = df_normal.sample(n=df_abnormal_test.shape[0])
    df_normal_trainval = df_normal.drop(df_normal_test.index)
    
    df_test = pd.concat([df_abnormal_test, df_normal_test])
    df_test = df_test.sample(frac=1)
    
    print("df_test", df_test.shape[0])
    print("df_abnormal_val", df_abnormal_val.shape[0])
    print("df_normal_trainval", df_normal_trainval.shape[0])

    print("================\n\n")

    df_test.to_csv(f"{outdir}/{mt}-{mi}-test.csv", index=False)
    df_abnormal_val.to_csv(f"{outdir}/{mt}-{mi}-abnormal-val.csv", index=False)
    df_normal_trainval.to_csv(
        f"{outdir}/{mt}-{mi}-normal-trainval.csv", index=False)

