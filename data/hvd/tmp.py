import sys, os
import pandas as pd

file = [x for x in os.listdir(".") if x.endswith(".csv")][0]

df = pd.read_csv(file)

removed_col = set(["filename", "machine_type", "machine_id", "label"])

cols = set(df.columns)

unremoved_col = cols - removed_col

# print(unremoved_col)
for col in unremoved_col:
    print(col)
print(len(unremoved_col))
