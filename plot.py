import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df_train = pd.read_csv("datasets/processed/feature_engineered_training_data.csv", header=0)

plt.hist(df_train["same_process_name_as_parent"], bins=50)
plt.xlabel("same_process_name_as_parent")
plt.ylabel("Count")
plt.title("Distribution of same_process_name_as_parent in Training Data")
plt.show()

# for col in df_train.columns:
#     if col.endswith("_freq"):
#         plt.figure(figsize=(10, 6))
#         plt.hist(np.log(df_train[col]), bins=50)
#         plt.xlabel(col)
#         plt.ylabel("Count")
#         plt.title(f"Distribution of {col} in Training Data")
#         plt.show()

    