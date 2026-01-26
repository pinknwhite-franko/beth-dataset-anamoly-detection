import os
from model_class.feature_builder import FeatureBuilder
import pandas as pd

# Get the current directory
current_directory = os.getcwd()

# Get the source file path
file_path = f'{current_directory}/datasets/raw'
file = f"{file_path}/labelled_training_data.csv"
df_train = pd.read_csv(file, header=0)
df_train

build_features = FeatureBuilder()
FeatureBuilder.fit(build_features, df_train)

df = FeatureBuilder.transform(build_features, df_train)
print(df.columns)