import os

from sklearn.pipeline import Pipeline
from model_class.feature_builder import FeatureBuilder
from model_class.feature_builder_transformer import FeatureBuilderTransformer
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler
from matplotlib import pyplot as plt
# Get the current directory
current_directory = os.getcwd()

# Get the source file path
file_path = f'{current_directory}/datasets/raw'
file = f"{file_path}/labelled_training_data.csv"
df_train = pd.read_csv(file, header=0)

y_train = df_train['evil']
X_train = df_train.drop(columns=['sus', 'evil'])

# Get the source file path
file_path = f'{current_directory}/datasets/raw'
file = f"{file_path}/labelled_testing_data.csv"
df_test = pd.read_csv(file, header=0)
y_test = df_test['evil']
X_test = df_test.drop(columns=['sus', 'evil'])

build_features = FeatureBuilder()
FeatureBuilder.fit(build_features, X_train)

df_train = FeatureBuilder.transform(build_features, X_train)
df_test = FeatureBuilder.transform(build_features, X_test)

print(df_train.info())
print(df_test.info()) 

df_train.to_csv(f"{current_directory}/datasets/processed/feature_engineered_training_data.csv", index=False)
df_test.to_csv(f"{current_directory}/datasets/processed/feature_engineered_testing_data.csv", index=False)



