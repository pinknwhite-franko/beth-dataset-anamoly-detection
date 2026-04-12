import os
import sys

# sys.path.append('Users/cheesecake/Github/beth-dataset-anamoly-detection/model_class')

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

# Get the source file path
file_path = f'{current_directory}/datasets/raw'
file = f"{file_path}/labelled_validation_data.csv"
df_validation = pd.read_csv(file, header=0)
y_validation = df_validation['evil']
X_validation = df_validation.drop(columns=['sus', 'evil'])

build_features = FeatureBuilder()
FeatureBuilder.fit(build_features, X_train)

df_train = FeatureBuilder.transform(build_features, X_train)

# df_train[["stackAddresses_jump_std", "stackAddresses_jump_mean", "stackAddresses_jump_max","stackAddresses_len"]]

for col in ["stackAddresses_len"]:
    # plot the distribution of stackAddresses_jump_std for normal and abnormal samples
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 3, 1)
    plt.hist(df_train[[col]], bins=50, alpha=0.5, label='Normal')
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# df_train

# df_train["evil"] = y_train.values
# df_test = FeatureBuilder.transform(build_features, X_test)
# df_test["evil"] = y_test.values
# df_validation = FeatureBuilder.transform(build_features, X_validation)
# df_validation["evil"] = y_validation.values
# df_validation.to_csv(f"{current_directory}/datasets/processed/feature_engineered_validation_data.csv", index=False)

# print("df_train: ==============================")
# print(df_train.info())
# print(df_train.count())

# print("df_test: ==============================")
# print(df_test.info()) 
# print(df_test.count())

# df_train.to_csv(f"{current_directory}/datasets/processed/feature_engineered_training_data.csv", index=False)
# df_test.to_csv(f"{current_directory}/datasets/processed/feature_engineered_testing_data.csv", index=False)




