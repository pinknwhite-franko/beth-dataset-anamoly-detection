from model_class.feature_builder import FeatureBuilder



build_features = FeatureBuilder()
build_features = FeatureBuilder.fit(build_features, df_train)