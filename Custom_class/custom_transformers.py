from sklearn.base import BaseEstimator, TransformerMixin



# Custom transformer for frequency encoding
class FrequencyEncoder(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.freq_encoding = {}

    def fit(self, X, y=None):
        for column in X.columns:
            self.freq_encoding[column] = X[column].value_counts().to_dict()
        return self

    def transform(self, X, y=None):
        X_transformed = X.copy()
        for column in X.columns:
            X_transformed[column] = X[column].map(self.freq_encoding[column]).fillna(0)
        return X_transformed

# Custom transformer for target-guided mean encoding
class TargetMeanEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing=12):
        self.smoothing = smoothing
        self.target_means = {}
        self.global_mean = None

    def fit(self, X, y):
        self.global_mean = y.mean()
        for column in X.columns:
            category_means = y.groupby(X[column]).mean()
            category_counts = X[column].value_counts()
            smoothed_means = (category_counts * category_means + self.smoothing * self.global_mean) / (category_counts + self.smoothing)
            self.target_means[column] = smoothed_means
        return self

    def transform(self, X):
        X_transformed = X.copy()
        for column in X.columns:
            X_transformed[column] = X[column].map(self.target_means[column]).fillna(self.global_mean)
        return X_transformed
