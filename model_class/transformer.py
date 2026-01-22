




class FrequencyEncoder:
    def __init__(self, col):
        self.col = col
        self.counts_ = None

    def fit(self, df):
        self.counts_ = df[self.col].value_counts()
        return self

    def transform(self, df):
        return df[self.col].map(self.counts_).fillna(0).astype(int)
    

