from sklearn.base import TransformerMixin, BaseEstimator
import pandas as pd
import numpy as np

no_of_images_index, no_of_quotes_index, mean_paragraph_len_index, body_len_index = 6, 7, 3, 2
# numerical_feature=['day_of_week','subject_len', 'body_len', 'mean_paragraph_len','no_of_CTA', 'mean_CTA_len', 'no_of_image',  'no_of_quotes','no_of_emoticons']
# categorical_features=['is_discount', 'is_price', 'is_urgency', 'is_personalised',  'is_weekend', 'sender', 'category', 'product', 'target_audience']
# one_hot_features=['times_of_day']
class AttributeCombiner(BaseEstimator, TransformerMixin):

    def __init__(self, add_total_visual_content=False, add_paragraph_body_ratio=False):
        self.add_total_visual_content=add_total_visual_content
        self.add_paragraph_body_ratio=add_paragraph_body_ratio


    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        total_visual_content = X[:, no_of_images_index] + X[:, no_of_quotes_index]
        paragraph_body_ratio = X[:, mean_paragraph_len_index] / X[:, body_len_index]
        if self.add_total_visual_content and not self.add_paragraph_body_ratio:
            return np.c_[X, total_visual_content]
        elif self.add_paragraph_body_ratio and not self.add_total_visual_content:
            return np.c_[X, paragraph_body_ratio]
        elif self.add_paragraph_body_ratio and self.add_total_visual_content:
            return np.c_[X, total_visual_content, paragraph_body_ratio]
        else:
            return X

class ColumnStringConverter(BaseEstimator, TransformerMixin):
    def __init__(self, convert=True):
        self.convert=convert
      
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for idx, col in enumerate(X.columns, start=0):

            X[col] = X[col].astype(str)

        return X