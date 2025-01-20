import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
import pickle
import warnings
warnings.filterwarnings("ignore")

' ################### Transformer: Spaltenselektion ######'
class ColumnSelector(BaseEstimator, ClassifierMixin):
    def __init__(self, columns: list = []):
        """
        Transformer, bekommt ausgewählte Spaltennamen, die eingespeiste Menge wird
        auf diese Spalten reduziert"""
        self.columns = columns

    def fit(self, X, y=None):
        if not self.columns:
            raise ValueError("Es wurden im Spaltentransformer keine Spalten gesetzt")
        wrong_input_columns = [col for col in self.columns if col not in X.columns]
        if wrong_input_columns:
            raise ValueError(f"Folgende Eingabespalten des Transformers sind nicht auffindbar: {wrong_input_columns}!")
        return self

    def transform(self, X, y=None):
        """
        Wählt die angegebenen Spalten aus X aus.
        """
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Eingabedaten des Transformers sind kein pandas Dataframe!")
        return X[self.columns]


# ' ########### Daten holen für Tests ##########'
# with open("strokes_split_data.pkl", "rb") as file:
#     df = pickle.load(file)
#     X_train = df["X_train"]
#     Y_train = df["Y_train"]
#     X_test = df["X_test"]
#     Y_test = df["Y_test"]

# ' ################### Test Spaltentransformer ##########'
# print("----- Test Spaltenselektion durch Transformer ---------")
# columnselector = ColumnSelector(["age", "at_least_one_risk"])
# columnselector.fit(X_train)
# X_train_reduced = columnselector.transform(X_train)
# print(X_train.columns)
# print(X_train_reduced.columns)
#
# ' ----------------------Steuerung: verwendete Spalten --------------------'
# drop_columns = [
#     ,'age'
#     'gender'
#      ,'hypertension'
#      ,'heart_disease'
#      ,'Residence_type'
#      ,'avg_glucose_level'
#      ,'bmi'
#     , 'smoking_status'
#      ,'age_above_60'
#      , 'high_glucose'
#     , 'did_smoke'
#     , 'heart_risk'
#     # , 'at_least_one_risk'
#     , 'at_least_one_risk_and_high_age'
#     , 'all_risks'
#       , 'risk_sum'
#     ]