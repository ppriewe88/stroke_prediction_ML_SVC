import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")


"""
ALLEINSTEHEND funktioniert der Smote-Transformer für einen einfachen Test bereits!
ALLERDINGS noch nicht eingebettet in der Pipeline! Dort bekomme ich irgendwelche Dimensionsprobleme...
evt. mal bei Stackoverflow posten??

Hinweis: Hier muss ich eventuell die Pipeline aus imblearn verwenden, NICHT aus sklearn!
siehe Beschreibung bei imblearn:
"A surprising behaviour of the imbalanced-learn pipeline is that it breaks the 
scikit-learn contract where one expects estimmator.fit_transform(X, y) to be 
equivalent to estimator.fit(X, y).transform(X).
The semantic of fit_resample is to be applied only during the fit stage. 
Therefore, resampling will happen when calling fit_transform while it 
will only happen on the fit stage when calling fit and transform separately. 
Practically, fit_transform will lead to a resampled dataset while 
fit and transform will not."
"""


' ################### Transformer: Smote auf Indexmengen von X ######'
class PartialSmoteResampler(BaseEstimator, ClassifierMixin):
    def __init__(self, resampling_indices = None, smote_k_neighbors: int = 1):
        """
        Transformer, um SMOTE auf bestimmte Zeilen eines DataFrames anzuwenden.
        Parameters:
        - smote_k_neighbors: integer, parameter für smoting-nachbarschaften
        - indices: Liste der Zeilenindizes (integer), auf die SMOTE angewendet werden soll.
        """
        if resampling_indices == None:
            raise ValueError("Resampling: Keine Indizes übergeben!")
        self.smote_k_neighbors = smote_k_neighbors
        self.resampling_indices = resampling_indices
        # Smote initialisieren
        self.smote_ = SMOTE(k_neighbors=smote_k_neighbors, random_state=42)

    def fit(self, X, y=None):
        """
        Initialisiert SMOTE und überprüft die Eingaben.
        """
        # print("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Resampling: X muss pandas Dataframe sein!")
        if not isinstance(self.resampling_indices, (list, np.ndarray)):
            raise TypeError("Resampling: Indizes müssen Liste oder np-array sein!")
        if len(self.resampling_indices) == 0:
            raise ValueError("Resampling: Keine Resampling-Teilmenge angegeben!")
        if self.resampling_indices == None:
            raise ValueError("Resampling: Keine Indizes übergeben!")
        # # Smote initialisieren
        # self.smote_ = SMOTE(k_neighbors=self.smote_k_neighbors, random_state=42)

        return self

    def transform(self, X, y):
        """Wendet Resampling nur auf X an (ohne y)"""
        # print("YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY")
        raise NotImplementedError("Für PartialSmoteResampler wird fit_transform benötigt.")

    def fit_transform(self, X, y):
        """Verwendet fit_transform, um Resampling auf X und y anzuwenden"""
        # print("ZZZZZZZZZZZZZZZZZZZZZZZZZZ")
        if self.resampling_indices is None:
            raise ValueError("Resampling: Keine Indizes übergeben!")
        # print("111111")
        # Teilmengen von X und y für Resampling
        X_partial = X.loc[self.resampling_indices]
        y_partial = y.loc[self.resampling_indices]
        # print("222222")
        # Komplementärmengen davon
        mask = ~X.index.isin(self.resampling_indices)
        X_rest = X[mask]
        y_rest = y[mask]
        # print("333333")
        # Resampling der Teilmengen
        X_resampled, y_resampled = self.smote_.fit_resample(X_partial, y_partial)

        # print("444444")
        # Vereinigung bilden
        X_combined = pd.concat([X_rest, pd.DataFrame(X_resampled, columns=X.columns)])
        y_combined = pd.concat([y_rest, pd.Series(y_resampled, name=y.name)])
        # print(y_combined)
        # print(y_combined.shape)
        # print(X_combined.shape)
        # Indizes zurücksetzen, um Duplikate zu vermeiden
        X_combined = X_combined.reset_index(drop=True)
        y_combined = y_combined.reset_index(drop=True)

        return X_combined.to_numpy(), y_combined.to_numpy()

# ' ########### Daten holen für Tests ##########'
# with open("strokes_split_data.pkl", "rb") as file:
#     df = pickle.load(file)
#     X_train = df["X_train"]
#     Y_train = df["Y_train"]
#     X_test = df["X_test"]
#     Y_test = df["Y_test"]
#     X_train["gender"]= X_train["gender"].astype(int)
#
# # Filterbedingung für X (Einschränkung auf Teilmenge)
# X_train_filtered = X_train[((X_train["age"]>=55)&(X_train["age"]<=60)) | (X_train["age"]>=75)]
# X_partial_indices = X_train_filtered.index.tolist()
#
# ' ################### Test Smote-Resampler ##########'
# print("----- Test strokes erhöhen durch Smote-Resampler ---------")
# data_X = {
#         "age": [99,88,3, 10,20,30,40,50],
#         "bmi": [10,20,30, 1,2,3,4,5]
# }
# X_test1 = pd.DataFrame(data_X)
#
# data_Y = {"stroke": [0, 0, 0, 0, 0, 0, 1, 1]}
# Y_test1 = pd.Series(data_Y["stroke"])
#
# partialsmoteresampler = PartialSmoteResampler(resampling_indices=[2,3,4,5,6,7])
# X_resampled, Y_resampled = partialsmoteresampler.fit_transform(X_test1, Y_test1)
# print(type(X_resampled), type(Y_resampled))
# # print(pd.concat([X_resampled, Y_resampled], axis = 1))
# print(pd.concat([X_test1, Y_test1], axis = 1))
# # Horizontales Stapeln der Arrays
# resampled_combined = np.hstack((X_resampled, Y_resampled.reshape(-1, 1)))
# # Ausgabe
# print(resampled_combined)
#
# partialsmoteresampler = PartialSmoteResampler(resampling_indices=X_partial_indices)
# X_resampled, Y_resampled = partialsmoteresampler.fit_transform(X_train, Y_train)
# print(np.hstack((X_resampled, Y_resampled.reshape(-1, 1))))