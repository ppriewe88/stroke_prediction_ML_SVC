from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC
import pickle

' ################## Classifier: Classifier mit Thresholding #########################'
class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, classifier = None, threshold=0.1):
        self.classifier = classifier
        self.threshold = threshold

    def fit(self, X, y):
        self.classifier.fit(X, y)
        return self

    def predict_proba(self, X):
        return self.classifier.predict_proba(X)

    def predict(self, X):
        y_proba = self.predict_proba(X)[:, 1]  # Wahrscheinlichkeiten für Klasse 1
        return (y_proba >= self.threshold).astype(int)


# ' ########### Daten holen für Tests ##########'
# with open("strokes_split_data.pkl", "rb") as file:
#     df = pickle.load(file)
#     X_train = df["X_train"]
#     Y_train = df["Y_train"]
#     X_test = df["X_test"]
#     Y_test = df["Y_test"]
#
# ' ################### Test ThresholdedClassifier ##########'
# print("----- Test thresholded classifier ---------")
# svc = SVC(C=1  # erhöhen macht grenzen schärfer!
#           , kernel='rbf'  # wir verwenden immer rbf
#           , degree=3   # nicht relevant für rbf
#           , gamma= 0.2  # relevant. Große werte machen kleine Bereiche und andersrum!
#           , coef0=0.0  # nicht relevant für rbf
#           , shrinking=True  # wirkt sich auf Laufzeit abh. v. Iterationszahl aus. Qualitativ hier nicht relevant
#           , probability=True # steuert, ob Wahrscheinlichkeiten ausgegeben werden!
#           , tol=0.001  # Toleranz stopping criterion
#           , cache_size=200  # prozessorparameter
#           , class_weight={0: 1, 1: 3}  # WICHTIG
#           , verbose=False  # printing parameter
#           , max_iter=-1  # iterationen. -1 = unbeschränkt
#           , decision_function_shape='ovo'  # ovr= one vs. rest, ovo = one vs. one
#           , break_ties=False # für ovr wichtig
#           , random_state=42)
#
# svc.fit(X_train, Y_train)
# print("Original-Classifier: \n", svc.predict_proba(X_test[:5]), svc.predict(X_test[:5]))
# thresholdedSVC = ThresholdedClassifier(classifier = svc, threshold=0.125)
# thresholdedSVC.fit(X_train, Y_train)
# print("Thresholded-Classifier: \n", thresholdedSVC.predict_proba(X_test[:5]), thresholdedSVC.predict(X_test[:5]))
#
