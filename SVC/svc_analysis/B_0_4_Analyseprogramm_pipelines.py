from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.metrics import classification_report

import sys
import os

# Projektverzeichnis hinzufügen
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Absolute Importe
from transformers_and_threshold_model.B_0_1_Transformer_ColumnSelector import ColumnSelector
from transformers_and_threshold_model.B_0_2_Transformer_PartialSmoteResampler import PartialSmoteResampler
from transformers_and_threshold_model.B_0_3_ThresholdClassifier import ThresholdedClassifier
import pickle

' ######################### Daten laden ###################'
with open('strokes_split_data.pkl', 'rb') as f:
    df = pickle.load(f)
X_train = df["X_train"]
Y_train = df["Y_train"]
X_test = df["X_test"]
Y_test = df["Y_test"]

' ##################### STEUERUNGSBLOCK ################################'
resampling = "none"           # none, oder partial_smote
classweight_y = 10           # Gewichtung für strokes
gamma = 0.5                 # gamma für SVC
C = 0.005                       # C für SVC
thresholded_classifier = 1     # 1 = pipeline mit thresholding.; 0= pipeline ohne!
threshold = 0.07
'######################### Zu verwendende Spalten #########################'
' ----------------------Steuerung: verwendete Spalten --------------------'
run_model_on_these_column = [
    'age'
    # ,'gender'
    #  ,'hypertension'
    #  ,'heart_disease'
    #  ,'Residence_type'
    #  ,'avg_glucose_level'
    #  ,'bmi'
    # , 'smoking_status'
    #  ,'age_above_60'
    #  , 'high_glucose'
    # , 'did_smoke'
    # , 'heart_risk'
    , 'at_least_one_risk'
    # , 'at_least_one_risk_and_high_age'
    # , 'all_risks'
    #   , 'risk_sum'
    ]

'######################### Sampling #########################'
print(f"--------------------- Sampling -------------------------------")
print("resampling: ", resampling)
if resampling == "partial_smote":
    # Filterbedingung für X (Einschränkung auf Teilmenge)
    X_train_filtered = X_train[((X_train["age"]>=55)&(X_train["age"]<=60)) | (X_train["age"]>=75)]
    X_partial_indices = X_train_filtered.index.tolist()
elif resampling == "none":
    X_partial_indices = X_train.index.tolist()

' ############## Classifier-Instanz ####################'
svc = SVC(C=C  # erhöhen macht grenzen schärfer!
          , kernel='rbf'  # wir verwenden immer rbf
          , degree=3   # nicht relevant für rbf
          , gamma= gamma  # relevant. Große werte machen kleine Bereiche und andersrum!
          , coef0=0.0  # nicht relevant für rbf
          , shrinking=True  # wirkt sich auf Laufzeit abh. v. Iterationszahl aus. Qualitativ hier nicht relevant
          , probability=True # steuert, ob Wahrscheinlichkeiten ausgegeben werden!
          , tol=0.001  # Toleranz stopping criterion
          , cache_size=200  # prozessorparameter
          , class_weight={0: 1, 1: classweight_y}  # WICHTIG
          , verbose=False  # printing parameter
          , max_iter=-1  # iterationen. -1 = unbeschränkt
          , decision_function_shape='ovo'  # ovr= one vs. rest, ovo = one vs. one
          , break_ties=False # für ovr wichtig
          , random_state=42)

' ############### Instanz Scaler #############'
standard_scaler = StandardScaler()
' ###################### Instanz Spaltentransformator ##############'
column_selector = ColumnSelector(columns= run_model_on_these_column)
' ###################### Instan Smote Resampler ####################'
partial_smote_resampler = PartialSmoteResampler(resampling_indices=X_partial_indices)
' ###################### Instanz Thresholded Classifier ############'
thresholded_svc = ThresholdedClassifier(classifier=svc, threshold=threshold)


' ################ Pipeline-Selektion aufgrund von Steuerung ganz oben ####'
if thresholded_classifier == 1:
    ' ##################### Pipeline mit Threshold-Classifier ###########'
    pipeline_thresholded_classifier = Pipeline([
        ("columnselector", column_selector),
        # ("partialsmote", partial_smote_resampler),   #alleinstehend funktioniert er; hier eingebettet NICHT!
        ("scaler", standard_scaler),
        ("thresholdclassifier", thresholded_svc),
    ])
    chosen_pipeline = pipeline_thresholded_classifier
# Funktioniert!! Allerdings aktuell noch NICHT FÜR PARTIALSMOTE.
else:
    ' ##################### Pipeline mit normalem Classifier ###########'
    pipeline_regular_classifier = Pipeline([
        ("columnselector", column_selector),
        # ("partialsmote", partial_smote_resampler),  # alleinstehend funktioniert er; hier eingebettet NICHT!
        ("scaler", standard_scaler),
        ("classifier", svc),
    ])
    chosen_pipeline = pipeline_regular_classifier
# Funktioniert!! Allerdings aktuell noch NICHT FÜR PARTIALSMOTE.

' ################ Fitting ###########'
chosen_pipeline.fit(X_train, Y_train)

' ############ Reporting ######'
def print_scores(chosen_set):
    apply_on = chosen_set
    if apply_on == "train":
        X_score = X_train[run_model_on_these_column]
        Y_score = Y_train
    elif apply_on == "test":
        X_score = X_test[run_model_on_these_column]
        Y_score = Y_test
    print(f"--------- Confusion Matrix und Scoring Report {chosen_set} ----------")
    Y_pred = chosen_pipeline.predict(X_score)
    cm = confusion_matrix(Y_score, Y_pred)
    print(f"Confusion Matrix auf {apply_on}:\n", cm)
    target_names = ['stroke = 0', 'stroke = 1']
    print(classification_report(Y_score, Y_pred, target_names=target_names))

print_scores("train")
print_scores("test")