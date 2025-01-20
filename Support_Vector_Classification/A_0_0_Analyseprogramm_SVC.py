from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, recall_score, precision_score
from sklearn.metrics import classification_report
import pickle
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")


' ##################### STEUERUNGSBLOCK ################################'
resampling = "partial_smote"           # none, smote, oder partial_smote
classweight_y = 4           # Gewichtung für strokes
gamma = 0.5                 # gamma für SVC
C = 0.005                       # C für SVC
scaling_method = "Standard"   # "Standard", "MinMax", oder leer (dann unskaliert)
show_rbf_plot = 1              # opb plot für brbf-bereiche gezeigt werden soll
show_threshold_curve = 1     # 1: thresholdingkurve anzeigen, 0 sonst

'------------------------ Steuerung: Sampling ---------------------------------'
print(f"--------------------- Sampling -------------------------------")
print("resampling: ", resampling)
if resampling == "smote":
    with open('strokes_split_data_smoted.pkl', 'rb') as f:
        df = pickle.load(f)
    X_train = df["X_train_resampled"]
    Y_train = df["Y_train_resampled"]
    X_test = df["X_test"]
    Y_test = df["Y_test"]
elif resampling == "partial_smote":
    with open('strokes_split_data_partially_smoted.pkl', 'rb') as f:
        df = pickle.load(f)
    X_train = df["X_train_partially_resampled"]
    Y_train = df["Y_train_partially_resampled"]
    X_test = df["X_test"]
    Y_test = df["Y_test"]
elif resampling == "none":
    with open('strokes_split_data.pkl', 'rb') as f:
        df = pickle.load(f)
    X_train = df["X_train"]
    Y_train = df["Y_train"]
    X_test = df["X_test"]
    Y_test = df["Y_test"]

' ----------------------Steuerung: verwendete Spalten --------------------'
drop_columns = [
    #   ,'age'
    'gender'
     ,'hypertension'
     ,'heart_disease'
     ,'Residence_type'
     ,'avg_glucose_level'
     ,'bmi'
    , 'smoking_status'
     ,'age_above_60'
     , 'high_glucose'
    , 'did_smoke'
    , 'heart_risk'
    # , 'at_least_one_risk'
    , 'at_least_one_risk_and_high_age'
    , 'all_risks'
      , 'risk_sum'
    ]
X_train = X_train.drop(columns=drop_columns)
X_test = X_test.drop(columns=drop_columns)

' --------------------------- Steuerung: Modellparameter ---------------------------'
print("----------------------- Modellparameter ----------------------")
print("classweight y=1: \t", classweight_y, "\nthreshold Kurve: ", show_threshold_curve,
      "\ngamma: ", gamma, "\nC: ", C)

' ------------------------------- Steuerung Scaling ----------------------------------'
print("--------------- Skalierung ----------")
print(scaling_method)
if scaling_method == "Standard":
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
elif scaling_method == "MinMax":
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
else:
    1==1

' ######################## BLOCK MODELLINSTANZ UND FITTING #######################'
' ------------------------ Modellinstanz ------------------------'
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

' --------------------------------- fitting ------------------------------'
svc.fit(X_train, Y_train)
' ############ BLOCK REPORT AUF TRAININGSMENGE UND TESTMENGE ######'
def print_scores(chosen_set):
    apply_on = chosen_set
    if apply_on == "train":
        X_score = X_train
        Y_score = Y_train
    elif apply_on == "test":
        X_score = X_test
        Y_score = Y_test
    print(f"--------- Confusion Matrix und Scoring Report {chosen_set} ----------")
    Y_pred = svc.predict(X_score)
    cm = confusion_matrix(Y_score, Y_pred)
    print(f"Confusion Matrix auf {apply_on}:\n", cm)
    target_names = ['stroke = 0', 'stroke = 1']
    print(classification_report(Y_score, Y_pred, target_names=target_names))

print_scores("train")
print_scores("test")

' #################### BLOCK PLOT ###############################'
if show_rbf_plot == 1:
    if resampling == "none":
        X_train_temp = df["X_train"]
    elif resampling == "smote":
        X_train_temp = df["X_train_resampled"]
    elif resampling == "partial_smote":
        X_train_temp = df["X_train_partially_resampled"]
    Y_train_2D = Y_train
    X_test_temp = df["X_test"]
    X_test_temp["stroke"] = df["Y_test"]
    X_test_temp_strokes = X_test_temp[X_test_temp["stroke"] == 1][["age", "avg_glucose_level"]]
    X_train_2D = X_train_temp[["age", "avg_glucose_level"]]
    # Modell nochmal fitten
    svc.fit(X_train_2D, Y_train)
    # Achsen für 2d-Plot bestimmen
    X_train_2D_maxima = X_train_2D.max(axis = 0)
    X_train_2D_minima = X_train_2D.min(axis = 0)
    # für Färbung von X_train
    colors = ["blue" if y == 0 else "red" for y in Y_train_2D]
    # plot-rumpf
    plt.figure(figsize=(12, 10))
    xx, yy = np.meshgrid(np.linspace(X_train_2D_minima[0], X_train_2D_maxima[0], 200), np.linspace(X_train_2D_minima[1], X_train_2D_maxima[1], 200))
    # grid aufbauen
    Z = svc.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # visualize decision function for these parameters
    plt.title(f"gamma = {gamma}, C={C}", size="medium")
    # visualize parameter's effect on decision function
    plt.pcolormesh(xx, yy, -Z, cmap=plt.cm.RdBu)
    # Trainingsdaten und rbf-Bereiche
    plt.scatter(X_train_2D["age"],
                X_train_2D["avg_glucose_level"],
                s=50
                #, c=Y_train_2D
                , c=colors
                #,cmap=plt.cm.RdBu_r,
                ,edgecolors="k"
                , label="Trainingsdaten")
    # Testdaten
    plt.scatter(
        X_test_temp_strokes["age"].to_numpy(),
        X_test_temp_strokes["avg_glucose_level"].to_numpy(),
        color="yellow",
        label="Testdaten (nur strokes, also y=1)",
        edgecolors="k",
        s=50
    )
    # Achsentitel und Ticks hinzufügen
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Average Glucose Level", fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(loc="upper right")  # Legende für die Datenpunkte
    plt.axis("tight")
    plt.show()


' ################## BLOCK THRESHOLDING: manipulieren der Output-Wahrscheinlichkeiten ###########'
if show_threshold_curve == 1:
    # nochmal fitten
    svc.fit(X_train, Y_train)
    y_proba = svc.predict_proba(X_test)[:,1]
    y_pred = svc.predict(X_test)
    global_thresholds = np.arange(0.01, 0.25, 0.001)
    results = []
    # für alle thresholds recall und precision des
    # gefitteten modells berechnen anhand manipulierter prognose
    for global_threshold in global_thresholds:
        # Anpassung der Vorhersage basierend auf dem aktuellen Threshold
        y_pred_adjusted = (y_proba >= global_threshold).astype(int)
        # Berechnung Confusion Matrix
        cm = confusion_matrix(Y_test, y_pred_adjusted)
        # Recall und Precision für strokes rausholen
        recall = recall_score(Y_test, y_pred_adjusted)
        precision = precision_score(Y_test, y_pred_adjusted)
        results.append([global_threshold, recall, precision])
    # Umwandlung in array
    results_array = np.array(results)
    # Extrahiere die global_thresholds, Recall und Precision aus dem results_array
    thresholds = results_array[:, 0]
    recall_values = results_array[:, 1]
    precision_values = results_array[:, 2]
    # Plot
    plt.figure(figsize=(8, 6))
    plt.plot(thresholds, recall_values, marker='o', label='Recall', color='blue', markersize=3)
    plt.plot(thresholds, precision_values, marker='o', label='Precision', color='red', markersize=3)
    # Achsen, Titel, Legende
    plt.xlabel('Thresholds')
    plt.ylabel('Recall; Precision')
    plt.title('Recall und Precision bei verschiedenen Thresholds')
    plt.legend()
    plt.grid(True)
    plt.show()



