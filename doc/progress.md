Best scores:
    Date        Type                       MyLLScore  KaggleLLScore  Percent  Submission Tag
    ========================================================================================
    04/27/2016  BaseLinePredictor          20.61577        20.25113           Submission00
    04/27/2016  NearestNeighborsPredictor  14.37823        13.94696           Submission01
    04/27/2016  NearestNeighborsPredictor   5.07153         5.20698           Submission02
    04/28/2016  NearestNeighborsPredictor   3.70919         3.59477       81  Submission03
    04/28/2016  NearestNeighborsPredictor   1.00052         1.00036       66  Submission04
    04/28/2016  DecisionTreePredictor       0.94061         0.90950       59  Submission05
    04/29/2016  DecisionTreePredictor       0.92471         0.90939       58  Submission06
    04/29/2016  RandomForestPredictor       0.88390         0.86738       50  Submission07
    04/29/2016  RandomForestPredictor       0.85608         0.84534       47  Submission08
    04/29/2016  RandomForestPredictor       0.82221         0.84454       46  Submission09
    05/03/2016  RandomForestPredictor       0.81721         0.81486       40  Submission10
    05/04/2016  RandomForestPredictor       0.79887         0.79934       35  Submission11
    05/04/2016  RandomForestPredictor       0.79229         0.79204       34  Submission12
    05/04/2016  RandomForestPredictor       0.79955         0.79167       35  Submission13

Submission05: Started using DecisionTreeClassifier(max_depth=6).
Submission06: Broke down SexuponOutcome into IsIntact and IsMale.
Submission07: Started using SelectKBest.
Submission08: Started using GridSearchCV on RandomForestPredictor.
Submission09: Set n_estimators on RandomForestClassifier.
Submission10: Started using hour of the day as a feature.
Submission11: Added IsFivePM and IsSixPM features.
Submission12: Added IsThreePM feature.
Submission13: Retrain on *all* the training data.
