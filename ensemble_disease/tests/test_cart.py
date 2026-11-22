import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from ensemble_disease.cart import CARTClassifier, gini, gini_table
from ensemble_disease.toy_data import GOLF, STUDENT


def test_gini_basic():
    assert gini(["a", "a"]) == 0
    assert abs(gini(["a", "b"]) - 0.5) < 1e-12


def test_golf_root_matches_annex():
    t = gini_table(GOLF, ["Outlook", "Temp", "Humidity", "Wind"], "Decision")
    assert round(t["Outlook"], 3) == 0.343  # slide: 0.342 (rounded sub-terms)
    assert round(t["Temp"], 3) == 0.440     # slide: 0.439
    assert round(t["Humidity"], 3) == 0.367
    assert round(t["Wind"], 3) == 0.429     # slide: 0.428
    assert min(t, key=t.get) == "Outlook"


def test_golf_sunny_wind_corrected():
    sunny = GOLF[GOLF.Outlook == "Sunny"]
    w = sunny.groupby("Wind")["Decision"].apply(list)
    assert round(gini(w["Weak"]), 3) == 0.444   # slide says 0.266 (typo)
    assert round(gini(w["Strong"]), 3) == 0.5   # slide says 0.2 (typo)


def test_student_root_matches_course():
    t = gini_table(STUDENT, list(STUDENT.columns[:-1]), "Success")
    assert round(t["Attendance"], 4) == 0.3265
    assert round(t["StudyTime"], 4) == 0.4490  # slide: 0.4499 (rounded sub-terms)


def test_student_tree_and_fit():
    tree = CARTClassifier().fit(STUDENT, "Success")
    assert tree.root.feature == "Attendance"
    assert tree.root.children["High"].feature == "StudyTime"
    assert tree.root.children["Low"].feature == "Homework"
    assert tree.score(STUDENT) == 1.0


def test_golf_final_tree_and_sklearn_agreement():
    tree = CARTClassifier().fit(GOLF, "Decision")
    assert tree.root.feature == "Outlook"
    assert tree.root.children["Sunny"].feature == "Humidity"
    assert tree.root.children["Rain"].feature == "Wind"
    X = pd.get_dummies(GOLF.drop(columns="Decision"))
    sk = DecisionTreeClassifier(criterion="gini", random_state=0).fit(X, GOLF.Decision)
    assert sk.score(X, GOLF.Decision) == tree.score(GOLF) == 1.0
