import pandas as pd
import mlflow
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

dagshub.init(
    repo_owner="Herrens",
    repo_name="modelling_tuning",
    mlflow=True
)

mlflow.set_experiment("Telco-Churn-Tuning")

df = pd.read_csv("Membangun_model/data_clean/telco_clean.csv")

target_col = [col for col in df.columns if "Churn" in col][0]

X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {
    "C": [0.01, 0.1, 1, 10],
    "solver": ["liblinear"]
}

with mlflow.start_run():
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000),
        param_grid,
        cv=3,
        scoring="f1"
    )
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    mlflow.sklearn.log_model(best_model, "model")

    print("Best params:", grid.best_params_)
    print("Accuracy:", acc)
    print("F1:", f1)