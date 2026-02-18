from typing import Dict, Tuple, Any
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from fraud.models.evaluate import evaluate_model


def get_models(random_state: int = 42) -> Dict[str, Any]:
    models = {
        "random_forest": RandomForestClassifier(random_state=random_state),
        "extra_trees": ExtraTreesClassifier(random_state=random_state),
        "decision_tree": DecisionTreeClassifier(random_state=random_state),
        "xgboost": XGBClassifier(
            random_state=random_state,
            eval_metric="logloss"
        )
    }
    return models


def train_models(X_train, y_train) -> Dict[str, Any]:
    models = get_models()
    trained_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        print(f"{name} trained")

    return trained_models


def select_best_model(
    trained_models: Dict[str, Any],
    X_test,
    y_test
) -> Tuple[Any, str]:

    first_name = list(trained_models.keys())[0]
    best_model = trained_models[first_name]
    best_name = first_name
    best_score = evaluate_model(best_model, X_test, y_test)

    for name, model in trained_models.items():
        score = evaluate_model(model, X_test, y_test)

        if score > best_score:
            best_score = score
            best_model = model
            best_name = name

    return best_model, best_name
