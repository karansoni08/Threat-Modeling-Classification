from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


def train_lightgbm(X_train, y_train):
    model = LGBMClassifier(
        boosting_type="gbdt",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=63,
        random_state=42
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, num_classes: int):
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="multi:softprob" if num_classes > 2 else "binary:logistic",
        num_class=num_classes if num_classes > 2 else None,
        eval_metric="mlogloss" if num_classes > 2 else "logloss",
        random_state=42
    )
    model.fit(X_train, y_train)
    return model
