from catboost import CatBoostClassifier

catboost_model = CatBoostClassifier(iterations=10000, eval_metric="Accuracy")
