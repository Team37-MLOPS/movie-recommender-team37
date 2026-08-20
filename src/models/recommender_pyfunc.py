"""MLflow pyfunc wrapper around SVDModel/ALSModel for the Model Registry.

Input columns: user_id (int, required), k (int, optional, default 10).
"""
import pickle

import mlflow.pyfunc
import pandas as pd


class RecommenderPyfunc(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        with open(context.artifacts["model"], "rb") as f:
            self.model = pickle.load(f)

    def predict(self, context, model_input: pd.DataFrame, params=None):
        recommendations = []
        for _, row in model_input.iterrows():
            user_id = int(row["user_id"])
            k = int(row["k"]) if "k" in row and not pd.isna(row["k"]) else 10
            recs = self.model.recommend(user_id, k, exclude_items=set())
            recommendations.append(recs)
        return recommendations
