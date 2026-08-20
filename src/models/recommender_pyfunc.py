"""MLflow pyfunc wrapper around SVDModel so it's logged in proper MLflow
Model format (servable via mlflow.pyfunc.load_model, registerable in the
Model Registry) instead of as a bare pickle.

Expected model_input: a pandas DataFrame with columns:
  - user_id (int, required)
  - k (int, optional, defaults to 10)
Output: for each input row, a ranked list of recommended movie_ids.
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
