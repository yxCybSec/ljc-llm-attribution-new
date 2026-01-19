import numpy as np
import pandas as pd
from sklearn import metrics
from typing import Tuple


class EvaluationMetrics:
    @staticmethod
    def calculate_metrics(y_true, y_pred) -> Tuple[float, float, float, float]:
        acc = round(metrics.accuracy_score(y_true, y_pred) * 100, 2)
        f1_w = round(metrics.f1_score(y_true, y_pred, average='weighted') * 100, 2)
        f1_micro = round(metrics.f1_score(y_true, y_pred, average='micro') * 100, 2)
        f1_macro = round(metrics.f1_score(y_true, y_pred, average='macro') * 100, 2)

        return acc, f1_w, f1_micro, f1_macro

    @staticmethod
    def evaluate_batch_predictions(
            df_res_all: pd.DataFrame,
            n_eval: int = 10
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:
        ls_acc, ls_f1_w, ls_f1_micro, ls_f1_macro = [], [], [], []

        for i in range(0, len(df_res_all.index), n_eval):
            df_reps = df_res_all[i: i + n_eval]
            acc, f1_w, f1_micro, f1_macro = EvaluationMetrics.calculate_metrics(
                df_reps["label"],
                df_res_all["answer"]
            )
            ls_acc.append(acc)
            ls_f1_w.append(f1_w)
            ls_f1_micro.append(f1_micro)
            ls_f1_macro.append(f1_macro)

        muti_avg = (
            round(np.mean(ls_acc), 2),
            round(np.mean(ls_f1_w), 2),
            round(np.mean(ls_f1_micro), 2),
            round(np.mean(ls_f1_macro), 2)
        )
        muti_std = (
            round(np.std(ls_acc), 2),
            round(np.std(ls_f1_w), 2),
            round(np.std(ls_f1_micro), 2),
            round(np.std(ls_f1_macro), 2)
        )

        return muti_avg, muti_std