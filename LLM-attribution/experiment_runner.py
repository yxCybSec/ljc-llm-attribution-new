import pandas as pd
from typing import List, Tuple
from attribution_core import AuthorAttribution
from config import BASELINE_MODELS, EMBED_TYPES, EVALUATION_COLUMNS
from evaluation_metrics import EvaluationMetrics


class ExperimentRunner:
    def __init__(self, attribution_model: AuthorAttribution):
        self.attribution_model = attribution_model

    def compare_baseline_models(
            self,
            df_sub: pd.DataFrame,
            llm_results: List[pd.DataFrame],
            llm_methods: List[str],
            llm_models: List[str],
            n_eval: int,
            baseline_idx: int=len(BASELINE_MODELS),
            std_flag: bool = False,
    ) -> pd.DataFrame:

        ls_res_avg, ls_res_std = [], []
        baseline_idx = baseline_idx or len(BASELINE_MODELS)

        for key, val in list(BASELINE_MODELS.items())[:baseline_idx]:
            muti_avg, muti_std = self.attribution_model.run_aa_baseline(df_sub, val, EMBED_TYPES[key])
            ls_res_avg.append((key, val) + muti_avg + (0,))
            ls_res_std.append((key, val) + muti_std + (0,))
        for i, df_tmp in enumerate(llm_results):
            muti_avg, muti_std= EvaluationMetrics.evaluate_batch_predictions(df_tmp, n_eval)
            answer_tmp = df_tmp
            ls_res_avg.append((llm_methods[i], llm_models[i])+muti_avg+(abs(answer_tmp[answer_tmp.answer==-1]['answer'].astype('int').sum()),))
            ls_res_std.append((llm_methods[i], llm_models[i]) + muti_std + (None,))

        return pd.DataFrame(ls_res_avg, columns=EVALUATION_COLUMNS)