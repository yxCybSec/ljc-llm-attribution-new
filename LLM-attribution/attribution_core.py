import random
import json
import time
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Dict, Tuple, Any
from embedding_utils import EmbeddingGenerator
from evaluation_metrics import EvaluationMetrics


class AuthorAttribution:
    def __init__(self, embedding_generator=None):
        self.embedding_generator = embedding_generator or EmbeddingGenerator()

    def sampler_aa_fn_pro(
            self,
            df: pd.DataFrame,
            n: int,
            reps: int
    ) -> pd.DataFrame:
        dict_to_df = []
        ls_unique_author = df.id.unique().tolist()

        for _ in range(reps):
            candidate_authors = random.sample(ls_unique_author, n)
            ls_unique_author = [e for e in ls_unique_author if e not in candidate_authors]
            ls_queries, ls_potential_texts = [], []
            dict_row = {}

            for author_id in candidate_authors:
                text, text_same_author = df.loc[author_id == df.id].text.sample(2, replace=True)
                ls_queries.append(text)
                ls_potential_texts.append(text_same_author)

            dict_row["query_text"] = ls_queries
            dict_row["potential_text"] = ls_potential_texts
            dict_to_df.append(dict_row)

        return pd.DataFrame(dict_to_df)

    def run_aa_baseline(
            self,
            df_sub: pd.DataFrame,
            model_name: str,
            baseline_type: str = 'bert'
    ) -> Tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]]:

        ls_acc, ls_f1_w, ls_f1_micro, ls_f1_macro = [], [], [], []

        for i in df_sub.index:
            ls_query_text = df_sub.loc[i, 'query_text']
            ls_potential_text = df_sub.loc[i, 'potential_text']

            embed_query_texts = self.embedding_generator.normalize_embeddings(
                self.embedding_generator.embed_texts(ls_query_text, model_name, baseline_type)
            )
            embed_potential_texts = self.embedding_generator.normalize_embeddings(
                self.embedding_generator.embed_texts(ls_potential_text, model_name, baseline_type)
            )

            preds = embed_query_texts @ embed_potential_texts.T
            preds = F.softmax(preds, dim=-1)
            labels = np.arange(0, len(ls_query_text))

            acc, f1_w, f1_micro, f1_macro = EvaluationMetrics.calculate_metrics(
                labels,
                preds.argmax(-1).cpu().numpy()
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

    def run_aa_llm(
            self,
            df: pd.DataFrame,
            method: str,
            model_name: str,
            prompt_input: str,
            system_msg: str,
            client: Any,
            n_eval: int = 10
    ) -> pd.DataFrame:
        df_res_all = pd.DataFrame()

        for i in df.index:
            ls_reps = []
            text_label_map = {}
            sampled_queries = []

            ls_query_text, ls_potential_text = df.loc[i, 'query_text'], df.loc[i, 'potential_text']
            random.seed(0)

            for idx, val in random.sample(list(enumerate(ls_query_text)), n_eval):
                text_label_map[val] = idx
                sampled_queries.append(val)

            for query_idx, query_text in enumerate(sampled_queries):
                example_texts = json.dumps(dict(enumerate(ls_potential_text)))
                prompt = prompt_input + f"""The input texts are delimited with triple backticks. ```\n\nQuery text: {query_text} \n\nTexts from potential authors: {example_texts}\n\n```"""

                raw_response = client.chat.completions.create(
                    model=model_name,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0,
                    max_tokens=1024
                )

                response_str = raw_response.choices[0].message.content

                try:
                    response = json.loads(response_str, strict=False)
                except json.JSONDecodeError:
                    response = {"analysis": f"JSON解析失败: {response_str}", "answer": -1}

                response["query_text"] = query_text
                response["example_texts"] = example_texts
                response["tokens"] = raw_response.usage.total_tokens
                response["label"] = text_label_map[query_text]
                ls_reps.append(response)

            df_reps = pd.DataFrame(ls_reps)
            df_reps['answer'] = pd.to_numeric(df_reps['answer'], errors='coerce')
            df_reps['answer'] = df_reps['answer'].fillna(-1)
            df_res_all = pd.concat([df_res_all, df_reps]).reset_index(drop=True)

        return df_res_all