import pandas as pd
from attribution_core import AuthorAttribution
from data_loader import DataLoader
from experiment_runner import ExperimentRunner
from prompts import Prompts

data_loader = DataLoader()
df_blog = data_loader.load_blog_data("data/blogtext.csv")
df_email = data_loader.load_email_data("data/emails.csv")
attribution_model = AuthorAttribution()

df_10_test = attribution_model.sampler_aa_fn_pro(df_blog, 10, 3)
#df_10_test=pd.read_csv("data/blog_n10_reps3.csv")

from zai import ZhipuAiClient
client = ZhipuAiClient(api_key="API_KEY")

ls_df, ls_model, ls_method = [], [], []
prompts = Prompts()
df1 = attribution_model.run_aa_llm(
    df_10_test, 'no_guidance', 'glm-4.5-air',
    prompts.get_no_guidance_prompt(),
    prompts.get_system_message(),
    client
)
ls_df.append(df1)
ls_model.append('glm-4.5-air')
ls_method.append('no_guidance')

experiment_runner = ExperimentRunner(attribution_model)
results = experiment_runner.compare_baseline_models(
    df_10_test, ls_df, ls_model, ls_method,10
)

print(results)