# %% import packages
import pandas as pd
import numpy as np
import glob
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from pingouin import pairwise_ttests


#
# %% merge aggregated data
path = '/Users/cnj678/Documents/GitHub/SDS384_FinalProject_Team2/Analyses/scientificProject/Model AUCs/*.csv'
files = glob.glob(path)
auc_merge = pd.concat([pd.read_csv(file) for file in files],
                            ignore_index=True)

# %%
agg_select = [
    'Maintain',
    'Replace',
    'Suppress'
]

agg_dict = {col: 'sum' for col in agg_select}
rename_dict = {col: col.lower() + '_sum' for col in agg_select}

auc_agg = auc_merge.groupby(['model']).agg(agg_dict).rename(columns=rename_dict).reset_index()

# %%
auc_agg = auc_agg.assign(maintain_avg = auc_agg['maintain_sum'] / 5)
auc_agg = auc_agg.assign(replace_avg = auc_agg['replace_sum'] / 5)
auc_agg = auc_agg.assign(suppress_avg = auc_agg['suppress_sum'] / 5)

# %%
auc_anova = auc_agg

auc_anova_select = auc_anova.iloc[0:4, [0, 4, 5, 6]]
auc_anova_select.columns = ['Model', 'Maintain', 'Replace', 'Suppress']
auc_anova_long = pd.melt(auc_anova_select, id_vars=['Model'], value_vars=['Maintain', 'Replace', 'Suppress'], var_name='Operation')

AUC_anova_long = pd.melt(auc_anova_select, id_vars=['Model'], value_vars=['Maintain', 'Replace', 'Suppress'], var_name='Operation')

AUC_anova_long.rename(columns={'value': 'AUC'}, inplace=True)

# %%
anova_model = ols('AUC ~ C(Model) + C(Operation) + C(Model)*C(Operation)', data=AUC_anova_long).fit()

anova_model_table = sm.stats.anova_lm(anova_model, typ=2)
print(anova_model_table)

# %%
result = pairwise_ttests(dv='AUC', between='Model', data=AUC_anova_long, padjust='bonferroni')

print(result)