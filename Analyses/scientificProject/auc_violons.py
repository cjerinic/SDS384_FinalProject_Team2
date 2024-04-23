# %% plot aucs as violin plot for each operation (points for individuals subjects)
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#  merge each subject auc data
#path = "/Users/cnj678/Desktop/repclear_caleb/caleb_replicate/results/operation decoding/within-sub/data/*.csv"
#files = glob.glob(path)
#auc_merge_within = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)
auc_merge_within = pd.read_csv('/Users/cnj678/Documents/GitHub/SDS384_FinalProject_Team2/Analyses/scientificProject/Model AUCs/aucs_df_XGBoost.csv')

auc_data = auc_merge_within

auc_data_select = auc_data.iloc[0:5, [3, 0, 1, 2]]
auc_data_select.columns = ['Subject', 'Maintain', 'Replace', 'Suppress']

AUC_data_long = pd.melt(auc_data_select, id_vars=['Subject'], value_vars=['Maintain', 'Replace', 'Suppress'], var_name='operation')

AUC_data_long.rename(columns={'value': 'auc'}, inplace=True)

# %%
# plot
sns.set(style="whitegrid")
color_dict = {"Maintain": "green", "Replace": "blue", "Suppress": "red"}

plt.ylim(0.4, 1)
sns.violinplot(
    x="operation",
    y="auc",
    data=AUC_data_long,
    inner="quartile",
    palette=color_dict.values(),
)
sns.swarmplot(x="operation", y="auc", data=AUC_data_long, color="white")
plt.axhline(y=0.5, color="black", linestyle="--")

plt.xlabel("Operation")
plt.ylabel("AUC")
plt.title("Between-Subject Operation Decoding AUCs")

plt.savefig(
    "/Users/cnj678/Documents/GitHub/SDS384_FinalProject_Team2/btwnSub_op_auc_violin_XGBoost.png"
)
plt.show()