
# %% plot aucs as violin plot for each operation (points for individuals subjects)
#  merge each subject auc data
path = "/Users/cnj678/Desktop/repclear_caleb/caleb_replicate/results/operation decoding/within-sub/data/*.csv"
files = glob.glob(path)
auc_merge_within = pd.concat([pd.read_csv(file) for file in files], ignore_index=True)

# plot
sns.set(style="whitegrid")
color_dict = {"Maintain": "green", "Replace": "blue", "Suppress": "red"}

plt.ylim(0.4, 1)
sns.violinplot(
    x="operation",
    y="auc",
    data=auc_merge_within,
    inner="quartile",
    palette=color_dict.values(),
)
sns.swarmplot(x="operation", y="auc", data=auc_merge_within, color="white")
plt.axhline(y=0.5, color="black", linestyle="--")

plt.xlabel("Operation")
plt.ylabel("AUC")
plt.title("Within-Subject Operation Decoding AUCs")

plt.savefig(
    "/Users/cnj678/Desktop/repclear_caleb/caleb_replicate/results/operation decoding/within-sub/images/withinSub_op_auc_violin.png"
)
plt.show()