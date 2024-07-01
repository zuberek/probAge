# %%
import sys
sys.path.append("..")   # fix to import modules from root
from src.general_imports import *

import src.modelling_bio_beta as modelling

amdata = ad.read_h5ad('../exports/wave3_person_fitted.h5ad', backed='r')
# part_indexes = modelling.sample_to_uniform_age(amdata, n_part=2000)
# amdata[:, part_indexes].var.age.hist()
# amdata = amdata[:, part_indexes]


# Cut off age to make the bins have more participants
amdata = amdata[:, amdata.var.age<80].to_memory()

N_BINS = 5
amdata.var['age_bin'] = pd.cut(amdata.var.age, N_BINS)
# amdata.var['age_bin'] = pd.qcut(amdata.var.age, N_BINS)

# round age bins to int
amdata.var['age_bin'] = amdata.var['age_bin'].apply(
    lambda x: f'({round(x.left)}, {round(x.right)}]')
age_bins = np.sort(amdata.var.age_bin.unique())

max_part = amdata.var.age_bin.value_counts().min()

# Run only for the sites in Horvath
# site_meta = pd.read_csv('../resources/AllGS_CpGs.csv', index_col=0)
# site_hor = site_meta[site_meta.Clock=='Horvath'].index
# intersection = amdata.obs.index.intersection(site_hor)
# amdata = amdata[intersection].to_memory()


part_index = []
for bin in amdata.var.age_bin.unique():
    age_bin = amdata[:, amdata.var.age_bin == bin].copy()
    part_index.extend(list(np.random.choice(age_bin.var.index, max_part, replace=False)))

amdata = amdata[:, part_index].copy()

participants = amdata.var
participants['Horvath'] = participants.Horvath/participants.Horvath.std()

# %%
# Boxplot showing our Accelerationeleration variance change next to Horvath
plot.fonts(8)
ax = plot.row(figsize=(9.5, 6))
sns.despine()

# shift to be around zero
participants.Horvath = participants.Horvath - participants.groupby('age_bin')['Horvath'].transform('mean')

sns.boxplot(data=participants, y='Horvath', x='age_bin', showfliers=False, color=colors[1])
# df = participants.reset_index().melt(id_vars='age_bin', value_vars=['Horvath'])
# sns.boxplot(data=df, x='age_bin', y='value', color=colors[1], hue='variable', showfliers=False)
sns.despine()

ax.set_xlabel('Age bin (years)')
ax.set_ylabel('Horvath acceleration')
ax.get_figure().tight_layout()
# ax.tick_params(axis='x', labelrotation=30)
ax.get_figure().tight_layout()
# ax2.set_ylabel('Methylation level \n(beta values)')

# %%
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext1/Ext1F_boxplot_accvarVSage_Horvath.png', transparent=True)
ax.get_figure().savefig(f'{paths.FIGURES_DIR}/ext1/Ext1F_boxplot_accvarVSage_Horvath.svg', transparent=True)
# %%
