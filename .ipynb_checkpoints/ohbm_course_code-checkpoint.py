# -*- coding: utf-8 -*-
"""
import pyperm as pr
"""
import numpy as np
import pandas as pd
import pyperm as pr
from nilearn.image import get_data, load_img
from nilearn.input_data import NiftiMasker
ohbm_course_folder = 'C:\\Users\\12SDa\\davenpor\\davenpor\\Toolboxes\\Courses\\2023_obhm_course\\'
data_loc = ohbm_course_folder + 'data\\example_real_data\\HCP_U77_WM\\'
bold_files = pr.list_files(data_loc, '.nii.gz', 1)
fwhm = 3
mask = 'C:\\Users\\12SDa\\global\\Intern\\drago\\Images\\mask_GM_forFunc.nii'
masker = NiftiMasker(smoothing_fwhm=fwhm, mask_img=mask).fit()
data = masker.transform(bold_files).transpose()
# data = np.sqrt(data)
nsubj = 77
# %%
tstat, _, _ = pr.mvtstat(data)
tstat_3d = np.squeeze(
    get_data(masker.inverse_transform(tstat.transpose())))

plt.imshow(tstat_3d[:, :, 30])
pvalues = pr.tstat2pval(tstat, nsubj-1)

_, nrej, _ = pr.fdr_bh(pvalues, 0.05)

print(nrej)

# %%
plt.hist(pvalues)

# %% Load the covariates
pd_data = pd.read_csv(
    data_loc + 'behavioural_data_subset_77.csv')
print(pd_data)
design = np.zeros((77, 5))
design[:, 1:5] = pd_data.values[:, 1:5]

# Add an intercept:
design[:, 0] = 1

# %% Load the covariates
pd_data = pd.read_csv(
    data_loc + 'behavioural_data_subset_77.csv')
print(pd_data)
design = np.zeros((77, 4))
design[:, 0:4] = pd_data.values[:, 1:5]

# %%
design = np.ones((77, 4))
design[:, 1:4] = pd_data.values[:, 1:4]

# %%
img = load_img(bold_files[0])

# Convert the data to a field
data_field = pr.make_field(data.copy())

# Obtain the number of subjects
nsubj = data_field.fibersize

# contrast_matrix = np.array((1, 0, 0, 0, 0))
# contrast_matrix = np.array((1))
# contrast_matrix = np.array((1, 0))
contrast_matrix = np.array((0, 0, 0, 1))
n_params = 4

# Obtain the test statistics and convert to p-values
test_stats, _ = pr.contrast_tstats(data_field, design, contrast_matrix)

# pvalues = 2*(1 - t.cdf(abs(test_stats.field), nsubj-n_params))
pvalues = pr.tstat2pval(test_stats.field, nsubj-n_params, 1)

pvalues_3d = np.squeeze(
    get_data(masker.inverse_transform(pvalues.transpose())))

# test_stats_3d =

# %%
plt.imshow(pvalues_3d[:, :, 30])

# %%
plt.hist(pvalues)

# %%
_, nrej, _ = pr.fdr_bh(pvalues, 0.1)
print(nrej)

# %%
n_bootstraps = 100
minPperm, origpvals, pivotal_stats, bootstore = pr.boot_contrasts(
    data.copy(), design, contrast_matrix, n_bootstraps=n_bootstraps,
    display_progress=1, store_boots=1, doonesample=1)

# %%
plt.hist(origpvals.field)

# %%
# Obtain the lambda calibration
alpha = 0.1
lambda_quant = np.quantile(pivotal_stats, alpha)
nhypotheses = np.prod(origpvals.field.shape)

# %%
sorted_pvalues = np.sort(origpvals.field, 0)
pvalue_subset = sorted_pvalues[0:1000]

# %%
FP_bound, TP_bound, FDP_bound, TDP_bound = pr.get_bounds(
    pvalue_subset, lambda_quant, nhypotheses)
print(TP_bound)

# %%
FP_bound, TP_bound, FDP_bound, TDP_bound = pr.get_bounds(
    pvalue_subset, lambda_quant, nhypotheses)
