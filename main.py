import count2_with_spins as count
import fit_ex as fit
import data_check as dc

# DFT data file
vasp_data = 'NiMnIn_aust_all'
# Cluster rule file. Cluster rules are defined as follows:
# [[atomic species included in cluster], [distance between atoms], [chemical cluster (0), spin cluster (1)]]
cluster_file = 'clusters_A_all'
clusters = [] # List of each cluster rule
enrgies = [] # Energy for each DFT data point
counts = [] # Predictor matrix (number of times each cluster appears in each DFT data point)
comps = [] # List of compositions for each data point
normalize = True # Normalize fit
intercept = True # Fit with intercept
energy_above = False # Fit to energy above hull
fold = 40 # fold pick for cross validation in LASSOCV and RIDGECV

count.count_clusters(vasp_data, cluster_file, clusters, enrgies, counts, comps) # Fill predictor matrix, energies, and compositions (this takes the longest to run)
dc.corr_check(clusters,enrgies,counts,comps) # Generate matrix of correlation coefficients for testing and validation THIS CAN BE IGNORED IN THE CODE REVIEW
# Fit using Random Forest Regression, LASSO, RIDGE, and Least Squares (should be called four way fit but I added Random forrest)
fit.three_way_fit(clusters, enrgies, counts, comps,fold_pick=fold, Normalize=normalize, Intercept=intercept, Energy_above_hull=energy_above)
# The results of each fit (Except for random forest) are found in Fit_summery.txt