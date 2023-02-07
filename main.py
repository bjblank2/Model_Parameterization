import count2_with_spins as count
import fit_ex as fit
import data_check as dc
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# DFT data file
vasp_data = 'NiMnIn_Mart_12'
#vasp_data = 'Ni1_with_n'
#vasp_data = 'NiMnIn_aust_all2'
#vasp_data = 'NiMn_mart'
#vasp_data = 'Int_mart'
#vasp_data = 'Ni2MnIn_mart'
#vasp_data = 'NiMn_aust'
#vasp_data = 'Int_aust'
#vasp_data = 'Ni2MnIn_aust'


# Cluster rule file. Cluster rules are defined as follows:
# [[atomic species included in cluster], [distance between atoms], [chemical cluster (0), spin cluster (1)]]
cluster_file = 'clusters_M_test'
#cluster_file = 'clusters_A_all'

save_file = 'data_save'
clusters = [] # List of each cluster rule
enrgies = [] # Energy for each DFT data point
counts = [] # Predictor matrix (number of times each cluster appears in each DFT data point)
comps = [] # List of compositions for each data point
vols = [] # List of volume for each data point
normalize = True # Normalize fit
intercept = True # Fit with intercept
energy_above = False # Fit to energy above hull
fold = 89# fold pick for cross validation in LASSOCV and RIDGECV ( 91 for mart all and 40 for aust all )
use_saved = True
if use_saved == True:
    clusters = []
    enrgies = []
    counts = []
    comps = []
    count.read_data(save_file, cluster_file, clusters, enrgies, counts, comps)
    first_fit = fit.three_way_fit(clusters, enrgies, counts, comps,fold_pick=fold, Normalize=normalize, Intercept=intercept, Energy_above_hull=energy_above)
    #forced_clusters = [[19, 0.008]]#,[20, 0.02],[21, -0.0015], [22, -0.0015]]
    #first_fit = fit.forced_param_fit(forced_clusters, clusters, enrgies, counts, comps, fold_pick=fold, Normalize=normalize, Intercept=intercept, Energy_above_hull=energy_above)
    #first_fit = fit.forced_param_ridge_fit(forced_clusters, clusters, enrgies, counts, comps, fold_pick=fold, Normalize=normalize, Intercept=intercept, Energy_above_hull=energy_above)

else:
    count.count_clusters(vasp_data, cluster_file, clusters, enrgies, counts, comps, vols) # Fill predictor matrix, energies, and compositions (this takes the longest to run)
    # count.count_clusters(vasp_data, cluster_file, clusters, enrgies, counts, comps, vols) # Fill predictor matrix, energies, and compositions (this takes the longest to run)
    # dc.corr_check(clusters,enrgies,counts,comps) # Generate matrix of correlation coefficients for testing and validation THIS CAN BE IGNORED IN THE CODE REVIEW
    # Fit using Random Forest Regression, LASSO, RIDGE, and Least Squares (should be called four way fit but I added Random forrest)
    for i in range(len(enrgies)):
        print(str(enrgies[i])+ ", "+str(vols[i]))
    plt.scatter(enrgies,vols)
    plt.xlabel("DFT Energy (eV/atom)")
    plt.ylabel("Volume (A^3/atom)")
    plt.title("Volume vs Energy NiMn Martensite")
    plt.show()
    count.save_data("data_save", clusters, enrgies, counts, comps)
    first_fit = fit.three_way_fit(clusters, enrgies, counts, comps, fold_pick=fold, Normalize=normalize, Intercept=intercept, Energy_above_hull=energy_above)
