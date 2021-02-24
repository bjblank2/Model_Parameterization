import count2_with_spins as count
import fit_ex as fit
import data_check as dc

vasp_data = 'NiMnIn_mart_all'
cluster_file = 'clusters_M_all2'
clusters = []
enrgies = []
counts = []
comps = []
normalize = True
intercept = True
energy_above = False
count.count_clusters(vasp_data, cluster_file, clusters, enrgies, counts, comps)
dc.corr_check(clusters,enrgies,counts,comps)
print(dc.clac_VIF(counts,clusters))
#fit.pert_test(clusters, enrgies, counts, comps,  noise=.01, Normalize=normalize, Intercept=intercept, Energy_above_hull=energy_above)
fit.three_way_fit(clusters, enrgies, counts, comps,fold_pick=70, Normalize=normalize, Intercept=intercept, Energy_above_hull=energy_above)
