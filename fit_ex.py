
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt


def three_way_fit(clusters, energies, counts, comps, fold_pick=10, Normalize=True, Intercept=True, Energy_above_hull = True, name=''):
    ###- Select fitting paramiters -###
    scale = lambda x0, y0, x1, y1, x2, y2: abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt(
        (y2 - y1) ** 2 + (x2 - x1) ** 2)
    ###- scale to energy above hull -###
    if Energy_above_hull == True:
        y1 = min(energies)
        y2 = max(energies)
        x2 = min(comps)
        x1 = max(comps)
        if x2==x1:
            energies = [energies[i]-y1 for i in range(len(energies))]
        else:
             energies = [scale(comps[i], energies[i], x1, y1, x2, y2) for i in range(len(energies))]

    ###- Set up output file -###
    file_out = 'Fit_summery.txt'
    file = open(file_out, 'w')
    file.write('Data set: ' + name + '\n\n' + 'Clusters:  [[species],[distance],[chem (0) / spin (1)]]' + '\n')
    [file.write(str(clusters[i]) + '\n') for i in range(len(clusters))]
    file.write('\n\nEnergy per Atom (eV) : Cluster Count per Atom\n')
    for i in range(len(energies)):
        file.write(str(energies[i]) + ' : ' + str(counts[i]) + '\n')

    ###- Set up alphas for CV -###
    alpha_range = [-10, 10]
    alpha_lasso = np.logspace(alpha_range[0], alpha_range[1], num=1000)
    n_alphas = 1010
    alpha_ridge = np.logspace(-15, 10, n_alphas)

    ###- Set range for plot -###
    axis_range = [min(energies) * 1.0001, max(energies) * .9999]

    # LASSO and RIDGE, Cross-Validation, Lin Reg without CV
    lassocv = LassoCV(alphas=alpha_lasso, normalize=Normalize, fit_intercept=Intercept, cv=fold_pick, max_iter=1e5)
    ridgecv = RidgeCV(alphas=alpha_ridge, normalize=Normalize, fit_intercept=Intercept, cv=None, store_cv_values=True)
    linreg = LinearRegression(fit_intercept=Intercept, normalize=Normalize)
    # Fit to data for each method
    lassocv.fit(counts, energies)
    ridgecv.fit(counts, energies)
    linreg.fit(counts, energies)
    lassocv_rmse = np.sqrt(lassocv.mse_path_)
    ridgecv_rmse = np.sqrt(ridgecv.cv_values_)

    RandF_reg = RandomForestRegressor(max_depth=5, random_state=0)
    RandF_reg.fit(counts, energies)

    ### Get stuff ready for EAH plots ###
    y1 = min(energies)
    y2 = max(energies)
    x2 = min(comps)
    x1 = max(comps)
    scale = lambda x0, y0, x1, y1, x2, y2: abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt(
        (y2 - y1) ** 2 + (x2 - x1) ** 2)
    eahDFT = [scale(comps[i], energies[i], x1, y1, x2, y2) for i in range(len(energies))]
    axis_rangeEAH = [-0.002, max(eahDFT) * 1.1]


    ########################################################################################################################
    ################ RANDOM FOREST FIT #############################################################################################
    ########################################################################################################################
    file.write("\n\n#### Random Forest #### ")

    # Show data from CV
    plt.figure()
    # Plot Fit vs DFT

    cluster_energy_RF = RandF_reg.predict(counts)
    print(RandF_reg.estimators_)
    print(RandF_reg.get_params())
    plt.figure()
    plt.scatter(energies, cluster_energy_RF, color='b', alpha=0.5)
    plt.plot(axis_range, axis_range, 'k', alpha=0.5)
    plt.xlim(axis_range)
    plt.ylim(axis_range)
    plt.gca().set_aspect('equal')
    plt.xlabel('Energy, DFT')
    plt.ylabel('Energy, CE')
    plt.title('Random Forest Fit Comparison: ' + name)
    plt.tight_layout()
    plt.show()
    eahCE = [scale(comps[i], cluster_energy_RF[i], x1, y1, x2, y2) for i in range(len(cluster_energy_RF))]
    plt.scatter(eahDFT, eahCE, color='b', alpha=0.5)
    plt.plot(axis_rangeEAH, axis_rangeEAH, 'k', alpha=0.5)
    plt.xlim(axis_rangeEAH)
    plt.ylim(axis_rangeEAH)
    plt.gca().set_aspect('equal')
    plt.xlabel('EAH, DFT')
    plt.ylabel('EAH, CE')
    plt.title('Random Forest Fit Comparison: ' + name)
    plt.tight_layout()
    plt.show()


    ########################################################################################################################
    ################ LASSO FIT #############################################################################################
    ########################################################################################################################

    file.write("\n\n#### LASSO #### \nk-folds cross validation\n")
    file.write("alpha: %7.6f\n" % lassocv.alpha_)
    file.write("avg rmse: %7.4f\n" % min(lassocv_rmse.mean(axis=-1)))
    file.write("score: %7.4f\n" % lassocv.score(counts, energies))
    file.write("non-zero coefficient: %7.4f\n" % np.count_nonzero(lassocv.coef_))
    file.write('Intercept: ')
    file.write(str(lassocv.intercept_))
    file.write('\n')
    # Show data from CV
    plt.figure()
    m_log_alphas = -np.log10(lassocv.alphas_)
    plt.plot(m_log_alphas, lassocv_rmse, ':')
    plt.plot(m_log_alphas, lassocv_rmse.mean(axis=-1), 'k', label='Average across the folds', linewidth=2)
    plt.axvline(-np.log10(lassocv.alpha_), linestyle='--', color='k', label='alpha: CV estimate')
    plt.xlabel('-log(alpha)')
    plt.ylabel('Root-mean-square error')
    plt.title('Root-mean-square error on each fold: ' + name)
    plt.legend()
    plt.tight_layout()
    plt.show()
    # Plot Fit vs DFT
    cluster_energy_ce = lassocv.predict(counts)
    plt.figure()
    plt.scatter(energies, cluster_energy_ce, color='b', alpha=0.5)
    plt.plot(axis_range, axis_range, 'k', alpha=0.5)
    plt.xlim(axis_range)
    plt.ylim(axis_range)
    plt.gca().set_aspect('equal')
    plt.xlabel('Energy, DFT')
    plt.ylabel('Energy, CE')
    plt.title('LASSO Fit Comparison: ' + name)
    plt.tight_layout()
    plt.show()
    eahCE = [scale(comps[i], cluster_energy_ce[i], x1, y1, x2, y2) for i in range(len(cluster_energy_ce))]
    plt.scatter(eahDFT, eahCE, color='b', alpha=0.5)
    plt.plot(axis_rangeEAH, axis_rangeEAH, 'k', alpha=0.5)
    plt.xlim(axis_rangeEAH)
    plt.ylim(axis_rangeEAH)
    plt.gca().set_aspect('equal')
    plt.xlabel('EAH, DFT')
    plt.ylabel('EAH, CE')
    plt.title('LASSO Fit Comparison: ' + name)
    plt.tight_layout()
    plt.show()

    # Show Non-zero clusters
    cluster_energy_new = []
    for i in range(len(energies)):
        cluster_energy_new.append(energies[i] - cluster_energy_ce[i])
    cluster_coef = []
    cluster_pick = []
    cluster_coef.append(lassocv.intercept_)
    cluster_coef_all = lassocv.coef_
    cluster_nonzero = [c for c, v in enumerate(cluster_coef_all) if abs(v) >= 0.00000000001 ]
    for i in cluster_nonzero:
        cluster_coef.append(cluster_coef_all[i])
        cluster_pick.append(clusters[i])
    file.write("\n Clusters \n")
    for i in range(len(cluster_pick)):
        if len(cluster_pick[i]) == 2:
            file.write(str(cluster_pick[i][0]) + ':' + '[0]' + ':' + str(cluster_pick[i][1][0]) + ':' + str(
                cluster_coef[i + 1]) + '\n')
        else:
            file.write(
                str(cluster_pick[i][0]) + ':' + str(cluster_pick[i][1]) + ':' + str(cluster_pick[i][2][0]) + ':' + str(
                    cluster_coef[i + 1]) + '\n')
    file.write("\n")
    file.write("\n")

    ########################################################################################################################
    ############# RIDGE FIT ################################################################################################
    ########################################################################################################################

    file.write("### RIDGE ### \nk-folds cross validation\n")
    file.write("alpha: %7.6f\n" % ridgecv.alpha_)
    file.write("avg rmse: %7.4f\n" % min(ridgecv_rmse.mean(axis=-1)))
    file.write("score: %7.4f\n" % ridgecv.score(counts, energies))
    file.write("non-zero coefficient: %7.4f\n" % np.count_nonzero(ridgecv.coef_))
    # Plot Fit vs DFT
    cluster_energy_ce = ridgecv.predict(counts)
    plt.figure()
    plt.scatter(energies, cluster_energy_ce, color="r", alpha=0.5)
    plt.plot(axis_range, axis_range, 'k', alpha=0.5)
    plt.xlim(axis_range)
    plt.ylim(axis_range)
    plt.gca().set_aspect('equal')
    plt.xlabel('Energy, DFT')
    plt.ylabel('Energy, CE')
    plt.title('RIDGE Fit Comparison: ' + name)
    plt.tight_layout()
    plt.show()
    eahCE = [scale(comps[i], cluster_energy_ce[i], x1, y1, x2, y2) for i in range(len(cluster_energy_ce))]
    plt.scatter(eahDFT, eahCE, color='r', alpha=0.5)
    plt.plot(axis_rangeEAH, axis_rangeEAH, 'k', alpha=0.5)
    plt.xlim(axis_rangeEAH)
    plt.ylim(axis_rangeEAH)
    plt.gca().set_aspect('equal')
    plt.xlabel('EAH, DFT')
    plt.ylabel('EAH, CE')
    plt.title('RIDGE Fit Comparison: ' + name)
    plt.tight_layout()
    plt.show()
    # Show Non-zero clusters
    cluster_coef = []
    cluster_pick = []
    cluster_coef.append(ridgecv.intercept_)
    cluster_coef_all = ridgecv.coef_
    cluster_nonzero = [c for c, v in enumerate(cluster_coef_all) if abs(v) >= 0.00000000001]
    for i in cluster_nonzero:
        cluster_coef.append(cluster_coef_all[i])
        cluster_pick.append(clusters[i])
    file.write("\n Clusters\n")
    for i in range(len(cluster_pick)):
        if len(cluster_pick[i]) == 2:
            file.write(str(cluster_pick[i][0]) + ':' + '[0]' + ':' + str(cluster_pick[i][1][0]) + ':' + str(
                cluster_coef[i + 1]) + '\n')
        else:
            file.write(
                str(cluster_pick[i][0]) + ':' + str(cluster_pick[i][1]) + ':' + str(cluster_pick[i][2][0]) + ':' + str(
                    cluster_coef[i + 1]) + '\n')

    ########################################################################################################################
    ############# LIN REG FIT ##############################################################################################
    ########################################################################################################################

    file.write("\n #### Lin Reg #### \nNo cross validation\n")
    file.write("score: %7.4f\n" % ridgecv.score(counts, energies))
    file.write("non-zero coefficient: %7.4f\n" % np.count_nonzero(ridgecv.coef_))
    file.write('Intercept: %7.4f\n' % linreg.intercept_)
    # Plot Fit vs DFT
    cluster_energy_ce = linreg.predict(counts)
    plt.figure()
    plt.scatter(energies, cluster_energy_ce, color="g", alpha=0.5)
    plt.plot(axis_range, axis_range, 'k', alpha=0.5)
    plt.xlim(axis_range)
    plt.ylim(axis_range)
    plt.gca().set_aspect('equal')
    plt.xlabel('Energy, DFT')
    plt.ylabel('Energy, CE')
    plt.title('LinReg Fit Comparison: ' + name)
    plt.tight_layout()
    plt.show()
    eahCE = [scale(comps[i], cluster_energy_ce[i], x1, y1, x2, y2) for i in range(len(cluster_energy_ce))]
    plt.scatter(eahDFT, eahCE, color='g', alpha=0.5)
    plt.plot(axis_rangeEAH, axis_rangeEAH, 'k', alpha=0.5)
    plt.xlim(axis_rangeEAH)
    plt.ylim(axis_rangeEAH)
    plt.gca().set_aspect('equal')
    plt.xlabel('EAH, DFT')
    plt.ylabel('EAH, CE')
    plt.title('LinReg Fit Comparison: ' + name)
    plt.tight_layout()
    plt.show()
    # Show Non-zero clusters
    cluster_coef = []
    cluster_pick = []
    cluster_coef.append(linreg.intercept_)
    cluster_coef_all = linreg.coef_
    cluster_nonzero = [c for c, v in enumerate(cluster_coef_all) if abs(v) >= 0.00000000001]
    for i in cluster_nonzero:
        cluster_coef.append(cluster_coef_all[i])
        cluster_pick.append(clusters[i])
    file.write('\nClusters\n')
    for i in range(len(cluster_pick)):
        if len(cluster_pick[i]) == 2:
            file.write(str(cluster_pick[i][0]) + ':' + '[0]' + ':' + str(cluster_pick[i][1][0]) + ':' + str(
                cluster_coef[i + 1]) + '\n')
        else:
            file.write(
                str(cluster_pick[i][0]) + ':' + str(cluster_pick[i][1]) + ':' + str(cluster_pick[i][2][0]) + ':' + str(
                    cluster_coef[i + 1]) + '\n')
    file.write('\n')
    file.close()
    return


def pert_test(clusters, energies, counts, comps, noise=0.1, Normalize=True, Intercept=True, Energy_above_hull = True, name=''):
    fold_pick = 10
    lasso_coefs = []
    ridge_coefs = []
    linreg_coefs = []
    counts = np.array(counts)
    energies = np.array(energies)
    ###- scale to energy above hull -###
    if Energy_above_hull == True:
        y1 = min(energies)
        y2 = max(energies)
        x2 = min(comps)
        x1 = max(comps)
        scale = lambda x0, y0, x1, y1, x2, y2: abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1) / np.sqrt(
            (y2 - y1) ** 2 + (x2 - x1) ** 2)
        energies = [scale(comps[i], energies[i], x1, y1, x2, y2) for i in range(len(energies))]

    ###- Set up output file -###
    file_out = 'pert_summery.txt'
    file = open(file_out, 'w')

    ###- Set up alphas for CV -###
    alpha_range = [-10, 10]
    alpha_lasso = np.logspace(alpha_range[0], alpha_range[1], num=1000)
    n_alphas = 1010
    alpha_ridge = np.logspace(-15, 10, n_alphas)

    # LASSO and RIDGE, Cross-Validation, Lin Reg without CV
    lassocv = LassoCV(alphas=alpha_lasso, normalize=Normalize, fit_intercept=Intercept, cv=fold_pick, max_iter=1e5)
    ridgecv = RidgeCV(alphas=alpha_ridge, normalize=Normalize, fit_intercept=Intercept, cv=None, store_cv_values=True)
    linreg = LinearRegression(fit_intercept=Intercept, normalize=Normalize)
    # Fit to data for each method
    noise = np.linspace(0.001,1,25)
    lasso_vars = [[] for _ in range(len(clusters))]
    for n in noise:
        lasso_coefs = []
        ridge_coefs = []
        linreg_coefs = []
        lassocv.fit(counts, energies)
        lasso_coefs.append(lassocv.coef_)
        ridgecv.fit(counts, energies)
        ridge_coefs.append(ridgecv.coef_)
        linreg.fit(counts, energies)
        linreg_coefs.append(linreg.coef_)
        for i in range(100):
            data_noise = np.random.normal(0, n, counts.shape)
            counts_new = counts + data_noise
            data_noise = np.random.normal(0, n, energies.shape)
            energies_new = energies + data_noise
            lassocv.fit(counts_new, energies_new)
            lasso_coefs.append(lassocv.coef_)
            ridgecv.fit(counts_new, energies_new)
            ridge_coefs.append(ridgecv.coef_)
            linreg.fit(counts_new, energies_new)
            linreg_coefs.append(linreg.coef_)
        lasso_coefs = np.array(lasso_coefs)
        ridge_coefs = np.array(ridge_coefs)
        linreg_coefs = np.array(linreg_coefs)
        for i in range(len(clusters)):
            data = np.transpose(lasso_coefs[:, i])
            var = data.var()
            lasso_vars[i].append(var)
    for i in range(len(lasso_vars)):
        plt.plot(noise,lasso_vars[i])
    file.close()
    return