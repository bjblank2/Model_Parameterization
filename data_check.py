from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor


def corr_check(clusters, enrgies, counts, comps):
    ###                        ###
    # Look at Correlation Matrix #
    ###                        ###
    cluster_count = np.array(counts)
    col = [str(i) for i in range(len(clusters))]
    df = pd.DataFrame(cluster_count, columns=col)
    f = plt.figure()
    corr = df.corr()
    plt.matshow(corr, fignum=f.number, cmap='RdBu_r')
    plt.xticks(range(df.shape[1]), df.columns, rotation=45)
    plt.yticks(range(df.shape[1]), df.columns)
    plt.clim(vmin=-1, vmax=1)
    cb = plt.colorbar()
    cb.ax.tick_params()
    for (x, y), value in np.ndenumerate(corr):
        plt.text(x, y, f"{value:.2f}", va="center", ha="center")
    plt.title('Correlation Matrix');
    plt.show()
    return


def r2(enrgies_true, enrgies_fit):
    score = r2_score(enrgies_true, enrgies_fit)
    return score

def clac_VIF(counts, clusters):
    counts = np.array(counts)
    column_label = [str(i) for i in range(len(clusters))]
    count_df = pd.DataFrame(data=counts, columns=column_label)
    vif = pd.DataFrame()
    vif["VIF Factor"] = [variance_inflation_factor(count_df.values, i) for i in range(count_df.shape[1])]
    vif["features"] = count_df.columns
    return vif
