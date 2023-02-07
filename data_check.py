from sklearn.metrics import r2_score
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.outliers_influence import variance_inflation_factor
import json
import vasp as vp
import celib as celib # Functions used to make the predictor matrix (list of how many times each cluster appears in each DFT data point)
import numpy as np

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

def dist_check(data_file, cluster_file):
    # Read in Cluster rules
    with open(cluster_file) as f:
        data_old = json.load(f)
    cluster = data_old['List'] # list of clusters
    # Read in DFT data
    data_f = open(data_file)
    data = data_f.readlines() # raw data file (asci)
    data_f.close()
    pos = {} # Dictionary containing all data and metadata for each DFT calculation
    enrgs = [] # list of all DFT energies
    pos_old_list = [] # Copy of "pos" dictionary used for supercell creation
    comp_list = [] # list of compositions
    count = -1 # index used for parsing data file
    sets = 0 # index used for parsing data file
    # Begin parsing DFT data file
    for i in range(len(data)):
        if "#" in data[i]: # "#" indicates new DFT data point
            count += 1
            sets += 1
            species = (data[i].split()) # List of the chemical species in the DFT data (strings Ni, Mn, In)
            species.pop(0)
            set_data = data[i + 1]
            set_data = set_data.split()
            name = set_data[3]
            element_number = 3
            atom_numb = [int(set_data[0].strip('[').strip(',')), int(set_data[1].strip(',')), int(set_data[2].strip(']'))] # Number of atoms of each type
            atom_sum = sum(atom_numb) # Total number of atoms in each DFT data point
            lat_true = [float(set_data[6].strip('[').strip(',')), float(set_data[7].strip(',')), float(set_data[8].strip(']'))]
            # In order to make finding distances between neighboring atoms easier all, all martensite data is given a c/a ratio of 1.5
            factor = np.power(atom_sum/2, 1/3)/2
            lat_const = factor
            if set_data[5] == 'mart':
                base = [[factor, 0, 0], [0, factor, 0], [0, 0, factor * 1.5]]
            elif set_data[5] == 'aust':
                base = [[factor, 0, 0], [0, factor, 0], [0, 0, factor]]
            element_name = ['Ni', 'Mn', 'In']
            issel = 0
            latType = 'Direct'
            # Scale energy to per atom
            enrgs.append(float(set_data[4]) / atom_sum)
            pos_list = [] # list of positions for each atom
            spin_list = [] # list of spin at each atom
            type_list = [] # list of species of each atom
            comp_list.append(atom_numb[1]/atom_sum) # number of each atomic species
            for j in range(int(atom_sum)): # fill the above three lists
                line = data[i + j + 2]
                line = line.split()
                atom_pos = [float(line[3]) * base[0][0], float(line[4]) * base[1][1], float(line[5]) * base[2][2]]
                spin = float(line[2])
                atom_type = int(line[1])
                pos_list.append(atom_pos)
                spin_list.append(spin)
                type_list.append(atom_type)
            # Fill dictionary pos
            pos['CellName'] = name
            pos['LattConst'] = lat_const
            pos['LattTrue'] = lat_true
            pos['UnitVolume'] = lat_true[0]*lat_true[1]*lat_true[2]/atom_sum
            pos['Base'] = base
            pos['EleName'] = element_name
            pos['EleNum'] = element_number
            pos['AtomNum'] = atom_numb
            pos['AtomSum'] = atom_sum
            pos['IsSel'] = issel
            pos['LatType'] = latType
            pos['LattPnt'] = pos_list
            pos['SpinList'] = spin_list
            pos['TypeList'] = type_list
            pos_old_list.append(pos.copy())
    newPOS = celib.createSupercell(pos_old_list) # Create copy of dictionary for supercell creation (this is needed to
    # count the number of instances of each cluster in each DFT calculation)
    for i in range(len(newPOS)):
        newPOS[i] = vp.dismatcreate(newPOS[i]) # Find neighbor distance matrix
    # count numbers of each cluster
    cluster_count = []
    print("\nCounting clusters on each set..")
    for i in range(len(newPOS)):
        pos_pt = newPOS[i]
        # These are the two functions that count the instances of each cluster for a particular DFT data point
        cl_list = celib.clustercount1(cluster, pos_pt, TS=0.03)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # for i, clust in enumerate(cl_list[22]):
        #     atm1 = [pos_pt['LattPnt'][clust[0]]]
        #     atm2 = [pos_pt['LattPnt'][clust[1]]]
        #     atm3 = [pos_pt['LattPnt'][clust[2]]]
        #     poly = [atm1[0], atm2[0], atm3[0]]
        #     area = poly_area(poly)
            # plt.plot(i, area, 'bo')

            # fig = plt.figure(figsize=(4, 4))
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(atm1[0][0],atm1[0][1],atm1[0][2])
            # ax.scatter(atm2[0][0],atm2[0][1],atm2[0][2])
            # ax.scatter(atm3[0][0],atm3[0][1],atm3[0][2])
            # plt.show()

            # x = [atm1[0][0], atm2[0][0], atm3[0][0]]
            # y = [atm1[0][1], atm2[0][1], atm3[0][1]]
            # z = [atm1[0][2], atm2[0][2], atm3[0][2]]
            # vertices = [list(zip(x, y, z))]
            # poly = Poly3DCollection(vertices, alpha=0.8)
            #
            # ax.add_collection3d(poly)
            # ax.set_xlim(-1, 3)
            # ax.set_ylim(-1, 3)
            # ax.set_zlim(-1, 4)

        # plt.show()
        ##################################################################################################################################################
        ##################################################################################################################################################
        cl_count = celib.countCluster_Spins(cl_list, cluster, pos_pt)
        #cl_count = celib.clustercount2(cluster, pos_pt, TS=0.03)
        ##################################################################################################################################################
        ##################################################################################################################################################
        cluster_count.append(cl_count)
        print(cl_count)

    print("Done!\n")

    counts_new = []
    energies_new = []
    comps_new = []
    vols_new = []
    ###- eliminate duplicate data -###
    # If a multiple DFT data points give the same values for cluster counts, only keep the one with the lowest energy
    # for i in range(len(cluster_count)):
    #     if cluster_count[i] not in counts_new:
    #         counts_new.append(cluster_count[i])
    #         energies_new.append(enrgs[i])
    #         comps_new.append(comp_list[i])
    #         vols_new.append(pos_old_list[i]['UnitVolume'])
    #     elif enrgs[i] < energies_new[counts_new.index(cluster_count[i])]:
    #         energies_new[counts_new.index(cluster_count[i])] = enrgs[i]
    # print(len(counts_new))

    # print('cluster = ' + str(data_old['List']))
    # print('cluster_energy = ' + str(energies_new))
    # print('cluster_count = ' + str(counts_new))
    # print('comp = ' + str(comps_new))

    # Write to output lists
    # clust_out.extend(data_old['List'])
    # eng_out.extend(energies_new)
    # counts_out.extend(counts_new)
    # comps_out.extend(comps_new)
    # vols_out.extend(vols_new)

    print(len(cluster_count))
    return