# Version 1.9 This is a reworked version of NamHoon's cluster expansion code

import json
import vasp as vp
import celib as celib # Functions used to make the predictor matrix (list of how many times each cluster appears in each DFT data point)
import numpy as np

# Read in the DFT data (data_file) and cluster rules (cluster_file) and output lists of: How many times each cluster shows up (clust_out),
# the total DFT energy (eng_out), the composition of (comp_out)
def count_clusters(data_file, cluster_file, clust_out, eng_out, counts_out, comps_out):
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
            atom_numb = [set_data[0], set_data[1], set_data[2]] # Number of atoms of each type
            atom_numb[0] = int(atom_numb[0].strip('[').strip(','))
            atom_numb[1] = int(atom_numb[1].strip(','))
            atom_numb[2] = int(atom_numb[2].strip(']'))
            atom_sum = sum(atom_numb) # Total number of atoms in each DFT data point
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
        cl_count = celib.countCluster_Spins(cl_list, cluster, pos_pt)
        cluster_count.append(cl_count)
    print("Done!\n")

    counts_new = []
    energies_new = []
    comps_new = []
    ###- eliminate duplicate data -###
    # If a multiple DFT data points give the same values for cluster counts, only keep the one with the lowest energy
    for i in range(len(cluster_count)):
        if cluster_count[i] not in counts_new:
            counts_new.append(cluster_count[i])
            energies_new.append(enrgs[i])
            comps_new.append(comp_list[i])
        elif enrgs[i] < energies_new[counts_new.index(cluster_count[i])]:
            energies_new[counts_new.index(cluster_count[i])] = enrgs[i]
    #print(len(counts_new))

    # print('cluster = ' + str(data_old['List']))
    # print('cluster_energy = ' + str(energies_new))
    # print('cluster_count = ' + str(counts_new))
    # print('comp = ' + str(comps_new))

    # Write to output lists
    clust_out.extend(data_old['List'])
    eng_out.extend(energies_new)
    counts_out.extend(counts_new)
    comps_out.extend(comps_new)
    return
