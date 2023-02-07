# Version 1.9 This is a reworked version of NamHoon's cluster expansion code

import json
import vasp as vp
import celib as celib # Functions used to make the predictor matrix (list of how many times each cluster appears in each DFT data point)
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

#unit normal vector of plane defined by points a, b, and c
def unit_normal(a, b, c):
    x = np.linalg.det([[1,a[1],a[2]],
         [1,b[1],b[2]],
         [1,c[1],c[2]]])
    y = np.linalg.det([[a[0],1,a[2]],
         [b[0],1,b[2]],
         [c[0],1,c[2]]])
    z = np.linalg.det([[a[0],a[1],1],
         [b[0],b[1],1],
         [c[0],c[1],1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x/magnitude, y/magnitude, z/magnitude)

#area of polygon poly
def poly_area(poly):
    if len(poly) < 3: # not a plane - no area
        return 0
    total = [0, 0, 0]
    N = len(poly)
    for i in range(N):
        vi1 = poly[i]
        vi2 = poly[(i+1) % N]
        prod = np.cross(vi1, vi2)
        total[0] += prod[0]
        total[1] += prod[1]
        total[2] += prod[2]
    result = np.dot(total, unit_normal(poly[0], poly[1], poly[2]))
    return abs(result/2)

# Read in the DFT data (data_file) and cluster rules (cluster_file) and output lists of: How many times each cluster shows up (clust_out),
# the total DFT energy (eng_out), the composition of (comp_out)
def count_clusters(data_file, cluster_file, clust_out, eng_out, counts_out, comps_out, vols_out):
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
    ###- If a multiple DFT data points give the same values for cluster counts, only keep the one with the lowest energy
    for i in range(len(cluster_count)):
        if cluster_count[i] not in counts_new:
            counts_new.append(cluster_count[i])
            energies_new.append(enrgs[i])
            comps_new.append(comp_list[i])
            vols_new.append(pos_old_list[i]['UnitVolume'])
        elif enrgs[i] < energies_new[counts_new.index(cluster_count[i])]:
            energies_new[counts_new.index(cluster_count[i])] = enrgs[i]
    print(len(counts_new))

    # print('cluster = ' + str(data_old['List']))
    # print('cluster_energy = ' + str(energies_new))
    # print('cluster_count = ' + str(counts_new))
    # print('comp = ' + str(comps_new))

    # Write to output lists
    clust_out.extend(data_old['List'])
    eng_out.extend(energies_new)
    counts_out.extend(counts_new)
    comps_out.extend(comps_new)
    vols_out.extend(vols_new)

    # clust_out.extend(data_old['List'])
    # eng_out.extend(enrgs)
    # counts_out.extend(cluster_count)
    # comps_out.extend(comp_list)
    # vols_out.extend([pos_old_list[i]['UnitVolume'] for i in range(len(pos_old_list))])
    print(len(cluster_count))
    return

def count_clusters_Potts(data_file, cluster_file, clust_out, eng_out, counts_out, comps_out, vols_out):
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
        cl_count = celib.countCluster_Spins_Potts(cl_list, cluster, pos_pt)
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

    clust_out.extend(data_old['List'])
    eng_out.extend(enrgs)
    counts_out.extend(cluster_count)
    comps_out.extend(comp_list)
    vols_out.extend([pos_old_list[i]['UnitVolume'] for i in range(len(pos_old_list))])
    print(len(cluster_count))
    return

def count_clusters_duplicate(data_file, cluster_file, clust_out, eng_out, counts_out, comps_out):
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
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        for i, clust in enumerate(cl_list[22]):
            atm1 = [pos_pt['LattPnt'][clust[0]]]
            atm2 = [pos_pt['LattPnt'][clust[1]]]
            atm3 = [pos_pt['LattPnt'][clust[2]]]
            poly = [atm1[0], atm2[0], atm3[0]]
            area = poly_area(poly)
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


        cl_count = celib.countCluster_Spins(cl_list, cluster, pos_pt)
        cluster_count.append(cl_count)
    print("Done!\n")

    counts_new = []
    energies_new = []
    comps_new = []
    ###- eliminate duplicate data -###
    # If a multiple DFT data points give the same values for cluster counts, only keep the one with the lowest energy
    for i in range(len(cluster_count)):
         # if cluster_count[i] not in counts_new:
         #     counts_new.append(cluster_count[i])
         #     energies_new.append(enrgs[i])
         #     comps_new.append(comp_list[i])
         # elif enrgs[i] < energies_new[counts_new.index(cluster_count[i])]:
         #     energies_new[counts_new.index(cluster_count[i])] = enrgs[i]

        counts_new.append(cluster_count[i])
        energies_new.append(enrgs[i])
        comps_new.append(comp_list[i])

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

def count_clusters_second_fit(data_file, cluster_file, clust_out, eng_in, eng_out, counts_out, comps_out):
    # Read in Cluster rules
    with open(cluster_file) as f:
        data_old = json.load(f)
    cluster = data_old['List'] # list of clusters
    # Read in DFT data
    data_f = open(data_file)
    data = data_f.readlines() # raw data file (asci)
    data_f.close()
    pos = {} # Dictionary containing all data and metadata for each DFT calculation
    enrgs = eng_in # list of all DFT energies
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
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        for i, clust in enumerate(cl_list[22]):
            atm1 = [pos_pt['LattPnt'][clust[0]]]
            atm2 = [pos_pt['LattPnt'][clust[1]]]
            atm3 = [pos_pt['LattPnt'][clust[2]]]
            poly = [atm1[0], atm2[0], atm3[0]]


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

def save_data(output_file, clusters, energies, counts, comps):
    file = open(output_file, 'w')
    file.write("Clusters \n")
    [file.write(str(line) + "\n") for line in clusters]
    file.write("Counts\n")
    [file.write(str(comps[i]) + ", " + str(energies[i]) + ", " + str(counts[i]) + "\n") for i in range(len(energies))]
    file.close()

def read_data(input_file, cluster_file, clusters, energies, counts, comps):
    file = open(input_file, 'r')
    lines = file.readlines()
    fill_counts = False
    clusters_string = []
    for line in lines:
        if "Clusters" not in line and "Counts" not in line and fill_counts == False:
            clusters_string.append(line.strip('\n'))
        elif "Clusters" not in line and "Counts" not in line and fill_counts == True:
            line = line.split(',')
            comps.append(float(line[0]))
            energies.append(float(line[1]))
            counts.append([float(line[i].strip(" [").strip("]\n")) for i in range(2,len(line))])
        elif "Counts" in line:
            fill_counts = True
    with open(cluster_file) as f:
        data_old = json.load(f)
    clusters.extend(data_old['List'])
    clusters_as_string = [str(cluster) for cluster in clusters]