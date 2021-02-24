# Version 1.9

import json
import vasp as vp
import celib as celib
import numpy as np
def count_clusters(data_file, cluster_file, clust_out, eng_out, counts_out, comps_out):
    with open(cluster_file) as f:
        data_old = json.load(f)
    cluster = data_old['List']
    data_f = open(data_file)
    data = data_f.readlines()
    data_f.close()
    sets = 0
    pos = {}
    enrgs = []
    pos_old_list = []
    comp_list = []
    count = -1
    for i in range(len(data)):
        if "#" in data[i]:
            count += 1
            sets += 1
            species = (data[i].split())
            species.pop(0)
            set_data = data[i + 1]
            set_data = set_data.split()
            name = set_data[3]
            element_number = 3
            atom_numb = [set_data[0], set_data[1], set_data[2]]
            atom_numb[0] = int(atom_numb[0].strip('[').strip(','))
            atom_numb[1] = int(atom_numb[1].strip(','))
            atom_numb[2] = int(atom_numb[2].strip(']'))
            atom_sum = sum(atom_numb)
            factor = np.power(atom_sum/2, 1/3)/2
            lat_const = factor
            if set_data[5] == 'mart':
                base = [[factor, 0, 0], [0, factor, 0], [0, 0, factor * 1.5]]
            elif set_data[5] == 'aust':
                base = [[factor, 0, 0], [0, factor, 0], [0, 0, factor]]
            element_name = ['Ni', 'Mn', 'In']
            issel = 0
            latType = 'Direct'
            enrgs.append(float(set_data[4]) / atom_sum)
            pos_list = []
            spin_list = []
            type_list = []
            comp_list.append(atom_numb[1]/atom_sum)
            for j in range(int(atom_sum)):
                line = data[i + j + 2]
                line = line.split()
                atom_pos = [float(line[3]) * base[0][0], float(line[4]) * base[1][1], float(line[5]) * base[2][2]]
                spin = float(line[2])
                atom_type = int(line[1])
                pos_list.append(atom_pos)
                spin_list.append(spin)
                type_list.append(atom_type)
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
    newPOS = celib.createSupercell(pos_old_list)
    for i in range(len(newPOS)):
        newPOS[i] = vp.dismatcreate(newPOS[i])

    # count numbers of each cluster
    cluster_count = []
    print("\nCounting clusters on each set..")
    for i in range(len(newPOS)):
        pos = newPOS[i]
        cl_list = celib.clustercount1(cluster, pos, TS=0.03)
        cl_count = celib.countCluster_Spins(cl_list, cluster, pos)
        cluster_count.append(cl_count)
    print("Done!\n")

    counts_new = []
    energies_new = []
    comps_new = []
    ###- eliminate duplicate data -###
    for i in range(len(cluster_count)):
        if cluster_count[i] not in counts_new:
            counts_new.append(cluster_count[i])
            energies_new.append(enrgs[i])
            comps_new.append(comp_list[i])
        elif enrgs[i] < energies_new[counts_new.index(cluster_count[i])]:
            energies_new[counts_new.index(cluster_count[i])] = enrgs[i]
    print(len(counts_new))

    print('cluster = ' + str(data_old['List']))
    print('cluster_energy = ' + str(energies_new))
    print('cluster_count = ' + str(counts_new))
    print('comp = ' + str(comps_new))

    clust_out.extend(data_old['List'])
    eng_out.extend(energies_new)
    counts_out.extend(counts_new)
    comps_out.extend(comps_new)
    return
