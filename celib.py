# Version 1.9
# Hacked version of code from NamHoon
import copy
import numpy as np

import mathkit as MathKit



##########
# @def clustercount1(Clusterdes, POS, TS=0.2):
def clustercount1(Clusterdes, POS, TS=0.01):  # !
    '''
    enumerate and count clusters in a given lattce, this version is cleaner
    and more robust than the method below: clustercount
    Args:
        Clusterdes: Cluster description, in the format of list, somthing
                    like [[[0,1,2],[2.6,2.7,2.8]],[[1,1],[2.5]],[[2]]]
        POS: Dictionary containing the position information, in the format of POSCAR
        TS: Allowed variation of cluster bond length
        Outputs: ClusterLst, which is a list with all the description of indentified
                 clusters as specified in Clusterdes
    '''
    ClusterNum = len(Clusterdes) # Number of clusters
    ClusterLst = [[] for i in range(ClusterNum)] # Three deep list made up of the following:
    # [each DFT point [each cluster [ index (or indexes) of every group of atoms that make up a cluster]]]
    SumLst = [sum(POS['AtomNum'][0:i]) for i in range(POS['EleNum'])]

    for CInd, Cluster in enumerate(Clusterdes): # Loop over all clusters
        CSize = len(Cluster[0]) # Cluster Size

        if CSize == 1: # If cluster is a Monomer
            for i in range(POS['AtomNum'][Cluster[0][0]]): # for every atom of same type as monomer
                GInd = i + sum(POS['AtomNum'][0:Cluster[0][0]])
                ClusterLst[CInd].append([GInd]) # Add index of atom to list for every time momomer appears
        else: # For two and three atom terms...
            IndLst = [0] * CSize # incriment for every atom in cluster, counting up to the total number of atoms of each type (in the cluster)
            IndLstMax = [] # List of the maximum number of times an atom in a cluster could appear in the DFT data
                           # eg: Ni 64 Mn 32 In 32 and cluster [[1, 1],[0.5]] the maximum times 1 (or Mn) appears is 32
                           # so IndLstMax would be [31,31] (31 not 32 because of 0 indexing)
            GIndLst = [0] * CSize # list of actual index of the atoms in the cluster as stored in the POS dictionary
            GrpLst = MathKit.findGrpLst(Cluster[0])
            PermuteLst = MathKit.grporderPermute(Cluster[0], GrpLst)
            DistRef = MathKit.grpSort(Cluster[1], PermuteLst) # used to find permutations of neighbor distances and atom types that could represent a cluster
            # (really only applies to 3 atom terms)

            for Ele in Cluster[0]: # find max number of atoms of each type in the cluster
                IndLstMax.append(POS['AtomNum'][Ele] - 1)

            if -1 in IndLstMax:  # Return to the top of the loop
                continue  # !

            while (IndLst[-1] <= IndLstMax[-1]): # loop through all atoms

                for i, Ind in enumerate(IndLst):
                    GIndLst[i] = Ind + SumLst[Cluster[0][i]] # find new group indexes (indexes of each atom to test of they fit in the cluster)

                Dist = []
                for i in range(CSize):
                    for j in range(i + 1, CSize):
                        Dist.append(POS['dismat'][GIndLst[i]][GIndLst[j]]) # get neighbor distances

                flag = 1
                Dist = MathKit.grpSort(Dist, PermuteLst) # sort the list of neighbor distances so they mach up with reference distances for the cluster
                for Dind, D in enumerate(Dist):
                    if abs(D - DistRef[Dind]) > TS:
                        flag = 0 # if threshold is met, flag the distances as being the same
                        break
                if flag:
                    ClusterLst[CInd].append(list(GIndLst)) # add group index (the actual index of the atoms in the cluster) to the list of counted clusters

                MathKit.lstGrpAdd(IndLst, IndLstMax, GrpLst) # Increment index list

    return ClusterLst
##########

##########
def countCluster_Spins(ClusterLst, cluster, pos):
    # ClusterList is set up as follows [ for each DFT calculation [for each cluster [list of groups of atoms that make up a cluster] ] ]
    # cluster is the list of cluster rules eg: [[1, 1, 2],[0.70710678,0.5,0.86602540],[0]]
    # For everything except spin clusters, the number of times a cluster is counted is just the length of correct ClusterCount element.
    # For spin clusters, each pair of cluster atoms is referenced against the spin at the correct index to get the Ising like spin product
    ClusterCount = []
    Spins = pos['SpinList']
    for i in range(len(ClusterLst)):
        count = 0
        clust = cluster[i]
        if len(clust[0]) == 2:
            if clust[2][0] == 1: #Check to see if it is a spin cluster
                pairs = ClusterLst[i]
                for j in range(len(pairs)):
                    pair = pairs[j]
                    count += 2*Spins[pair[0]]*Spins[pair[1]] # find spin product
                ClusterCount.append(count) # append to list of cluster counts
            else:
                ClusterCount.append(2 * len(ClusterLst[i])) # append to list of cluster counts
        else:
            ClusterCount.append(3*len(ClusterLst[i])) # append to list of cluster counts
    print(ClusterCount)
    for i in range(len(ClusterCount)):
        ClusterCount[i] /= pos['AtomSum'] # normalize to counts per atom
    return ClusterCount


##########

##########
def countCluster_Spins2(ClusterLst, cluster, pos):
    # ClusterList is set up as follows [ for each DFT calculation [for each cluster [list of groups of atoms that make up a cluster] ] ]
    # cluster is the list of cluster rules eg: [[1, 1, 2],[0.70710678,0.5,0.86602540],[0]]
    # For everything except spin clusters, the number of times a cluster is counted is just the length of correct ClusterCount element.
    # For spin clusters, each pair of cluster atoms is referenced against the spin at the correct index to get the Ising like spin product
    ClusterCount = []
    zero_check = lambda a: a if a != 0 else 1
    Spins = pos['SpinList']
    for i in range(len(ClusterLst)):
        count = 0
        clust = cluster[i]
        if len(clust[0]) == 2:
            if clust[2][0] == 1: #Check to see if it is a spin cluster
                pairs = ClusterLst[i]
                for j in range(len(pairs)):
                    pair = pairs[j]
                    count += 2*Spins[pair[0]]*Spins[pair[1]] # find spin product
                ClusterCount.append(count) # append to list of cluster counts
            else:
                ClusterCount.append(2 * len(ClusterLst[i])) # append to list of cluster counts
        elif len(clust[0]) == 3:
            if clust[2][0] == 1: #Check to see if it is a spin cluster
                pairs = ClusterLst[i]
                for j in range(len(pairs)):
                    pair = pairs[j]
                    if round(pos['dismat'][pair[0]][pair[1]],4) == round(clust[1][0],4):
                        #count += 2 * Spins[pair[0]] * Spins[pair[1]]
                        if round(pos['dismat'][pair[0]][pair[2]],4) == round(clust[1][1],4):
                        #count += 2 * Spins[pair[0]]*Spins[pair[2]]
                            if round(pos['dismat'][pair[1]][pair[2]],4) == round(clust[1][2],4):
                                count += 2 * Spins[pair[1]]*Spins[pair[0]]
                ClusterCount.append(count) # append to list of cluster counts
            else:
                ClusterCount.append(3*len(ClusterLst[i])) # append to list of cluster counts
    print(ClusterCount)
    for i in range(len(ClusterCount)):
        ClusterCount[i] /= pos['AtomSum'] # normalize to counts per atom
    return ClusterCount


##########

##########
# Create a 2x2x2 supercell
def createSupercell(POS):
    new_POS = POS.copy()
    for mat_inc in range(len(POS)):
        numb_orig = len(POS[mat_inc]['LattPnt'])
        for i in range(3):
            new_POS[mat_inc]['AtomNum'][i] *= 8
        new_POS[mat_inc]['AtomSum'] *= 8
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    if [i, j, k] != [0, 0, 0]:
                        for atom in range(numb_orig):
                            site = POS[mat_inc]['LattPnt'][atom]
                            new_site = [site[0] + i * new_POS[mat_inc]['Base'][0][0],
                                        site[1] + j * new_POS[mat_inc]['Base'][1][1],
                                        site[2] + k * new_POS[mat_inc]['Base'][2][2]]
                            new_POS[mat_inc]['LattPnt'].append(new_site)
                            new_POS[mat_inc]['SpinList'].append(POS[mat_inc]['SpinList'][atom])
                            new_POS[mat_inc]['TypeList'].append(POS[mat_inc]['TypeList'][atom])
    for mat_inc, mat in enumerate(new_POS):
        T = [y for y, x, z in sorted(zip(mat['TypeList'], mat['SpinList'], mat['LattPnt']))]
        S = [x for y, x, z in sorted(zip(mat['TypeList'], mat['SpinList'], mat['LattPnt']))]
        L = [z for y, x, z in sorted(zip(mat['TypeList'], mat['SpinList'], mat['LattPnt']))]
        mat['TypeList'] = T
        mat['SpinList'] = S
        mat['LattPnt'] = L
        mat['dismat'] = [[]]
    return new_POS
##########






















########################################################################################################################
# # Version 1.9
# # Hacked version of code from NamHoon
# import copy
# import numpy as np
# import mathkit as MathKit
#
#
#
# def clustercount2(Clusterdes, POS, TS=0.01):
#     ClusterLst = []
#     '''
#     enumerate and count clusters in a given lattce, this version is cleaner
#     and more robust than the method below: clustercount
#
#     Args:
#         Clusterdes: Cluster description, in the format of list, somthing
#                     like [[[0,1,2],[2.6,2.7,2.8]],[[1,1],[2.5]],[[2]]]
#         POS: Dictionary containing the position information, in the format of POSCAR
#         TS: Allowed variation of cluster bond length
#     '''
#     NumClust = len(Clusterdes)
#     for c in range(NumClust):
#         cluster = Clusterdes[c]
#         if len(cluster[0]) == 1:
#             count = POS['AtomNum'][cluster[0][0]]
#         elif len(cluster[0]) == 2:
#             count = 0
#             for i in range(len(POS['dismat'])):
#                 for j in range(len(POS['dismat'][i])):
#                     if round(POS['dismat'][i][j],4) == round([1][0],4):
#                         if POS['TypeList'][i] == cluster[0][0] and POS['TypeList'][j] == cluster[0][1]:
#                             if cluster[2][0] == 0:
#                                 count += 1
#                             elif cluster[2][0] == 1:
#                                 count += POS['SpinList'][i] * POS['SpinList'][j]
#                         elif POS['TypeList'][i] == cluster[0][1] and POS['TypeList'][j] == cluster[0][0]:
#                             if cluster[2][0] == 0:
#                                 count += 1
#                             elif cluster[2][0] == 1:
#                                 count += POS['SpinList'][i] * POS['SpinList'][j]
#         elif len(cluster[0]) == 3:
#             count = 0
#             for i in range(len(POS['dismat'])):
#                 if POS['TypeList'][i] in cluster[0]:
#                     for j in range(len(POS['dismat'][i])):
#                         if POS['TypeList'][j] in cluster[0]:
#                             for k in range(len(POS['dismat'][j])):
#                                 if [POS['TypeList'][i],POS['TypeList'][j],POS['TypeList'][k]] == cluster[0] and [round(POS['dismat'][i][j],4), round(POS['dismat'][i][k],4), round(POS['dismat'][j][k],4)] == [round(cluster[1][0],4), round(cluster[1][1],4), round(cluster[1][2],4)]:
#                                     if cluster[2][0] == 1:
#                                         count += POS['SpinList'][i] * POS['SpinList'][j]
#                                     elif cluster[2][0] == 0:
#                                         count += 1
#                                 if [POS['TypeList'][i],POS['TypeList'][k],POS['TypeList'][j]] == cluster[0] and [round(POS['dismat'][i][k],4), round(POS['dismat'][i][j],4), round(POS['dismat'][j][k],4)] == [round(cluster[1][0],4), round(cluster[1][1],4), round(cluster[1][2],4)]:
#                                     if cluster[2][0] != 1:
#                                         count += 1
#                                 if [POS['TypeList'][j],POS['TypeList'][i],POS['TypeList'][k]] == cluster[0] and [round(POS['dismat'][j][i],4), round(POS['dismat'][j][k],4), round(POS['dismat'][i][k],4)] == [round(cluster[1][0],4), round(cluster[1][1],4), round(cluster[1][2],4)]:
#                                     if cluster[2][0] != 1:
#                                         count += 1
#                                 if [POS['TypeList'][j],POS['TypeList'][k],POS['TypeList'][i]] == cluster[0] and [round(POS['dismat'][j][k],4), round(POS['dismat'][j][i],4), round(POS['dismat'][i][k],4)] == [round(cluster[1][0],4), round(cluster[1][1],4), round(cluster[1][2],4)]:
#                                     if cluster[2][0] != 1:
#                                         count += 1
#                                 if [POS['TypeList'][k],POS['TypeList'][i],POS['TypeList'][j]] == cluster[0] and [round(POS['dismat'][k][i],4), round(POS['dismat'][k][j],4), round(POS['dismat'][i][j],4)] == [round(cluster[1][0],4), round(cluster[1][1],4), round(cluster[1][2],4)]:
#                                     if cluster[2][0] != 1:
#                                         count += 1
#                                 if [POS['TypeList'][k],POS['TypeList'][j],POS['TypeList'][i]] == cluster[0] and [round(POS['dismat'][k][j],4), round(POS['dismat'][k][i],4), round(POS['dismat'][i][j],4)] == [round(cluster[1][0],4), round(cluster[1][1],4), round(cluster[1][2],4)]:
#                                     if cluster[2][0] != 1:
#                                         count += 1
#         ClusterLst.append(count)
#     return ClusterLst
#
#
#
# # @def clustercount1(Clusterdes, POS, TS=0.2):
# def clustercount1(Clusterdes, POS, TS=0.01):  # !
#     '''
#     enumerate and count clusters in a given lattce, this version is cleaner
#     and more robust than the method below: clustercount
#
#     Args:
#         Clusterdes: Cluster description, in the format of list, somthing
#                     like [[[0,1,2],[2.6,2.7,2.8]],[[1,1],[2.5]],[[2]]]
#         POS: Dictionary containing the position information, in the format of POSCAR
#         TS: Allowed variation of cluster bond length
#         Outputs: ClusterLst, which is a list with all the description of indentified
#                  clusters as specified in Clusterdes
#     '''
#     ClusterNum = len(Clusterdes) # Number of clusters
#     ClusterLst = [[] for i in range(ClusterNum)] # Three deep list made up of the following:
#     # [each DFT point [each cluster [ index (or indexes) of every group of atoms that make up a cluster]]]
#     SumLst = [sum(POS['AtomNum'][0:i]) for i in range(POS['EleNum'])]
#
#     for CInd, Cluster in enumerate(Clusterdes): # Loop over all clusters
#         CSize = len(Cluster[0]) # Cluster Size
#
#         if CSize == 1: # If cluster is a Monomer
#             for i in range(POS['AtomNum'][Cluster[0][0]]): # for every atom of same type as monomer
#                 GInd = i + sum(POS['AtomNum'][0:Cluster[0][0]])
#                 ClusterLst[CInd].append([GInd]) # Add index of atom to list for every time momomer appears
#         else: # For two and three atom terms...
#             IndLst = [0] * CSize # incriment for every atom in cluster, counting up to the total number of atoms of each type (in the cluster)
#             IndLstMax = [] # List of the maximum number of times an atom in a cluster could appear in the DFT data
#                            # eg: Ni 64 Mn 32 In 32 and cluster [[1, 1],[0.5]] the maximum times 1 (or Mn) appears is 32
#                            # so IndLstMax would be [31,31] (31 not 32 because of 0 indexing)
#             GIndLst = [0] * CSize # list of actual index of the atoms in the cluster as stored in the POS dictionary
#             GrpLst = MathKit.findGrpLst(Cluster[0])
#             PermuteLst = MathKit.grporderPermute(Cluster[0], GrpLst)
#             DistRef = MathKit.grpSort(Cluster[1], PermuteLst) # used to find permutations of neighbor distances and atom types that could represent a cluster
#             # (really only applies to 3 atom terms)
#
#             for Ele in Cluster[0]: # find max number of atoms of each type in the cluster
#                 IndLstMax.append(POS['AtomNum'][Ele] - 1)
#
#             if -1 in IndLstMax:  # Return to the top of the loop
#                 continue  # !
#
#             while (IndLst[-1] <= IndLstMax[-1]): # loop through all atoms
#
#                 for i, Ind in enumerate(IndLst):
#                     GIndLst[i] = Ind + SumLst[Cluster[0][i]] # find new group indexes (indexes of each atom to test of they fit in the cluster)
#
#                 Dist = []
#                 for i in range(CSize):
#                     for j in range(i + 1, CSize):
#                         Dist.append(POS['dismat'][GIndLst[i]][GIndLst[j]]) # get neighbor distances
#
#                 flag = 1
#                 Dist = MathKit.grpSort(Dist, PermuteLst) # sort the list of neighbor distances so they mach up with reference distances for the cluster
#                 for Dind, D in enumerate(Dist):
#                     if abs(D - DistRef[Dind]) > TS:
#                         flag = 0 # if threshold is met, flag the distances as being the same
#                         break
#                 if flag:
#                     ClusterLst[CInd].append(list(GIndLst)) # add group index (the actual index of the atoms in the cluster) to the list of counted clusters
#
#                 MathKit.lstGrpAdd(IndLst, IndLstMax, GrpLst) # Increment index list
#
#     return ClusterLst
#
#
#
# def countCluster_Spins(ClusterLst, cluster, pos):
#     # ClusterList is set up as follows [ for each DFT calculation [for each cluster [list of groups of atoms that make up a cluster] ] ]
#     # cluster is the list of cluster rules eg: [[1, 1, 2],[0.70710678,0.5,0.86602540],[0]]
#     # For everything except spin clusters, the number of times a cluster is counted is just the length of correct ClusterCount element.
#     # For spin clusters, each pair of cluster atoms is referenced against the spin at the correct index to get the Ising like spin product
#     ClusterCount = []
#     Spins = pos['SpinList']
#     for i in range(len(ClusterLst)):
#         count = 0
#         clust = cluster[i]
#         if len(clust[0]) == 1:
#             if clust[2][0] == 1:
#                 pairs = ClusterLst[i]
#                 for j in range(len(pairs)):
#                     pair = pairs[j]
#                     count += np.power(Spins[pair[0]], 2)
#                 ClusterCount.append(count)
#             else:
#                 ClusterCount.append(len(ClusterLst[i])) # append to list of cluster counts
#         elif len(clust[0]) == 2:
#             if clust[2][0] == 1: #Check to see if it is a spin cluster
#                 pairs = ClusterLst[i]
#                 for j in range(len(pairs)):
#                     pair = pairs[j]
#                     count += 2*Spins[pair[0]]*Spins[pair[1]] # find spin product
#                 ClusterCount.append(count) # append to list of cluster counts
#             else:
#                 ClusterCount.append(2 * len(ClusterLst[i])) # append to list of cluster counts
#         else:
#             ClusterCount.append(3*len(ClusterLst[i])) # append to list of cluster counts
#     print(ClusterCount)
#     for i in range(len(ClusterCount)):
#         ClusterCount[i] /= pos['AtomSum'] # normalize to counts per atom
#     return ClusterCount
#
#
#
# def countCluster_Spins_Potts(ClusterLst, cluster, pos):
#     # ClusterList is set up as follows [ for each DFT calculation [for each cluster [list of groups of atoms that make up a cluster] ] ]
#     # cluster is the list of cluster rules eg: [[1, 1, 2],[0.70710678,0.5,0.86602540],[0]]
#     # For everything except spin clusters, the number of times a cluster is counted is just the length of correct ClusterCount element.
#     # For spin clusters, each pair of cluster atoms is referenced against the spin at the correct index to get the Ising like spin product
#     ClusterCount = []
#     zero_check = lambda a: a if a != 0 else 1
#     Spins = pos['SpinList']
#     delta = lambda x ,y : 1 if (x == y) else 0
#     for i in range(len(ClusterLst)):
#         count = 0
#         clust = cluster[i]
#         if len(clust[0]) == 1:
#             if clust[2][0] == 1:
#                 pairs = ClusterLst[i]
#                 for j in range(len(pairs)):
#                     pair = pairs[j]
#                     count += np.power(Spins[pair[0]], 2)
#                 ClusterCount.append(count)
#             else:
#                 ClusterCount.append(len(ClusterLst[i])) # append to list of cluster counts
#         elif len(clust[0]) == 2:
#             if clust[2][0] == 1: #Check to see if it is a spin cluster
#                 pairs = ClusterLst[i]
#                 for j in range(len(pairs)):
#                     pair = pairs[j]
#                     count += 2 * delta(Spins[pair[0]], Spins[pair[1]]) # find spin product
#                 ClusterCount.append(count) # append to list of cluster counts
#             else:
#                 ClusterCount.append(2 * len(ClusterLst[i])) # append to list of cluster counts
#         elif len(clust[0]) == 3 and clust[2][0] == 1: #Check to see if it is a spin cluster
#             pairs = ClusterLst[i]
#             for j in range(len(pairs)):
#                 pair = pairs[j]
#                 # should this be else if????################################################################################################################################
#                 if round(pos['dismat'][pair[0]][pair[1]],4) == round(clust[1][0],4) and pos['TypeList'][pair[0]] == clust[0][0] and pos['TypeList'][pair[1]] == clust[0][1]:
#                     count += 1 * delta(Spins[pair[0]], Spins[pair[1]])
#                 if round(pos['dismat'][pair[0]][pair[2]],4) == round(clust[1][0],4) and pos['TypeList'][pair[0]] == clust[0][0] and pos['TypeList'][pair[2]] == clust[0][1]:
#                     count += 1 * delta(Spins[pair[0]], Spins[pair[2]])
#                 if round(pos['dismat'][pair[1]][pair[2]],4) == round(clust[1][0],4) and pos['TypeList'][pair[1]] == clust[0][0] and pos['TypeList'][pair[2]] == clust[0][1]:
#                     count += 1 * delta(Spins[pair[1]], Spins[pair[2]])
#             ClusterCount.append(count) # append to list of cluster counts
#             #print(count)
#         else:
#             ClusterCount.append(len(ClusterLst[i])) # append to list of cluster counts
#             # pairs = ClusterLst[i]
#             # for j in range(len(pairs)):
#             #     pair = pairs[j]
#             #     if len(pair) > 1:
#             #         p1 = round(pos['dismat'][pair[0]][pair[1]],4)
#             #         p2 = round(pos['dismat'][pair[0]][pair[2]], 4)
#             #         p3 = round(pos['dismat'][pair[1]][pair[2]], 4)
#             #         x = 0
#     print(ClusterCount)
#     for i in range(len(ClusterCount)):
#         ClusterCount[i] /= pos['AtomSum'] # normalize to counts per atom
#     return ClusterCount
#
#
#
# def countCluster_Spins2(ClusterLst, cluster, pos):
#     # ClusterList is set up as follows [ for each DFT calculation [for each cluster [list of groups of atoms that make up a cluster] ] ]
#     # cluster is the list of cluster rules eg: [[1, 1, 2],[0.70710678,0.5,0.86602540],[0]]
#     # For everything except spin clusters, the number of times a cluster is counted is just the length of correct ClusterCount element.
#     # For spin clusters, each pair of cluster atoms is referenced against the spin at the correct index to get the Ising like spin product
#     ClusterCount = []
#     zero_check = lambda a: a if a != 0 else 1
#     Spins = pos['SpinList']
#     for i in range(len(ClusterLst)):
#         count = 0
#         clust = cluster[i]
#         if len(clust[0]) == 1:
#             if clust[2][0] == 1:
#                 pairs = ClusterLst[i]
#                 for j in range(len(pairs)):
#                     pair = pairs[j]
#                     count += np.power(Spins[pair[0]], 2)
#                 ClusterCount.append(count)
#             else:
#                 ClusterCount.append(len(ClusterLst[i])) # append to list of cluster counts
#         elif len(clust[0]) == 2:
#             if clust[2][0] == 1: #Check to see if it is a spin cluster
#                 pairs = ClusterLst[i]
#                 for j in range(len(pairs)):
#                     pair = pairs[j]
#                     count += 2*Spins[pair[0]]*Spins[pair[1]] # find spin product
#                 ClusterCount.append(count) # append to list of cluster counts
#             else:
#                 ClusterCount.append(2 * len(ClusterLst[i])) # append to list of cluster counts
#         elif len(clust[0]) == 3 and clust[2][0] == 1: #Check to see if it is a spin cluster
#             pairs = ClusterLst[i]
#             for j in range(len(pairs)):
#                 pair = pairs[j]
#                 # should this be else if????################################################################################################################################
#                 if round(pos['dismat'][pair[0]][pair[1]],4) == round(clust[1][0],4) and pos['TypeList'][pair[0]] == clust[0][0] and pos['TypeList'][pair[1]] == clust[0][1]:
#                     count += 1 * Spins[pair[0]] * Spins[pair[1]]
#                 if round(pos['dismat'][pair[0]][pair[2]],4) == round(clust[1][0],4) and pos['TypeList'][pair[0]] == clust[0][0] and pos['TypeList'][pair[2]] == clust[0][1]:
#                     count += 1 * Spins[pair[0]]*Spins[pair[2]]
#                 if round(pos['dismat'][pair[1]][pair[2]],4) == round(clust[1][0],4) and pos['TypeList'][pair[1]] == clust[0][0] and pos['TypeList'][pair[2]] == clust[0][1]:
#                     count += 1 * Spins[pair[1]]*Spins[pair[2]]
#             ClusterCount.append(count) # append to list of cluster counts
#             #print(count)
#         else:
#             ClusterCount.append(len(ClusterLst[i])) # append to list of cluster counts
#             # pairs = ClusterLst[i]
#             # for j in range(len(pairs)):
#             #     pair = pairs[j]
#             #     if len(pair) > 1:
#             #         p1 = round(pos['dismat'][pair[0]][pair[1]],4)
#             #         p2 = round(pos['dismat'][pair[0]][pair[2]], 4)
#             #         p3 = round(pos['dismat'][pair[1]][pair[2]], 4)
#             #         x = 0
#     print(ClusterCount)
#     for i in range(len(ClusterCount)):
#         ClusterCount[i] /= pos['AtomSum'] # normalize to counts per atom
#     return ClusterCount
#
#
#
# # Create a 2x2x2 supercell
# def createSupercell(POS):
#     new_POS = POS.copy()
#     for mat_inc in range(len(POS)):
#         numb_orig = len(POS[mat_inc]['LattPnt'])
#         for i in range(3):
#             new_POS[mat_inc]['AtomNum'][i] *= 8
#         new_POS[mat_inc]['AtomSum'] *= 8
#         for i in range(2):
#             for j in range(2):
#                 for k in range(2):
#                     if [i, j, k] != [0, 0, 0]:
#                         for atom in range(numb_orig):
#                             site = POS[mat_inc]['LattPnt'][atom]
#                             new_site = [site[0] + i * new_POS[mat_inc]['Base'][0][0],
#                                         site[1] + j * new_POS[mat_inc]['Base'][1][1],
#                                         site[2] + k * new_POS[mat_inc]['Base'][2][2]]
#                             new_POS[mat_inc]['LattPnt'].append(new_site)
#                             new_POS[mat_inc]['SpinList'].append(POS[mat_inc]['SpinList'][atom])
#                             new_POS[mat_inc]['TypeList'].append(POS[mat_inc]['TypeList'][atom])
#     for mat_inc, mat in enumerate(new_POS):
#         T = [y for y, x, z in sorted(zip(mat['TypeList'], mat['SpinList'], mat['LattPnt']))]
#         S = [x for y, x, z in sorted(zip(mat['TypeList'], mat['SpinList'], mat['LattPnt']))]
#         L = [z for y, x, z in sorted(zip(mat['TypeList'], mat['SpinList'], mat['LattPnt']))]
#         mat['TypeList'] = T
#         mat['SpinList'] = S
#         mat['LattPnt'] = L
#         mat['dismat'] = [[]]
#     return new_POS
