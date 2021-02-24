# Version 1.9

import copy
import numpy as np

import mathkit as MathKit


##########
# @def ceFind(SubLatt, POSRef, NCut=3, Isprint=0, DCut='default'):
def ceFind(SubLatt, POSRef, NCut, DCut, Isprint=0):  # !
    '''
    Method to find the clusters with a given reference lattice

    Args:
        SubLatt: the projection of solid solution into reference lattice
                 something like [[0,1],[1,2],[3,4]];
        POSRef: POSCAR dictionary for reference lattice
        NCut: Cutoff size of clusters (default: 3)
        DCut: Cutoff length of each dimension of the cluster
              (default: Half of the box size)
    '''

    # @print('#############################################################')
    if DCut == 'default':
        DCut = 100.0
        TS = 0.3
        for i in range(3):
            DMax = max(POSRef['Base'][i]) / 2.0 + TS
            if DMax < DCut:
                DCut = DMax
    # @print('Cutoff cluster length is %f A' %DCut)

    NSubLatt = POSRef['EleNum']
    ClusterDesLst = []
    PrimDistLst = []
    AllPrimLattLst = []
    ClusterNum = []
    IndMax = max(SubLatt[-1])

    FreeSubLatt = []
    FreePrim = []
    for i in range(NSubLatt):
        if len(SubLatt[i]) == 1:  # !
            FreeSubLatt.append(SubLatt[i][0:-1])  # !
        if len(SubLatt[i]) > 1:
            FreePrim.append(i)
            FreeSubLatt.append(SubLatt[i][0:-1])  # !
        # @FreeSubLatt.append(SubLatt[i][0:-1]) #B #Get rid of last one

    NFreePrim = len(FreePrim)
    NFreeSubLatt = len(FreeSubLatt)
    FreePrim = np.array(FreePrim)
    FreeSubLatt = np.array(FreeSubLatt)
    # B #print(NFreePrim,NFreeSubLatt,FreePrim,FreeSubLatt);

    for N in range(2, NCut + 1):
        PrimIndLst = [0] * N
        PrimDistLst.append([])
        AllPrimLattLst.append([])
        while (PrimIndLst[-1] <= NFreePrim - 1):
            # B #print(PrimIndLst);
            PrimLattLst = list(FreePrim[PrimIndLst])
            AllPrimLattLst[N - 2].append(PrimLattLst)

            DistLst = findCluster(POSRef, PrimLattLst, DCut)

            PrimDistLst[N - 2].append(DistLst)
            PrimIndLst = MathKit.lstOrderAdd(PrimIndLst, [NFreePrim - 1] * N)

    # @print('The Distance list of primary lattice is '+str(PrimDistLst))
    # @print('The cluster made from primary lattice is '+str(AllPrimLattLst))

    ClusterDesLst.append([])
    ClusterNum.append(0)
    for SubLatt in FreeSubLatt:
        if SubLatt:
            # @print(SubLatt)
            for Latt in SubLatt:
                # @ClusterDesLst[0].append([Latt])
                ClusterDesLst[0].append([[Latt]])  # !
                ClusterNum[0] += 1
    # B #print(ClusterDesLst);

    for N in range(2, NCut + 1):
        IndLst = [0] * N
        ClusterDesLst.append([])
        ClusterNum.append(0)

        while (IndLst[-1] <= NFreeSubLatt - 1):
            LattLst = list(FreeSubLatt[IndLst])

            if not [] in LattLst:
                # B #print('LattLst = '+str(LattLst));
                PrimLattLst = [0] * N
                for LattInd, Latt in enumerate(LattLst):
                    if Latt in list(FreeSubLatt):
                        SubInd = list(FreeSubLatt).index(Latt)
                        PrimLattLst[LattInd] = SubInd
                    else:
                        print('Cannot Latt in FreeSubLatt!!!')
                # @print('PrimLattLst = '+str(PrimLattLst))

                if PrimLattLst in AllPrimLattLst[N - 2]:
                    PrimInd = AllPrimLattLst[N - 2].index(PrimLattLst)
                    DistLst = PrimDistLst[N - 2][PrimInd]
                else:
                    print('Cannot find the relevant PrimLattLst!!!')
                    break

                for Dist in DistLst:
                    PermuteLattLst = MathKit.listPermute(LattLst)
                    for PermuteLst in PermuteLattLst:
                        Dist = [round(elem, 2) for elem in Dist]  # !
                        Cluster = [PermuteLst, Dist]
                        if not Cluster in ClusterDesLst[N - 2]:
                            ClusterDesLst[N - 1].append(Cluster)
                            ClusterNum[N - 1] += 1

            IndLst = MathKit.lstOrderAdd(IndLst, [NFreeSubLatt - 1] * N)  # B #Next one
    ClusterSum = sum(ClusterNum)
    # @print('#############################################################')

    if (Isprint):
        # @print('#############################################################')
        print('%i indepedent Clusters have been found in this structure' % (ClusterSum))
        for N in range(1, NCut + 1):
            print('%i Clusters with %i atoms is given below:' % (ClusterNum[N - 1], N))
            ClusterStr = ''
            for Cluster in ClusterDesLst[N - 1]:
                ClusterStr += str(Cluster)
                # @ClusterStr+='\t'
                ClusterStr += '\n'  # !
            print(ClusterStr)
        # @print('#############################################################')

    return ClusterSum, ClusterNum, ClusterDesLst


##########


##########
def findCluster(POSRef, LattLst, DCut):
    '''
    Find the Distance Lst for a given PrimAtomLst

    Args:
        POSRef: dictionary of POSRef
        PrimAtomLst: atom list in PrimAtomLst
        DCut: Cutoff distance of

    '''

    NLst = len(LattLst)
    IndLst = [0] * NLst
    GIndLst = [0] * NLst
    # @TS = 0.05*NLst
    TS = 0.1  # !
    DistLst = []
    IndLstMax = []

    for i in range(NLst):
        IndLstMax.append(POSRef['AtomNum'][LattLst[i]] - 1)

    while (IndLst[-1] <= IndLstMax[-1]):
        for i, Ind in enumerate(IndLst):
            # @Indtmp = LattLst[i] - 1
            Indtmp = LattLst[i]  # !
            GIndLst[i] = Ind + sum(POSRef['AtomNum'][0:Indtmp])

        Dist = []
        # @GrpLst = MathKit.findGrpLst(LattLst)
        for i in range(NLst):
            for j in range(i + 1, NLst):
                Distmp = POSRef['dismat'][GIndLst[i]][GIndLst[j]]
                Dist.append(Distmp)

        GrpLst = MathKit.findGrpLst(LattLst)  # !
        PermuteGrpLst = MathKit.grporderPermute(LattLst, GrpLst)
        Dist = MathKit.grpSort(Dist, PermuteGrpLst)

        flag = 1
        for Disttmp in DistLst:
            Distmp = MathKit.grpSort(Disttmp, PermuteGrpLst)
            # @DistDiff = sum(abs(np.array(Dist)-np.array(Disttmp)))
            DistDiff = sum(abs(np.array(Dist) - np.array(Distmp)))  # !
            if (DistDiff < TS):
                flag = 0
        if (min(Dist) > 0) & (max(Dist) < DCut) & flag:
            DistLst.append(Dist)

        # @GrpLst = MathKit.findGrpLst(LattLst)
        IndLst = MathKit.lstGrpAdd(IndLst, IndLstMax, GrpLst)

    return DistLst


##########


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
    ClusterNum = len(Clusterdes)
    ClusterLst = [[] for i in range(ClusterNum)]
    SumLst = [sum(POS['AtomNum'][0:i]) for i in range(POS['EleNum'])]

    for CInd, Cluster in enumerate(Clusterdes):
        CSize = len(Cluster[0])  # B #Cluster Size

        if CSize == 1:
            for i in range(POS['AtomNum'][Cluster[0][0]]):
                GInd = i + sum(POS['AtomNum'][0:Cluster[0][0]])
                ClusterLst[CInd].append([GInd])
        else:
            IndLst = [0] * CSize
            IndLstMax = []
            GIndLst = [0] * CSize
            GrpLst = MathKit.findGrpLst(Cluster[0])
            PermuteLst = MathKit.grporderPermute(Cluster[0], GrpLst)
            DistRef = MathKit.grpSort(Cluster[1], PermuteLst)

            for Ele in Cluster[0]:
                IndLstMax.append(POS['AtomNum'][Ele] - 1)

            if -1 in IndLstMax:  # !
                continue  # !

            while (IndLst[-1] <= IndLstMax[-1]):

                for i, Ind in enumerate(IndLst):
                    GIndLst[i] = Ind + SumLst[Cluster[0][i]]

                Dist = []
                for i in range(CSize):
                    for j in range(i + 1, CSize):
                        Dist.append(POS['dismat'][GIndLst[i]][GIndLst[j]])

                flag = 1
                Dist = MathKit.grpSort(Dist, PermuteLst)
                for Dind, D in enumerate(Dist):
                    if abs(D - DistRef[Dind]) > TS:
                        flag = 0
                        break
                if flag:
                    ClusterLst[CInd].append(list(GIndLst))

                MathKit.lstGrpAdd(IndLst, IndLstMax, GrpLst)

    return ClusterLst


##########


##########

def clusterFind_forCount(Cluster_ruels, POS):
    numbClusters = len(Cluster_ruels)
    ClusterLst = [[] for i in range(numbClusters)]
    for cind, cluster in enumerate(Cluster_ruels):
        cluster_size = len(cluster[0])
        if cluster_size == 1:
            for atom in range(POS['AtomSums']):
                if POS['TypeList'][atom] == cluster[0][0]:
                    ClusterLst[cind].append([atom])
        if cluster_size == 2:
            for atom1 in range(POS['AtomSums']):
                for atom2 in range(POS['AtomSums']):
                    if cluster[0][0] == [atom1, atom2] or cluster[0][0] == [atom2, atom1]:
                        dist1 = POS['dismat'][atom1][atom2]
                        if dist1 == cluster[1][0]:
                            if [atom1, atom2] not in ClusterLst[cind] and [atom2, atom1] not in ClusterLst[cind]:
                                ClusterLst[cind].append([atom1, atom2])
        if cluster_size == 3:
            for atom1 in range(POS['AtomSums']):
                for atom2 in range(POS['AtomSums']):
                    for atom3 in range(POS['AtomSums']):
                        if cluster[0][0] == [atom1, atom2, atom3] or cluster[0][0] == [atom1, atom3, atom2] or \
                                cluster[0][0] == [atom2, atom1, atom3] or cluster[0][0] == [atom2, atom3, atom1] or \
                                cluster[0][0] == [atom3, atom1, atom2] or cluster[0][0] == [atom3, atom2, atom1]:
                            dist1 = POS['distmat'][atom1][atom2]
                            dist2 = POS['distmat'][atom1][atom3]
                            dist3 = POS['distmat'][atom2][atom3]  # unfinished


def countCluster(ClusterLst):
    ClusterCount = []
    for i in range(len(ClusterLst)):
        # B #print('ClusterLst='+str(ClusterLst[i]))
        ClusterCount.append(len(ClusterLst[i]))
    print(ClusterCount)
    return ClusterCount


##########


##########
def countCluster_Spins(ClusterLst, cluster, pos):
    ClusterCount = []
    Spins = pos['SpinList']
    for i in range(len(ClusterLst)):
        # B #print('ClusterLst='+str(ClusterLst[i]))
        count = 0
        clust = cluster[i]
        if len(clust[0]) == 2:
            if clust[2][0] == 1:
                pairs = ClusterLst[i]
                for j in range(len(pairs)):
                    pair = pairs[j]
                    # if Spins[pair[0]] == Spins[pair[1]]:
                    #     count += 2  # *np.kron(Spins[pair[0]],Spins[pair[1]])
                    count += 2*Spins[pair[0]]*Spins[pair[1]]
                ClusterCount.append(count)
            else:
                ClusterCount.append(2 * len(ClusterLst[i]))
        else:
            ClusterCount.append(len(ClusterLst[i]))
    print(ClusterCount)
    for i in range(len(ClusterCount)):
        ClusterCount[i] /= pos['AtomSum']
    return ClusterCount


##########


##########
def clusterE(ClusterLst, ClusterCoef):
    #    '''
    #    Calculate total energy
    #
    #    Args:
    #        ClusterLst: List of indentified clusters
    #        ClusterCoef: ECI of each cluster
    #    '''

    # B #ClusterCount = [];
    # B #for i in range(len(ClusterLst)):
    # B #    ClusterCount.append(len(ClusterLst[i]));
    ClusterCount = countCluster(ClusterLst);
    # B #print ClusterCount,ClusterCoef, len(ClusterCount), len(ClusterCoef);

    ECE = 0.0;
    ECE = ECE + ClusterCoef[0];
    # B #print ECE;
    for i in range(len(ClusterCount)):
        ECE = ECE + ClusterCount[i] * ClusterCoef[i + 1];
        # B #print ECE
    return ECE


##########


##########
def dismatswap(dismat, Ind1, Ind2):
    #    '''
    #    Update the distance matrix
    #
    #    Args:
    #        dismat: distance matrix
    #        Ind1, Ind2: the indexes of two atoms that swap positions
    #    '''

    lendismat = len(dismat[1])

    tmp = dismat[Ind1][:]
    dismat[Ind1][:] = dismat[Ind2][:]
    dismat[Ind2][:] = tmp

    for i in range(len(dismat[1])):
        if (i != Ind1) & (i != Ind2):
            dismat[i][Ind1] = dismat[Ind1][i]
            dismat[i][Ind2] = dismat[Ind2][i]

    tmp = dismat[Ind1][Ind2]
    dismat[Ind1][Ind2] = dismat[Ind1][Ind1]
    dismat[Ind1][Ind1] = tmp

    tmp = dismat[Ind2][Ind1]
    dismat[Ind2][Ind1] = dismat[Ind2][Ind2]
    dismat[Ind2][Ind2] = tmp

    return dismat


##########


##########
# @def clusterswap1(ClusterDes, POS, ClusterLst, Atom1, Atom2, Ind1, Ind2, TS=0.2):
def clusterswap1(ClusterDes, POS, ClusterLst, Atom1, Atom2, Ind1, Ind2, TS=0.1):
    #    '''
    #    Update the cluster information after swapping atoms
    #    This is a cleaner and more robust version of clusterswap method below
    #
    #    Args:
    #        ClusterDes: Description about clusters
    #        POS: POSCAR dictionary
    #        ClusterLst: Cluster information
    #        Atom1, Atom2: Atom sublattice
    #        Ind1, Ind2: Atom indices
    #    '''

    ClusterNum = len(ClusterLst)
    SumLst = [sum(POS['AtomNum'][0:i]) for i in range(POS['EleNum'])]

    ClusterLst_cp = copy.deepcopy(ClusterLst)  # !
    for LstInd, Lst in enumerate(ClusterLst):
        for Ind, AtomInd in enumerate(Lst):
            if (Ind1 in AtomInd) | (Ind2 in AtomInd):
                # @ClusterLst[LstInd].remove(AtomInd)
                ClusterLst_cp[LstInd].remove(AtomInd)  # !
    ClusterLst = copy.deepcopy(ClusterLst_cp)  # !

    for CInd, Cluster in enumerate(ClusterDes):
        CSize = len(Cluster[0])

        if (CSize == 1) & (Atom1 == Cluster[0][0]):
            # @ClusterLst[ClusterInd].append([Ind1])
            ClusterLst[CInd].append([Ind1])  # !

        elif (CSize == 1) & (Atom2 == Cluster[0][0]):
            # @ClusterLst[ClusterInd].append([Ind2])
            ClusterLst[CInd].append([Ind2])  # !

        else:
            for AtomI, Atom in enumerate([Atom1, Atom2]):
                if Atom in Cluster[0]:

                    AtomInd = [Ind1, Ind2][AtomI]
                    AtomLoc = Cluster[0].index(Atom)
                    IndLst = [0] * (CSize - 1)
                    IndLstMax = []
                    GIndLst = [0] * (CSize - 1)

                    GrpLst = MathKit.findGrpLst(Cluster[0])
                    PermuteLst = MathKit.grporderPermute(Cluster[0], GrpLst)
                    DistRef = MathKit.grpSort(Cluster[1], PermuteLst)

                    ClusterTmp = copy.deepcopy(Cluster[0])
                    ClusterTmp.remove(Atom)
                    GrpLst_tmp = MathKit.findGrpLst(ClusterTmp)  # !

                    for Ele in ClusterTmp:
                        IndLstMax.append(POS['AtomNum'][Ele] - 1)

                    while (IndLst[-1] <= IndLstMax[-1]):
                        for i, Ind in enumerate(IndLst):
                            # @GIndLst[i] = Ind + SumLst[Cluster[0][i]]
                            GIndLst[i] = Ind + SumLst[ClusterTmp[i]]  # !
                        GIndLst.insert(AtomLoc, AtomInd)
                        GIndLst = MathKit.grpSort(GIndLst, GrpLst)

                        Dist = []
                        for i in range(CSize):
                            for j in range(i + 1, CSize):
                                Dist.append(POS['dismat'][GIndLst[i]][GIndLst[j]])

                        flag = 1
                        Dist = MathKit.grpSort(Dist, PermuteLst)
                        for Dind, D in enumerate(Dist):
                            # @if abs (D - DistRef[ind]) > TS:
                            if abs(D - DistRef[Dind]) > TS:  # !
                                flag = 0
                                break
                        if flag:
                            # @ClusterLst[CInd].append(list(GIndLst))
                            if GIndLst not in ClusterLst[CInd]:  # !
                                ClusterLst[CInd].append(GIndLst)  # !

                        GIndLst = [0] * (CSize - 1)  # !
                        # @MathKit.lstGrpAdd(IndLst,IndLstMax,GrpLst)
                        MathKit.lstGrpAdd(IndLst, IndLstMax, GrpLst_tmp)  # !

    return ClusterLst


##########


##########
def createSupercell(POS):
    new_POS = POS.copy()
    for mat_inc in range(len(POS)):
        numb_orig = len(POS[mat_inc]['LattPnt'])
        for i in range(3):
            # new_POS[mat_inc]['Base'][i][i]*=2
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