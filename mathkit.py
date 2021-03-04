# Version 1.9


##########
def findGrpLst(Lst):
    """
    Group the lst
    """

    GrpLst = [[0]]
    Loc = 0
    NLst = len(Lst)
    for Ind in range(1, NLst):
        if Lst[Ind] == Lst[Ind - 1]:
            GrpLst[Loc].append(Ind)
        else:
            Loc += 1
            GrpLst.append([])
            GrpLst[Loc].append(Ind)
    return GrpLst


##########


##########
def grporderPermute(Lst, GrpLst):
    '''
    Find the permution of group list

    Args:
        Lst: The original list, somthing like [1,100,1000]
        GrpLst: the list of group of Lst, something like [[0,1],2]
    '''

    LstLen = len(Lst)
    GrpLen = len(GrpLst)
    PermuteGrp = []
    flag = 0

    for GrpInd1, Grp1 in enumerate(GrpLst):
        for GrpInd2 in range(GrpInd1, GrpLen):
            Grp2 = GrpLst[GrpInd2]
            PermuteGrp.append([])
            for i in Grp1:
                for j in Grp2:
                    PermuteGrp[flag].append((i, j))
            if len(PermuteGrp[flag]) == 1:  # B #Should have at least 1 item
                PermuteGrp.remove(PermuteGrp[flag])
            else:
                flag += 1

    PermuteGrpLst = [[] for i in range(len(PermuteGrp))]
    PermuteLst = [[] for i in range(len(PermuteGrp))]

    Count = 0
    for i in range(LstLen):
        for j in range(i + 1, LstLen):
            for PInd, Permute in enumerate(PermuteGrp):
                if (i, j) in Permute:
                    PermuteGrpLst[PInd].append(Count)
                    PermuteLst[PInd].append((i, j))
                    break
            Count += 1

    return PermuteGrpLst


##########


##########
def grpSort(Lst, GrpLst):
    """
    Sort the Lst with certain group constaint
    """
    Lst0 = list(Lst)
    for Grp in GrpLst:
        if len(Grp) <= 1:
            continue

        else:
            SubLst = sorted([Lst0[i] for i in Grp])
            for i, ind in enumerate(Grp):
                Lst0[ind] = SubLst[i]
    return Lst0


##########


##########
def lstGrpAdd(Lst, MaxVal, GrpLst):
    """
    Add one into the lst, this operation is used to when we iterate all the
    combinations with candidate in Lst variable. In this case, the function
    would call the next combination, the list would also be ordered in this
    case, which means [1,2,3] is treated the same as [2,3,1] and [3,2,1].

    Args:
        Lst: The Lst to be operated, should consist only integers, the minimum
             of each candidate is 0, while the maximum is MaxVal
        MaxVal: Maximum value of integers expected in each Lst element
    """

    LstLen = len(Lst)
    if (Lst[0] < MaxVal[0]):
        Lst[0] += 1

    elif (Lst[0] == MaxVal[0]):
        NotFullInd = LstLen - 1

        for i in range(1, LstLen):
            if (Lst[i] != MaxVal[i]):
                NotFullInd = i
                break
        Lst[NotFullInd] += 1

        # B #Find the belonging group of NotFullInd
        for GrpInd, Grp in enumerate(GrpLst):
            if NotFullInd in Grp:
                ChgInd = GrpInd
                break

        for i in range(NotFullInd):
            if i in GrpLst[ChgInd]:
                Lst[i] = Lst[NotFullInd]
            else:
                Lst[i] = 0
    return Lst


##########


##########
def lstOrderAdd(Lst, MaxVal):
    """
    Add one into the lst, this operation is used to when we iterate all the
    combinations with candidate in Lst variable. In this case, the function
    would call the next combination, the list would also be ordered in this
    case, which means [1,2,3] is treated the same as [2,3,1] and [3,2,1].

    Args:
        Lst: The Lst to be operated, should consist only integers, the minimum
             of each candidate is 0, while the maximum is MaxVal
        MaxVal: Maximum value of integers expected in each Lst element
    """

    LstLen = len(Lst)

    if (Lst[0] < MaxVal[0]):
        Lst[0] += 1

    elif (Lst[0] == MaxVal[0]):
        NotFullInd = LstLen - 1

        for i in range(1, LstLen):
            if (Lst[i] != MaxVal[i]):
                NotFullInd = i
                break
        Lst[NotFullInd] += 1

        for i in range(NotFullInd):
            Lst[i] = Lst[NotFullInd]

    return Lst


##########


##########
def listPermute(Lst):
    '''
    Create permution with the consideration of degree of freedom
    '''

    PermuteLst = []
    Lstlen = len(Lst)
    LstItemInd = [0] * Lstlen
    MaxItemInd = [0] * Lstlen

    for Ind, LstItem in enumerate(Lst):
        MaxItemInd[Ind] = len(LstItem) - 1

    GrpLst = findGrpLst(Lst)
    while (LstItemInd[-1] <= MaxItemInd[-1]):
        ItemLst = []
        for i, Ind in enumerate(LstItemInd):
            ItemLst.append(Lst[i][Ind])
        PermuteLst.append(ItemLst)
        LstItemInd = lstGrpAdd(LstItemInd, MaxItemInd, GrpLst)

    return PermuteLst
##########