# Version 1.9

import numpy as np
import math


##########
def posreader(PosName='POSCAR'):
    """
    Read the atomic configuration from POSCAR

    Args:
        PosName (str): the name of the POSCAR File, (default: 'POSCAR')
    """
    POS = {}  # B #Initialize the dictionary for POSCAR information
    Fid = open(PosName, 'r')

    Line = Fid.readline()
    POS['CellName'] = Line.split('\n')[0]  # B #Comment line

    Line = Fid.readline()
    Sline = Line.split()
    POS['LattConst'] = float(Sline[0])  # B #Lattice constant

    POS['Base'] = [[0.0] * 3 for i in range(3)]  # B #Initilize the base list
    for i in range(3):
        Line = Fid.readline()
        Sline = Line.split()
        # @POS['Base'][i] = [float(Sline[i]) for i in range(3)];
        POS['Base'][i] = [float(Sline[i]) * POS['LattConst'] for i in range(3)]  # !

    Line = Fid.readline()
    Sline = Line.split()
    POS['EleName'] = Sline  # B #The name of each element
    POS['EleNum'] = len(POS['EleName'])  # B #number of elements involved

    Line = Fid.readline()
    Sline = Line.split()
    POS['AtomNum'] = [0] * POS['EleNum']
    POS['AtomSum'] = 0
    for ind, Num in enumerate(Sline):
        POS['AtomNum'][ind] = int(Num)
        POS['AtomSum'] += int(Num)

    Line = Fid.readline()
    Sline = Line.split()
    FL = Sline[0][0]  # B #Check the first letter
    if (FL == 'S'):
        POS['IsSel'] = 1
        POS['SelMat'] = [['X'] * 3 for i in range(POS['AtomSum'])]
        Line = Fid.readline()
        Sline = Line.split()
        FL = Sline[0][0]  # B #Check the first letter for coord
    else:
        POS['IsSel'] = 0

    # B # Set up the lattice type
    if (FL == 'D') | (FL == 'd'):
        POS['LatType'] = 'Direct'
    elif (FL == 'C') | (FL == 'c'):
        POS['LatType'] = 'Cartesian'
    else:
        print("Please check the POSCAR file, the lattice type is not direct or cartesian")

    POS['LattPnt'] = [[0.0] * 3 for i in range(POS['AtomSum'])]  # B #Initialize lattice points

    if (POS['LatType'] == 'Direct'):  # !
        for i in range(POS['AtomSum']):  # !
            Line = Fid.readline()  # !
            Sline = Line.split()  # !
            POS['LattPnt'][i] = [float(Sline[i]) for i in range(3)]  # !
            if (POS['IsSel']):  # !
                POS['SelMat'][i] = [Sline[i + 3] for i in range(3)]  # !

    elif (POS['LatType'] == 'Cartesian'):  # !
        BaseInv = np.linalg.inv(POS['Base'])  # !
        for i in range(POS['AtomSum']):  # !
            Line = Fid.readline()  # !
            Sline = Line.split()  # !
            POS['LattPnt'][i] = [float(Sline[i]) for i in range(3)]  # !
            POS['LattPnt'][i] = list(np.dot(BaseInv, POS['LattPnt'][i]))  # !
            if (POS['IsSel']):  # !
                POS['SelMat'][i] = [Sline[i + 3] for i in range(3)]  # !

    else:  # !
        print("Please check the POSCAR file, the lattice type is not direct or cartesian")  # !

    # @    for i in range(POS['AtomSum']):
    # @        Line = Fid.readline()
    # @        Sline = Line.split()
    # @        POS['LattPnt'][i] = [float(Sline[i]) for i in range(3)]
    # @        if(POS['IsSel']):
    # @            POS['SelMat'][i] = [Sline[i+3] for i in range(3)]

    Fid.close()
    # B #The current version does not support reading the POSCAR with velocity information!!!!!!!!!!!!!!!!
    return POS


##########


##########
def poswriter(PosName, POS):
    """
    Write out the POS into a POSCAR file

    Args:
        PosName: the name of the POSCAR file
        POS: the POS dictionary
    """
    Fid = open(PosName, 'w')
    Fid.write('%s ' % POS['CellName'])
    Fid.write('\n')

    Fid.write('%f \n' % POS['LattConst'])
    for i in range(3):
        Fid.write('%f %f %f \n' % (POS['Base'][i][0], POS['Base'][i][1], POS['Base'][i][2]))

    for i in range(POS['EleNum']):
        Fid.write('%s ' % POS['EleName'][i])
    Fid.write('\n')

    for i in range(POS['EleNum']):
        Fid.write('%i ' % POS['AtomNum'][i])
    Fid.write('\n')

    if (POS['IsSel']):
        Fid.write('Selective Dynamics \n')

    Fid.write('%s \n' % POS['LatType'])
    for i in range(POS['AtomSum']):
        Fid.write('%f %f %f ' % (POS['LattPnt'][i][0], POS['LattPnt'][i][1], POS['LattPnt'][i][2]))
        if (POS['IsSel']):
            Fid.write('%s %s %s ' % (POS['SelMat'][i][0], POS['SelMat'][i][1], POS['SelMat'][i][2]))
        Fid.write('\n')

    Fid.close()


##########


##########
def dismatcreate(POS):
    #    """
    #    Create the distance matrix for a given POS
    #
    #    Args:
    #        POS: the POS dictionary
    #    """
    POS['dismat'] = [[0.0] * POS['AtomSum'] for i in range(POS['AtomSum'])]

    for AtomInd1, Pnt1 in enumerate(POS['LattPnt']):
        for AtomInd2, Pnt2 in enumerate(POS['LattPnt']):
            Pnt1 = np.array(Pnt1)
            Pnt2 = np.array(Pnt2)
            PntDis = Pnt1 - Pnt2

            for i in range(3):
                if (PntDis[i] > POS['Base'][i][i]):
                    PntDis[i] = 2 * POS['Base'][i][i] - PntDis[i]
                elif (PntDis[i] < -POS['Base'][i][i]):
                    PntDis[i] = PntDis[i] + 2 * POS['Base'][i][i]
                elif (PntDis[i] >= -POS['Base'][i][i]) & (PntDis[i] <= POS['Base'][i][i]):
                    PntDis[i] = abs(PntDis[i])
                else:
                    print("Something is wrong when calculating dist matrix")

            # PntDis = np.dot(PntDis, POS['Base'])
            POS['dismat'][AtomInd1][AtomInd2] = math.sqrt(PntDis[0] ** 2 + PntDis[1] ** 2 + PntDis[2] ** 2)

    return POS
##########