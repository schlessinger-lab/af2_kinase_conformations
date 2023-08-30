import os
import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from Bio.PDB.PDBParser import PDBParser as parser
from pymol import cmd

warnings.filterwarnings("ignore")

font = {'weight' : 'bold',
        'size' : 10}
rc('font', **font)
rcParams['figure.dpi'] = 300

def dihedral(p):
    p0 = p[0]
    p1 = p[1]
    p2 = p[2]
    p3 = p[3]

    b0 = -1.0*(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2

    b0xb1 = np.cross(b0, b1)
    b1xb2 = np.cross(b2, b1)

    b0xb1_x_b1xb2 = np.cross(b0xb1, b1xb2)

    y = np.dot(b0xb1_x_b1xb2, b1)*(1.0/np.linalg.norm(b1))
    x = np.dot(b0xb1, b1xb2)

    return np.degrees(np.arctan2(y, x))

msa = input("Which MSA?: ")

cidi_dfg_1D = []
cidi_FG = []
cidi_rmsds = []
cido_dfg_1D = []
cido_FG = []
cido_rmsds = []
codi_dfg_1D = []
codi_FG = []
codi_rmsds = []
codo_dfg_1D = []
codo_FG = []
codo_rmsds = []
dfginter_dfg_1D = []
dfginter_FG = []
dfginter_rmsds = []
unassigned_dfg_1D = []
unassigned_FG= []
unassigned_rmsds = []

models_path = os.getcwd()+"/{}MSA_5models_1recycle_1sample/pdbs/".format(msa)
class_df = pd.read_csv("{}MSA_5models_1recycle_1sample/pdbs/kinases_classified_noissues.csv".format(msa))
pdb1 = cmd.load("1ATP_cAMP-dep_prot_kinase_ATP_DFGin_Reference.pdb","pdb1")

print(class_df)
for idx1,row1 in class_df.iterrows():
    inp = row1['Input']
    model,ext = inp.split(".")

    ###Get Model RMSD###
    pdb2 = cmd.load(models_path + inp,"pdb2")
    align_list = cmd.align("pdb2","pdb1")
    rmsd = align_list[0]
    print("RMSD:", rmsd)
    cmd.remove("pdb2")

    ###Get Model Dihedrals Conformation###
    asp = row1['DFG-Asp']
    asp_num = int(asp[:3])
    phe = row1['DFG-Phe']
    phe_num = int(phe[:3])
    dfg = row1['Spatial_label']
    chelix = row1['C-helix_label']

    structure = parser().get_structure('Model', models_path + inp)
    model_in_struct = structure[0]
    chain = model_in_struct['A']
    dfgm1 = chain[asp_num-1]['CA'].coord
    dfgm2 = chain[asp_num-2]['CA'].coord
    dfg_d = chain[asp_num]['CA'].coord
    dfg_f = chain[phe_num]['CA'].coord
    dfg_g = chain[phe_num+1]['CA'].coord
    dfgp1 = chain[phe_num+2]['CA'].coord
    p1 = np.array([dfgm2,dfgm1,dfg_d,dfg_f])
    p2 = np.array([dfg_d,dfg_f,dfg_g,dfgp1])
    dfg_1D_dihedral = dihedral(p1)
    FG_dihedral = dihedral(p2)
    if dfg_1D_dihedral < 0:
        dfg_1D_dihedral += 360
    if FG_dihedral < 0:
        FG_dihedral += 360

    if dfg == "DFGin":
        if chelix == "Chelix-in":
            cidi_dfg_1D.append(dfg_1D_dihedral)
            cidi_FG.append(FG_dihedral)
            cidi_rmsds.append(rmsd)
        elif chelix == "Chelix-out":
            codi_dfg_1D.append(dfg_1D_dihedral)
            codi_FG.append(FG_dihedral)
            codi_rmsds.append(rmsd)
        else:
            print("{} is {}.".format(model,chelix))
    elif dfg == "DFGout":
        if chelix == "Chelix-in":
            cido_dfg_1D.append(dfg_1D_dihedral)
            cido_FG.append(FG_dihedral)
            cido_rmsds.append(rmsd)
        elif chelix == "Chelix-out":
            codo_dfg_1D.append(dfg_1D_dihedral)
            codo_FG.append(FG_dihedral)
            codo_rmsds.append(rmsd)
    elif dfg == "DFGinter":
        dfginter_dfg_1D.append(dfg_1D_dihedral)
        dfginter_FG.append(FG_dihedral)
        dfginter_rmsds.append(rmsd)
    elif dfg == "Unassigned":
        unassigned_dfg_1D.append(dfg_1D_dihedral)
        unassigned_FG.append(FG_dihedral)
        unassigned_rmsds.append(rmsd)
    else:
        print("{} is {}.".format(model,dfg))

    print("Model {} is {} | Dihedrals are {} and {}.".format(model,dfg,dfg_1D_dihedral,FG_dihedral))
    print()

proj = input("Which projection? (2d/3d): ")
fig = plt.figure()

if proj == '3d':
    plt.tight_layout()
    fig.subplots_adjust(right=0.6)
    ax = fig.add_subplot(projection='3d')
    ax.scatter(unassigned_FG,unassigned_dfg_1D,unassigned_rmsds,color='#828d8b',label="Unassigned")
    ax.scatter(cidi_FG,cidi_dfg_1D,cidi_rmsds,color='#bf0f02',label="CIDI")
    ax.scatter(cido_FG,cido_dfg_1D,cido_rmsds,color='#00caff',label="CIDO")
    ax.scatter(codi_FG,codi_dfg_1D,codi_rmsds,color='#9500ff',label="CODI")
    ax.scatter(codo_FG,codo_dfg_1D,codo_rmsds,color='#045c00',label="CODO")
    ax.scatter(dfginter_FG,dfginter_dfg_1D,dfginter_rmsds,color='#0b00ff',label="DFGinter")
    ax.set_xlabel('ξ(DFG-Phe...DFG-Gly) (°)')
    ax.set_ylabel('ξ(DFG-1...DFG-Asp) (°)')
    ax.set_zlabel('RMSD (Å)')
    ax.legend(bbox_to_anchor=(1.15, 0.5),loc='center left',fancybox=True,shadow=True)
    plt.savefig('{}MSA_all_classified_models_MobitzDihedrals_RMSD.png'.format(msa))

elif proj == '2d':
    font = {'weight' : 'bold',
        'size' : 14}
    rc('font', **font)
    rcParams['figure.dpi'] = 300
    ax = fig.add_subplot()
    ax.scatter(unassigned_FG,unassigned_dfg_1D,color='#828d8b',label="Unassigned")
    ax.scatter(cidi_FG,cidi_dfg_1D,color='#bf0f02',label="CIDI")
    ax.scatter(cido_FG,cido_dfg_1D,color='#00caff',label="CIDO")
    ax.scatter(codi_FG,codi_dfg_1D,color='#9500ff',label="CODI")
    ax.scatter(codo_FG,codo_dfg_1D,color='#045c00',label="CODO")
    ax.scatter(dfginter_FG,dfginter_dfg_1D,color='#0b00ff',label="DFGinter")
    ax.set_xlabel('ξ(DFG-Phe...DFG-Gly) (°)')
    ax.set_ylabel('ξ(DFG-1...DFG-Asp) (°)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5,1.2), ncol=3,fancybox=True,shadow=True)
    plt.savefig('{}MSA_all_classified_models_MobitzDihedrals_2D.png'.format(msa))
