import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.metrics import auc
from rdkit.Chem import SDMolSupplier

font = {'family' : 'normal', 'weight' : 'bold', 'size' : 16}
rc('font', **font)

def create_csvs(sdf1, sdf2):

    suppl = SDMolSupplier(sdf1)
    suppl2 = SDMolSupplier(sdf2)

    all_ligands = []
    for mol in suppl:
        name = mol.GetProp("_Name")
        all_ligands.append(name)

    actives_csv = open("actives_names.csv", "w")
    decoys_csv = open("decoys_names.csv", "w")

    print("Ligand,Score", file=actives_csv)
    print("Ligand,Score", file=decoys_csv)

    ligands_in = []
    for mol in suppl:
        name = mol.GetProp("_Name")
        if f"{sys.argv[1]}_centers_and_decoys.smi:" not in name:
            continue
        if name not in ligands_in:
            ligands_in.append(name)
            score = mol.GetProp("r_i_docking_score")
            db, num = name.split(":")
            if int(num) <= sys.argv[2]:
                print(name + "," + score, file=actives_csv)
            else:
                print(name + "," + score, file=decoys_csv)
    for ligand in all_ligands:
        if ligand not in ligands_in:
            db, num = ligand.split(":")
            if int(num) <= int(sys.argv[2]):
                print(ligand + ",1000000", file=actives_csv)
                print("Known ligand didn't dock!!!")
            else:
                print(ligand + ",1000000", file=decoys_csv)

    actives_csv.close()
    decoys_csv.close()

conf_dict = {}
plddt_dict = {}
list_files = []
msas = ['512', '128', '32', '16', '8', '4', '2']

des_kinase = sys.argv[1]
condition = sys.argv[2].lower()

for msa in msas:
    df = pd.read_csv(f"{msa}MSA_pLDDT_scores.csv")
    for idx,row in df.iterrows():
        model = row['Kinase']
        info = model.split("_")
        kinase = info[0]
        if kinase == f"{des_kinase}" and float(row['pLDDT']) >= 70:
            info = model.split("_")
            model_name = f"{int(msa)}MSA_{info[0]}_{info[1]}_{info[2]}"
            plddt_dict[model_name] = float(row['pLDDT'])
            fn = str(int(int(msa))) + f"MSA_{model}_unrelaxed_aligned_pv.sdf"
            list_files.append(fn)
    
    if condition == 'conf':
        df = pd.read_csv(f"{des_kinase}_models_classified.csv")
        for idx,row in df.iterrows():
            info = row['Input'].split("_")
            model_name = f"{int(msa)}MSA_{info[0]}_{info[1]}_{info[2]}"
            dfg = row['Spatial_label']
            chelix = row['C-helix_label']
            if dfg == "DFGin" and chelix == "Chelix-in":
                conf = "CIDI"
            elif dfg == "DFGin" and chelix == "Chelix-out":
                conf = "CODI"
            elif dfg == "DFGinter":
                conf = "DFGinter"
            elif dfg == "DFGout" and chelix == "Chelix-in":
                conf = "CIDO"
            elif dfg == "DFGout" and chelix == "Chelix-out":
                conf = "CODO"
            conf_dict[model_name] = conf

low_quart = np.linspace(0,0.1,num=100)
midlow_quart = np.linspace(0.1,1,num=100)
midhigh_quart = np.linspace(1,10,num=100)
high_quart = np.linspace(10,100,num=100)

thresholds = np.concatenate((low_quart, midlow_quart, midhigh_quart, high_quart))

aucs = []
data = []
for fn in list_files:
    info = fn.split("_")
    model_name = "{info[0]}_{info[1]}_{info[2]}_{info[3]}"

    create_csvs(f"{des_kinase}_centers_and_decoys.sdf", fn)

    ligand_scores = {}
    actives = pd.read_csv("actives_names.csv")
    decoys = pd.read_csv("decoys_names.csv")
    num_knowns = len(list(actives['Ligand']))
    for idx, row in actives.iterrows():
        ligand_scores[row['Ligand']] = row['Score']
    for idx, row in decoys.iterrows():
        ligand_scores[row['Ligand']] = row['Score']

    ligand_scores = dict(sorted(ligand_scores.items(), key=lambda item: item[1:]))
    copy_dict = ligand_scores.copy()
    for ligand in copy_dict.keys():
        db,num = ligand.split(":")
        score = ligand_scores[ligand]
        if int(num) <= num_knowns and score == 1000000:
            ligand_scores[ligand] = ligand_scores.pop(ligand)
    
    found_hits = []
    for perc in thresholds:

        top_perc = {}
        ligands = list(ligand_scores.keys())
        for i in range(round(len(ligands) * (perc / 100))):
            top_perc[ligands[i]] = ligand_scores[ligands[i]]
        count = 0
        for ligand in top_perc.keys():
            base,num = ligand.split(":")
            if int(num) <= num_knowns:
                count += 1
        found_hits.append((count / num_knowns) * 100)
    data.append((thresholds, found_hits, model_name))

# Plot True Positive Rates
AUCs = []
logAUCs = []
x = [0,100]
y = [0,100]
plt.figure(figsize=[10, 6])
plt.plot(x, y, label='Random', color='blue', linestyle='dashed')
for n in data:
    AUC = auc(np.array(n[0]), np.array(n[1])) / 100
    AUCs.append(AUC)
    print(f"AUC: {AUC}\t{n[2]}")

    ## Color by MSA Depth ##

    if condition == "msa":
        info = n[2].split("_")
        if info[0] == "512MSA":
            color = "brown"
        elif info[0] == "128MSA":
            color = "red"
        elif info[0] == "32MSA":
            color = "black"
        elif info[0] == "16MSA":
            color = "green"
        elif info[0] == "8MSA":
            color = "pink"
        elif info[0] == "4MSA":
            color = "purple"
        elif info[0] == "2MSA":
            color = "orange"

    ## Color by pLDDT ##
    elif condition == "plddt":
        plddt = plddt_dict[n[2]]
        if plddt >= 90: color="red"
        elif plddt >= 80 and plddt < 90: color="purple"
        elif plddt >= 70 and plddt < 80: color="green"

    ## Color by Conformation ##
    elif condition == "conf":
        conf = conf_dict[n[2]]
        if conf == "CIDI": color="#bf0f02"
        elif conf == "CODI": color="#9500ff"
        elif conf == "CIDO": color="#00caff"
        elif conf == "CODO": color="#045c00"
        elif conf == "DFGinter": color="#0b00ff"

    plt.plot(n[0], n[1], color=color)
plt.xlabel('Percent of Ranked Database')
plt.ylabel('Percent of Knowns Found')
plt.title(f'AF2 {des_kinase} Model Enrichment')
plt.legend()

print(f"Average AUC: {np.average(AUCs)}", "±", f"{np.std(AUCs)}")

plt.show()
