### Author: Noah Herrington, Ph.D.
### Email: noah.herrington@mssm.edu


import os
import warnings
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, rc, rcParams

def norm(conf_list):
    norm_conf_list = []
    minimum = min(conf_list)
    maximum = max(conf_list)
    rang_e = maximum - minimum
    for item in conf_list:
        item_norm = (item - minimum) / rang_e
        norm_conf_list.append(item_norm)
    return norm_conf_list

def avg(conf_list):
    avg_conf_list = []
    sum_all = sum(conf_list)
    for item in conf_list:
        item_frac = item / sum_all
        avg_conf_list.append(item_frac)
    return avg_conf_list

warnings.filterwarnings("ignore")

font = {'weight' : 'bold',
        'size' : 11}
rc('font', **font)
rcParams['figure.dpi'] = 300

kinase_families = {}
family_names = {}
families_df = pd.read_csv(os.getcwd() + "/kinfam.csv")
for idx,row in families_df.iterrows():
    new_family = row['Group']
    if new_family not in family_names.keys():
        family_names[new_family] = {}
        family_names[new_family]["CIDI"] = 0
        family_names[new_family]["CIDO"] = 0
        family_names[new_family]["CODI"] = 0
        family_names[new_family]["CODO"] = 0
        family_names[new_family]["DFGinter"] = 0
        family_names[new_family]["Unassigned"] = 0
    kinase_families[row['HGNC Name']] = row['Group']
kinase_families['GRK2'] = "AGC"
kinase_families['GRK3'] = "AGC"
kinase_families['PDPK2P'] = "AGC"
kinase_families['RSKR'] = "AGC"
kinase_families['SIK1B'] = "CAMK"
kinase_families['CDK11A'] = "CMGC"
kinase_families['CSNK2A3'] = "CMGC"
kinase_families['PAK5'] = "STE"
kinase_families['MAP3K20'] = "STE"
kinase_families['MAP3K21'] = "STE"

classified_df = pd.read_csv(os.getcwd() + "/kinases_classified.csv")
used_uniprots = []
for idx,row in classified_df.iterrows():
    filename = row['Input']
    dfg = row['Spatial_label']
    chelix = row['C-helix_label']
    kinase,rank,rank_num,model,model_num,ptm,seed,seed_num,unrelax = filename.split("_")
    try:
        kinase,domain = kinase.split("-")
    except:
        pass
    else:
        pass
    family = kinase_families[kinase]
    if dfg == 'DFGin' and chelix == 'Chelix-in':
        family_names[family]['CIDI'] += 1
    elif dfg == 'DFGin' and chelix == 'Chelix-out':
        family_names[family]['CODI'] += 1
    elif dfg == 'DFGout' and chelix == 'Chelix-in':
        family_names[family]['CIDO'] += 1
    elif dfg == 'DFGout' and chelix == 'Chelix-out':
        family_names[family]['CODO'] += 1
    elif dfg == 'DFGinter':
        family_names[family]['DFGinter'] += 1
    elif dfg == 'Unassigned':
        family_names[family]['Unassigned'] += 1
family_names = dict(sorted(family_names.items()))
del family_names['Atypical']
del family_names['RGC']
families = list(family_names.keys())
conformations = {}
conformations["CIDI"] = []
conformations["CIDO"] = []
conformations["CODI"] = []
conformations["CODO"] = []
conformations["DFGinter"] = []
conformations["Unassigned"] = []

print("AF2 Structure Count")
print("-------------------")
list_of_lists = []
for family in family_names.keys():
    family_conf_list = []
    family_conf_list.append(family_names[family]["CIDI"])
    family_conf_list.append(family_names[family]["CIDO"])
    family_conf_list.append(family_names[family]["CODI"])
    family_conf_list.append(family_names[family]["CODO"])
    family_conf_list.append(family_names[family]["DFGinter"])
    family_conf_list.append(family_names[family]["Unassigned"])
    avg_family_conf_list = avg(family_conf_list)
    list_of_lists.append(avg_family_conf_list)
    sum_structures = 0
    sum_structures += family_names[family]["CIDI"]
    sum_structures += family_names[family]["CIDO"]
    sum_structures += family_names[family]["CODI"]
    sum_structures += family_names[family]["CODO"]
    sum_structures += family_names[family]["DFGinter"]
    sum_structures += family_names[family]["Unassigned"]
    print("{}: {}".format(family, sum_structures))
    print("CIDI Fraction: {}".format(family_names[family]["CIDI"] / sum_structures * 100, "%"))
    print("CIDO Fraction: {}".format(family_names[family]["CIDO"] / sum_structures * 100, "%"))
    print("CODI Fraction: {}".format(family_names[family]["CODI"] / sum_structures * 100, "%"))
    print("CODO Fraction: {}".format(family_names[family]["CODO"] / sum_structures * 100, "%"))
    print("DFGinter Fraction: {}".format(family_names[family]["DFGinter"] / sum_structures * 100, "%"))
    print("Unassigned Fraction: {}".format(family_names[family]["Unassigned"] / sum_structures * 100, "%"))

confs = ["CIDI","CIDO","CODI","CODO","DFGinter","Unassigned"]
for i in range(6):
    for j in range(8):
        conf = confs[i]
        conformations[conf].append(list_of_lists[j][i])

fig,ax = plt.subplots()
bottom = np.zeros(8)
colors = ['#000140','#002CB3','#4263B7','#7280C7','#B5B0EA','#808080']

for conf,conf_count,color in zip(conformations.keys(),conformations.values(), colors):
    p = ax.bar(families, conf_count, color=color, label=conf, bottom=bottom)
    bottom += conf_count

plt.xlabel("Group", fontsize=13)
plt.ylabel("Fraction of Structures", fontsize=13)
plt.ylim(0,1.1)
plt.show()
