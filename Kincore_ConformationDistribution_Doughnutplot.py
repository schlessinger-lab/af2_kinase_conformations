### Author: Noah Herrington, Ph.D.
### Email: noah.herrington@mssm.edu


import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib import rcParams
from plotly import graph_objects as go

font = {'weight' : 'bold',
        'size' : 10}
rc('font', **font)
rcParams['figure.dpi'] = 300

cidi = 0
cido = 0
codi = 0
codo = 0
omega = 0
unassigned = 0

headers = ['Empty','Input','Group','Model','Chain','Spatial_label','Dihedral_label','C-helix_label','Ligand','Ligand_label','X-DFG','Φ','Ψ','DFG-Asp','Φ.1','Ψ.1','DFG-Phe','Φ.2','Ψ.2','χ1']
df = pd.read_csv("kinases_classified_noissues.csv",names=headers)

pdbs = []
for idx,row in df.iterrows():
    if row['Empty'] != "#" and row['Group'] != "Model" and row['Group'] != "failed":
        kinase = row['Input']
        if kinase not in pdbs:
            pdbs.append(kinase)
            dfg = row['Spatial_label']
            chelix = row['C-helix_label']
            if dfg == 'DFGin' and chelix == 'Chelix-in':
                cidi += 1
            elif dfg == 'DFGin' and chelix == 'Chelix-out':
                codi += 1
            elif dfg == 'DFGinter':
                omega += 1
            elif dfg == 'DFGout' and chelix == 'Chelix-in':
                cido += 1
            elif dfg == 'DFGout' and chelix == 'Chelix-out':
                codo += 1
            elif dfg == 'Unassigned':
                unassigned += 1
            else:
                print(row['Input'], "\t", row['Spatial_label'], "\t", row['C-helix_label'])

colors = ['#000140','#002CB3','#4263B7','#7280C7','#B5B0EA','#808080']

sizes = [cidi, cido, codi, codo, omega, unassigned]
cidi_frac = cidi/sum(sizes) * 100
cido_frac = cido/sum(sizes) * 100
codi_frac = codi/sum(sizes) * 100
codo_frac = codo/sum(sizes) * 100
ωCD_frac = omega/sum(sizes) * 100
unassigned_frac = unassigned/sum(sizes) * 100

cidi_frac = str(round(cidi_frac,1)) + "%"
cido_frac = str(round(cido_frac,1)) + "%"
codi_frac = str(round(codi_frac,1)) + "%"
codo_frac = str(round(codo_frac,1)) + "%"
ωCD_frac = str(round(ωCD_frac,1)) + "%"
unassigned_frac = str(round(unassigned_frac,1)) + "%"
labels = ["CIDI: {}".format(cidi_frac), "CIDO: {}".format(cido_frac),\
        "CODI: {}".format(codi_frac), "CODO: {}".format(codo_frac),\
        "DFGinter: {}".format(ωCD_frac),\
        "Unassigned: {}".format(unassigned_frac)]
bold_labels = []
for label in labels:
    bold_labels.append('<b>' + label + '</b>')

print("CIDI:",cidi,cidi/sum(sizes) * 100, "%")
print("CIDO:",cido,cido/sum(sizes) * 100, "%")
print("CODI:",codi,codi/sum(sizes) * 100, "%")
print("CODO:",codo,codo/sum(sizes) * 100, "%")
print("DFGinter:",omega,omega/sum(sizes) * 100, "%")
print("Unassigned:",unassigned,unassigned/sum(sizes) * 100, "%")

### Plotly GO Method ###

fig = go.Figure(data=[go.Pie(labels=bold_labels, values=sizes, direction='clockwise', sort=False)])
fig.update_traces(textposition='outside',textfont_size=42, marker_colors=colors, hole=0.4, textinfo='label')
fig.update_layout(annotations=[dict(text='<b>MSA<br>Depth 8</b>', x=0.51, y=0.5, font_size=60)], font=dict(color="black"))
fig.show()

### Matplotlib PyPlot Method ###

#fig, ax = plt.subplots()
#ax.pie(sizes, labels=labels, colors=colors, wedgeprops={'linewidth': 1.0, 'edgecolor': 'white'})
#ax.legend(labels,loc="upper left")

#plt.show()
#plt.savefig("Dunbrack_TotalConformationCount.png", dpi=300)
