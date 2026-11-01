import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage

data = {
    'Program': ['TKB', 'MAC', 'COBOL', 'BASIC', 'Pascal', 'EDT', 'SOS'],
    'CPU': [14, 13, 8, 6, 6, 4, 1],
    'IO': [2735, 253, 27, 27, 12, 91, 33]
}

df = pd.DataFrame(data)
X_raw = df[['CPU', 'IO']].values

Z_raw = linkage(X_raw, method='centroid', metric='euclidean')

X_std = (X_raw -np.mean(X_raw, axis=0)) / np.std(X_raw, axis=0, ddof=1)
Z_std = linkage(X_std, method='centroid', metric='euclidean')

plt.figure(figsize=(10,6))

dendrogram(
    Z_raw,
    labels=df['Program'].values,
    leaf_rotation=0,
    leaf_font_size=11,
    color_threshold=0
)

plt.title(
    'Dendrograma - Algoritmo Spanning Tree (Centroid Linkage)', 
    fontsize=14, 
    fontweight='bold', 
    pad=15
    ) 

plt.xlabel(
    'Programas da Carga de Trabalho', 
    fontsize=12, 
    labelpad=10
    )

plt.ylabel(
    'Distância Euclidiana de Fusão (d)', 
    fontsize=12
    ) 

plt.grid(
    axis='y', 
    linestyle='--', 
    alpha=0.6
    )

plt.tight_layout() 
plt.savefig('dendrograma_spanning_tree.png', dpi=300) 
plt.show()