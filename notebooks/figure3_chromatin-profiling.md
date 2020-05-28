---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.4.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
import pandas as pd
import numpy as np
import seaborn as sns

import matplotlib as mpl
from matplotlib import pyplot as plt

from scipy.stats import zscore

from collections import defaultdict, Counter

import re

import scipy.spatial as sp
import scipy.cluster.hierarchy as hc

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
```

# Summary

This notebook describes how to roughly reproduce the analysis leading up to Figure 3.

```python
chromatin_profiling = pd.read_hdf(
    "../data/CCLE_GlobalChromatinProfiling_20181130.hdf", key="chromatin_profiling")
chromatin_profiling = chromatin_profiling.dropna(axis=1, thresh=875)
chromatin_align = chromatin_profiling.fillna(chromatin_profiling.mean())
chromatin_align = chromatin_align.apply(zscore)

hs_muts = pd.read_hdf("../data/hs_muts.h5", key="hs_muts")
damaging_muts = pd.read_hdf("../data/damaging_muts.h5", key="damaging_muts")
fusions = pd.read_csv("../data/CCLE_Fusions_20181130.txt", sep="\t")

hs_muts = hs_muts.loc[chromatin_align.index]

mutation_calls = pd.read_hdf(
    "../data/depmap_19Q1_mutation_calls.h5", key="mutation_calls")
```

# Add mutations and arrangements


## NSD2 fusions

It was [previously shown](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4262138/) that fusions and mutations in NSD2 lead to specific chromatin states. Here we pull these NSD2 fusion annotations out from the full list of fusions. Note that here NSD2 is identified with its alias, WHSC1.

```python
nsd2_fusions = fusions[(fusions["LeftGene"] == "WHSC1")
                       | (fusions["RightGene"] == "WHSC1")]
nsd2_fused = set(nsd2_fusions["BroadID"])
```

## CREBBP and EP300 mutations
One of the discoveries we made in clustering the chromatin modification data was that certain clusters were enriched in mutations of certain genes. This would hint that these mutations are perhaps the cause of these abnormal chromatin states.

In particular, we found that truncating mutations in CREBBP and EP300, two well-known chromatin modifiers, are associated with one cluster marked by increased acetylation of lysine 27 and di/tri-methylation of lysine 36 of histone H3 (H3K27ac1K36me2 and H3K27ac1K36me3). We were then able to narrow down these truncating mutations to those in the TAZ (transcription adaptor putative zinc finger) domains of these two proteins.

```python
crebbp = mutation_calls[mutation_calls["Hugo_Symbol"] == "CREBBP"]
crebbp = crebbp[crebbp["Variant_annotation"] == "damaging"]
crebbp = crebbp.dropna(subset=["Protein_Change"])

crebbp["codon_n"] = crebbp["Protein_Change"].apply(
    lambda x: int(re.search("[0-9]+", str(x))[0]))
crebbp = crebbp[(crebbp["codon_n"] >= 1745) & (
    crebbp["codon_n"] <= 1846)]  # TAZ2 domain
crebbp = set(crebbp["DepMap_ID"])

ep300 = mutation_calls[mutation_calls["Hugo_Symbol"] == "EP300"]
ep300 = ep300[ep300["Variant_annotation"] == "damaging"]
ep300 = ep300.dropna(subset=["Protein_Change"])

ep300["codon_n"] = ep300["Protein_Change"].apply(
    lambda x: int(re.search("[0-9]+", str(x))[0]))
ep300 = ep300[(ep300["codon_n"] >= 1708) & (
    ep300["codon_n"] <= 1809)]  # the TAZ2 domain
ep300 = set(ep300["DepMap_ID"])
```

## Aggregate alterations


Here we combine NSD2 fusions, CREBBP and EP300 mutations into a single dataframe for plotting. We also add in EZH2 mutations, which were [previously](https://www.sciencedirect.com/science/article/pii/S1046202314003600?via%3Dihub) shown to be associated with chromatin states.

```python
mut_df = pd.DataFrame(index=chromatin_profiling.index)

mut_df["EZH2"] = hs_muts["EZH2"]
mut_df["NSD2"] = mut_df.index.map(lambda x: x in nsd2_fused)
mut_df["CREBBP"] = mut_df.index.map(lambda x: x in crebbp)
mut_df["EP300"] = mut_df.index.map(lambda x: x in ep300)

mut_df = mut_df.astype(int)
```

# Clustered heatmap


To plot chromatin states in conjunction with mutations, we first cluster the chromatin states such that each cluster contains cell lines with similar chromatin marks. We can then plot these clusters as a grouped heatmap, and on top we can add columns indicating whether or not each cell line contains a mutation.

```python
def get_colors(s, cmap):

    pal = sns.color_palette(cmap, len(s.unique()))
    mapping = dict(zip(s.unique(), pal))
    colors = pd.Series(s).map(mapping)

    return colors, mapping
```

```python
n_clusters = 24

cell_line_linkage = hc.linkage(chromatin_align, method='ward')

clusters = hc.fcluster(cell_line_linkage, n_clusters, "maxclust")
clusters = pd.Series(clusters, index=chromatin_align.index)

cluster_colors = get_colors(clusters, "tab20")[0]
```

```python
cluster_splits = [chromatin_align.loc[clusters == x]
                  for x in range(1, n_clusters+1)]
cluster_muts = [mut_df.loc[clusters == x] for x in range(1, n_clusters+1)]

lengths = [len(x) for x in cluster_splits]

total_lines = sum(lengths)

cumulative_lengths = [0]+list(np.cumsum(lengths))
```

```python
fig = plt.figure(figsize=(16, 10))

spacing = 8

gs = mpl.gridspec.GridSpec(8, total_lines+spacing*(n_clusters-1))

for cluster_idx, cluster in enumerate(cluster_splits):

    muts = cluster_muts[cluster_idx]
    indent = spacing*cluster_idx

    ax = fig.add_subplot(gs[2:, cumulative_lengths[cluster_idx] +
                            indent:cumulative_lengths[cluster_idx+1]+indent])

    ax.imshow(cluster.T, aspect="auto", vmin=-8, vmax=8, cmap="bwr")
    plt.box(False)

    if cluster_idx == n_clusters-1:
        ax.set_xticklabels([])
        ax.set_yticks(list(range(len(chromatin_align.columns))))
        ax.set_yticklabels(chromatin_align.columns)

        ax.xaxis.set_ticks_position('none')

        ax.yaxis.tick_right()
        ax.yaxis.set_ticks_position('none')

    else:
        plt.axis('off')

    ax = fig.add_subplot(gs[1, cumulative_lengths[cluster_idx] +
                            indent:cumulative_lengths[cluster_idx+1]+indent])

    ax.imshow(muts.T, aspect="auto",
              cmap=mpl.colors.ListedColormap(["#f6f6f6", "black"]))
    plt.box(False)

    if cluster_idx == n_clusters-1:
        ax.set_xticklabels([])
        ax.set_yticks(list(range(4)))
        ax.set_yticklabels(mut_df.columns)

        ax.xaxis.set_ticks_position('none')

        ax.yaxis.tick_right()
        ax.yaxis.set_ticks_position('none')

    else:
        plt.axis('off')

plt.savefig("../plots/figure3.pdf", dpi=512,
            bbox_inches="tight", background="transparent")
```
