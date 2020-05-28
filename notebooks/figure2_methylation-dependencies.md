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

from scipy.stats import zscore, pearsonr
from statsmodels.stats.multitest import multipletests

from collections import defaultdict, Counter

import scipy.spatial as sp
import scipy.cluster.hierarchy as hc

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from tqdm.auto import tqdm
tqdm.pandas()
```

# Summary

This notebook describes how to roughly reproduce the analysis and results of Figure 2 and some associated supplemental figures, which center around the addition of genome-wide methylation data.


# Load annotations


## Characterization sets

```python
tss1kb_meth = pd.read_hdf("../data/CCLE_RRBS_TSS1kb_20181022.hdf",key="tss1kb_meth")
tssclust_meth = pd.read_hdf("../data/CCLE_RRBS_tss_CpG_clusters_20181022.hdf",key="tssclust_meth")
ccle_genex = pd.read_hdf("../data/CCLE_RNAseq_rsem_genes_tpm_20180929.hdf",key="ccle_genex")

avana = pd.read_hdf("../data/Achilles_gene_effect.hdf",key="avana")
drive = pd.read_hdf("../data/D2_DRIVE_gene_dep_scores.hdf",key="drive")
achilles = pd.read_hdf("../data/D2_Achilles_gene_dep_scores.hdf",key="achilles")

avana.columns = ["_".join(x.split("_")[:-1]) for x in avana.columns]
drive.columns = ["_".join(x.split("_")[:-1]) for x in drive.columns]
achilles.columns = ["_".join(x.split("_")[:-1]) for x in achilles.columns]
```

## Cell line subtypes

```python
cell_line_annotations = pd.read_excel("../data/41586_2019_1186_MOESM4_ESM.xlsx",
                                      sheet_name="Cell Line Annotations")


subtypes = cell_line_annotations[["depMapID","type_refined"]].set_index("depMapID")
subtypes = subtypes["type_refined"]
subtypes = subtypes.dropna()
subtypes = subtypes.apply(lambda x: x.capitalize().replace("_"," ")) # preformatting

# rename subtypes to display
rename_map = {"T-cell lymphoma other":"Other T-cell lymphoma",
              "Aml":"AML",
              "Ewings sarcoma": "Ewing's sarcoma",
              "Fibroblast like":"Fibroblast-like",
              "Lung nsc":"Lunc, NSC",
              "Lymphoma hodgkin":"Hodgkin's lymphoma",
              "Lymphoma dlbcl":"DLBCL",
              "T-cell all":"T-cell ALL",
              "B-cell all":"B-cell ALL",
              "Cml":"CML",
              "B-cell lymphoma other":"Other B-cell lymphoma",
              "Leukemia other":"Other leukemia",
              "Lymphoma burkitt":"Burkitt's lymphoma"
             }

subtypes = subtypes.apply(lambda x:rename_map.get(x,x))
```

# Methylation and mRNA expression


## Match methylation loci and genes

```python
meth_genes = pd.DataFrame(index=tssclust_meth.columns)
meth_genes["gene_name"] = meth_genes.index.map(lambda x: x.split("_")[0])

genex_genes = pd.DataFrame(index=ccle_genex.columns)
genex_genes["gene_name"] = genex_genes.index.map(lambda x: "_".join(x.split("_")[:-1]))
genex_genes["ensembl_id_v"] = genex_genes.index.map(lambda x: x.split("_")[-1])
genex_genes["ensembl_id"] = genex_genes["ensembl_id_v"].apply(lambda x: x.split(".")[0])
```

## Compute methylation-expression correlations

```python
genex_gene_map = dict(zip(genex_genes["gene_name"], genex_genes.index))

meth_matched = meth_genes.copy()

meth_matched["genex_id"] = meth_matched["gene_name"].apply(lambda x: genex_gene_map.get(x, ""))

meth_matched = meth_matched[meth_matched["genex_id"]!=""]

def meth_genex_correlate(row):
    meth_name = row.name
    genex_name = row["genex_id"]
    
    meth = tssclust_meth[meth_name].dropna()
    genex = ccle_genex[genex_name].dropna()
    
    meth, genex = meth.align(genex, axis=0, join="inner")
    
    r, pval = pearsonr(meth, genex)
    
    row["corr"] = r
    row["pval"] = pval
    row["n"] = len(meth)
    
    return row

meth_matched = meth_matched.progress_apply(meth_genex_correlate, axis=1)
meth_matched = meth_matched.dropna()
```

## Plot associations

```python
ax = plt.subplot(111)

sns.distplot(meth_matched["corr"],kde=False)
plt.xlabel("Methylation vs. expression correlation")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.ylabel("Frequency")
plt.xlim(-1,1)

plt.axvline(0,linestyle="--",color="black")
```

## Select negative associations

```python
high_negative = meth_matched[meth_matched["corr"]<-0.5].copy(deep=True)
high_negative_genes = set(high_negative["gene_name"])
```

# Methylation and dependency


## Load STRING relationships

```python
string = pd.read_csv("../data/9606.protein.links.v11.0.txt.gz",sep=" ")
string = string[string["combined_score"]>=800]

string["protein1_ensembl"] = string["protein1"].apply(lambda x: x[5:])
string["protein2_ensembl"] = string["protein2"].apply(lambda x: x[5:])

ensembl_gene_protein = pd.read_csv("../data/ensembl_gene_protein.txt",sep="\t")
ensembl_gene_protein = ensembl_gene_protein.dropna(subset=["Protein stable ID","NCBI gene ID"])
ensembl_gene_protein["NCBI gene ID"] = ensembl_gene_protein["NCBI gene ID"].astype(int).astype(str)

protein_gene_map = dict(zip(ensembl_gene_protein["Protein stable ID"],ensembl_gene_protein["Gene stable ID"]))

string["gene1_ensembl"]  = string["protein1_ensembl"].apply(lambda x: protein_gene_map.get(x,np.nan))
string["gene2_ensembl"]  = string["protein2_ensembl"].apply(lambda x: protein_gene_map.get(x,np.nan))

string = string.dropna()
```

## Add ENSEMBL gene names

```python
ensembl_name_map = dict(zip(genex_genes["ensembl_id"],genex_genes["gene_name"]))
entrez_ensembl_map = dict(zip(ensembl_gene_protein["NCBI gene ID"],ensembl_gene_protein["Gene stable ID"]))

string["gene1_name"] = string["gene1_ensembl"].apply(lambda x: ensembl_name_map.get(x,""))
string["gene2_name"] = string["gene2_ensembl"].apply(lambda x: ensembl_name_map.get(x,""))
string = string[(string["gene1_name"]!="")&(string["gene2_name"]!="")]
```

## Select genes with methylation-dependency correlation

```python
# since string contains pairs in both directions, only need to select for one
gene1_valid = string["gene1_name"].isin(high_negative_genes)

string_select = string[gene1_valid].copy(deep=True)

high_negative["locus"] = list(high_negative.index)
genes_meth_map = high_negative.groupby("gene_name")["locus"].apply(list)
genes_meth_map = dict(zip(genes_meth_map.index, genes_meth_map))

string_select["gene1_loci"] = string_select["gene1_name"].apply(lambda x: genes_meth_map[x])
string_select = string_select.explode("gene1_loci")
```

```python
avana_genes = set(avana.columns)
drive_genes = set(drive.columns)
achilles_genes = set(achilles.columns)

string_select["in_avana"] = string_select["gene2_name"].apply(lambda x: x in avana_genes)
string_select["in_drive"] = string_select["gene2_name"].apply(lambda x: x in drive_genes)
string_select["in_achilles"] = string_select["gene2_name"].apply(lambda x: x in achilles_genes)

string_select_avana = string_select[string_select["in_avana"]]
string_select_drive = string_select[string_select["in_drive"]]
string_select_achilles = string_select[string_select["in_achilles"]]
```

```python
def meth_dependency_correlate(row, dependency_set):
    meth_name = row["gene1_loci"]
    dependency_name = row["gene2_name"]
    
    meth = tssclust_meth[meth_name].dropna()
    dependency = dependency_set[dependency_name].dropna()
    
    meth, dependency = meth.align(dependency, axis=0, join="inner")
    
    r, pval = pearsonr(meth, dependency)
    
    row["corr"] = r
    row["pval"] = pval
    row["n"] = len(meth)
    
    return row

string_select_avana = string_select_avana.progress_apply(meth_dependency_correlate, 
                                                         dependency_set=avana, axis=1)

string_select_drive = string_select_drive.progress_apply(meth_dependency_correlate, 
                                                         dependency_set=drive, axis=1)

string_select_achilles = string_select_achilles.progress_apply(meth_dependency_correlate, 
                                                         dependency_set=achilles, axis=1)
```

```python
def process_string_correlates(string_corrs):
    string_corrs["qval"] = multipletests(string_corrs["pval"], alpha=0.01, method="fdr_bh")[1]
    string_corrs = string_corrs.reset_index()
    
    string_corrs["corr_id"] = string_corrs["gene1_name"]+"-"+string_corrs["gene2_name"]
    string_corrs = string_corrs.sort_values(by="pval")
    string_corrs = string_corrs.drop_duplicates(subset=["corr_id"],keep="first")
    
    string_corrs["pval_rank"] = string_corrs["pval"].rank()
    
    return string_corrs
    
string_select_avana = process_string_correlates(string_select_avana)
string_select_drive = process_string_correlates(string_select_drive)
string_select_achilles = process_string_correlates(string_select_achilles)
```

```python
def plot_meth_dependency(corr_set, ax):

    ax.scatter(corr_set["corr"],
               -np.log10(corr_set["qval"]),
               c="lightgray",
               rasterized=True
               )

    select_labels = corr_set[corr_set["pval_rank"] <= 10]
    for row_name in list(select_labels.index):
        row = select_labels.loc[row_name]

        ax.text(row["corr"]+0.15, -np.log10(row["qval"])+2,
                row["gene1_name"]+" ",
                ha="right",
                color="#3f72af")
        ax.text(row["corr"]+0.15, -np.log10(row["qval"])+2,
                "/",
                ha="center",
                color="grey")
        ax.text(row["corr"]+0.15, -np.log10(row["qval"])+2,
                " "+row["gene2_name"],
                ha="left",
                color="#e23e57")

    ax.set_xlim(-1, 1)

    ax.set_xticks([-1, -0.5, 0, 0.5, 1])

    ymax = plt.ylim()[1]

    ax.text(1, ymax/30,
            "Gene X methylation",
            ha="right",
            color="#3f72af")
    ax.text(1, ymax/30,
            "  /",
            ha="center",
            color="grey")
    ax.text(1, ymax/30+ymax/30,
            "\nGene Y dependency",
            ha="right",
            va="top",
            color="#e23e57")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("$-log_{10}(q-value)$")
```

## Volcano plots

```python
plt.figure(figsize=(16,4))

ax = plt.subplot(131)
plot_meth_dependency(string_select_avana,ax)
plt.title("Avana")

ax = plt.subplot(132)
plot_meth_dependency(string_select_drive,ax)
plt.title("DRIVE")

ax = plt.subplot(133)
plot_meth_dependency(string_select_achilles,ax)
plt.title("Achilles")

plt.savefig("../plots/figure2-a.pdf",bbox_inches="tight",dpi=512,background="transparent")
```

# Plots for gene relationships 

```python
def plot_methylation(meth_x, y, subtype_map):
    
    plt.figure(figsize=(3, 4))
    ax = plt.subplot(111)

    meth_x = meth_x.dropna().rename("meth")
    y = y.rename("y")

    info = pd.concat([meth_x,y,subtypes.rename("subtype")],axis=1,join="inner")
    highlight_subtypes = list([x for x in subtype_map.keys() if x != "Other"])

    info["highlight_subtype"] = info["subtype"].apply(
        lambda x: x if x in highlight_subtypes else "Other")

    sns.scatterplot(info["meth"],
                    info["y"],
                    hue=info["highlight_subtype"].rename(""),
                    palette=subtype_map,
                    linewidth=0,
                    alpha=0.5,
                    ax = ax,
                    hue_order = highlight_subtypes + ["Other"]
                   )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
```

## SOX10 methylation

One of the top correlates that we discover is a strong relationship between demethylation of SOX10 (a key transcription factor involved in embryonic development) and sensitivity to SOX10 knockdown. Here we see that SOX10 demethylation results in re-activation of SOX10 expression, which induces a dependency on SOX10 for sustained growth.

Interestingly, this reactivation of SOX10 through demethylation appears to be almost entirely specific to melanomas.

```python
plot_methylation(tssclust_meth["SOX10_1"], ccle_genex["SOX10_ENSG00000100146.12"],
                 {"Other": "lightgrey", "Melanoma": "black"},
                 )

plt.xlabel("SOX10 methylation")
plt.ylabel("SOX10 mRNA expression")

plt.savefig("../plots/figure2-b.pdf",bbox_inches="tight",dpi=512,background="transparent")
```

```python
plot_methylation(tssclust_meth["SOX10_1"], avana["SOX10"],
                 {"Other": "lightgrey", "Melanoma": "black"},
                 )

plt.xlabel("SOX10 methylation")
plt.ylabel("SOX10 dependency")

plt.savefig("../plots/figure2-c.pdf",bbox_inches="tight",dpi=512,background="transparent")
```

## RPP25 methylation

Another top correlate that we find is one in which increased methylation of RPP25 (a component of ribonuclease P, an essential complex responsible for tRNA processing), which leads to dependence on its paralog RPP25L.

This is a classic example of a **paralog dependency**, in which deactivation of one gene causes a **synthetic lethal** reliance on its functionally redundant paralog. In particular, we see that RPP25 hypermethylation causes near-silencing of RPP25 expression, which causes a cell line to become dependent on RPP25L.

```python
plot_methylation(tssclust_meth["RPP25_1"], ccle_genex["RPP25_ENSG00000178718.5"],
                 {"Other": "lightgrey",
                  "Urinary tract": "#aa96da",
                  "Ovary": "#a8d8ea",
                  "Endometrium": "#cbf1f5",
                  "Glioma": "#3f72af"},
                 )

plt.xlabel("RPP25 methylation")
plt.ylabel("RPP25 mRNA expression")
```

```python
plot_methylation(tssclust_meth["RPP25_1"], avana["RPP25L"],
                 {"Other": "lightgrey",
                  "Urinary tract": "#aa96da",
                  "Ovary": "#a8d8ea",
                  "Endometrium": "#cbf1f5",
                  "Glioma": "#3f72af"},
                 )

plt.xlabel("RPP25 methylation")
plt.ylabel("RPP25L1 dependency")

plt.savefig("../plots/figure2-d.pdf",bbox_inches="tight",dpi=512,background="transparent")
```

## LDHB methylation

The LDHA/LDHB axis is of particular interest in cancer because this pair of genes is responsible for the essential process of pyruvate-lactate conversion. Here we show that certain cancers are hypermethylated in LDHB, which leads to reduced expression of LDHB and a paralog lethal dependency on LDHA. 

```python
plot_methylation(tssclust_meth["LDHB_1"], ccle_genex["LDHB_ENSG00000111716.8"],
                 {"Other": "lightgrey",
                  "Liver": "#abedd8",
                  "Pancreas": "#00b8a9",
                  "Stomach": "#a8e6cf",
                  "Breast": "#cca8e9",
                  "Prostate": "#6639a6"
                  },
                 )

plt.xlabel("LDHB methylation")
plt.ylabel("LDHB mRNA expression")
```

```python
plot_methylation(tssclust_meth["LDHB_1"], avana["LDHA"],
                 {"Other": "lightgrey",
                  "Liver": "#abedd8",
                  "Pancreas": "#00b8a9",
                  "Stomach": "#a8e6cf",
                  "Breast": "#cca8e9",
                  "Prostate": "#6639a6"
                  },
                 )

plt.xlabel("LDHB methylation")
plt.ylabel("LDHA dependency")

plt.savefig("../plots/figure2-e.pdf",bbox_inches="tight",dpi=512,background="transparent")
```

## VHL methylation

VHL (a well-known tumor suppressor gene) is commonly inactivated by damaging mutations in several cancers. Here we find that another mechanism for VHL inactivation is hypermethylation, which is associated with a marked loss of VHL mRNA levels.

```python
plot_methylation(tssclust_meth["VHL_2"], ccle_genex["VHL_ENSG00000134086.7"],
                 {"Other": "lightgrey",
                  "Kidney": "black"
                  },
                 )

plt.xlabel("VHL methylation")
plt.ylabel("VHL mRNA expression")
```
