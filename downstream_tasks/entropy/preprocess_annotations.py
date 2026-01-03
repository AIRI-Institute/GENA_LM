# note: some code in this file depends on `datasets` package from expressoin branch
# follow datasets readme to download hg38 genome and gene annotations

import pandas as pd
from pybedtools import BedTool
nested_repeats = pd.read_csv("data/annotations/nestedRepeats.txt", sep="\t", header=None)

repeat_class_col = nested_repeats.columns.values[len(nested_repeats.columns)-1]
repeat_family_col = nested_repeats.columns.values[len(nested_repeats.columns)-2]

nested_repeats_subset = nested_repeats[(nested_repeats[repeat_class_col] != "") & \
                                       (nested_repeats[repeat_class_col] != " ") & \
                                       (nested_repeats[repeat_class_col] is not None)
                                    ]
nested_repeats_subset.dropna(inplace=True)
value_counts = nested_repeats_subset[repeat_class_col].value_counts()
common_repeats = value_counts[value_counts > 1000].index
print ("Common repeas are: ", common_repeats)
nested_repeats_subset = nested_repeats_subset[nested_repeats_subset[repeat_class_col].isin(common_repeats)]
nested_repeats_subset = nested_repeats_subset.iloc[:, [1, 2, 3, repeat_class_col, repeat_family_col]]
nested_repeats_subset.to_csv("data/annotations/nestedRepeats.bed", index=False, header=False, sep="\t")
value_counts = nested_repeats_subset[repeat_class_col].value_counts()
print ("----", value_counts.index[0])
print("Number of nested repeats in each class: \n", value_counts)

# simple repeats
simple_repeats = pd.read_csv("data/annotations/simpleRepeat.txt", sep="\t", header=None)
simple_repeats_subset = simple_repeats.iloc[:, [1, 2, 3]]
simple_repeats[4] = "simpleRepeat"
simple_repeats_subset.to_csv("data/annotations/simpleRepeats.bed", index=False, header=False, sep="\t")
# print ("Number of simple repeats: ", len(simple_repeats_subset)

# exons
exons = pd.read_csv("../expression_prediction/datasets/data/genomes/hg38/hg38_exons.csv")
exons["feature"] = "exon"
exons[["chromosome", "start", "end", "feature"]].to_csv("data/annotations/exons.bed", index=False, header=False, sep="\t")

# introns
genes = BedTool("../expression_prediction/datasets/data/genomes/hg38/hg38_genes.bed").sort().merge()
introns = genes.subtract(BedTool.from_dataframe(exons[["chromosome", "start", "end"]]).sort().merge()).to_dataframe()
introns["feature"] = "intron"
introns[["chrom", "start", "end", "feature"]].to_csv("data/annotations/introns.bed", index=False, header=False, sep="\t")

# promoters
promoters = pd.read_csv("../expression_prediction/datasets/data/genomes/hg38/hg38_genes.csv")
promoters["P_start"] = promoters.apply(lambda row: row["TSS"] - 2000 if row["gene_strand"] == "+" else row["TSS"], axis=1)
promoters["P_end"] = promoters.apply(lambda row: row["TSS"] if row["gene_strand"] == "+" else row["TSS"] + 2000, axis=1)
promoters["feature"] = "promoter"
promoters[["chromosome", "P_start", "P_end", "feature"]].to_csv("data/annotations/promoters.bed", index=False, header=False, sep="\t")