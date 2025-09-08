#Example
gt_path = 'NC_060944.1.tsv'
labels = ['TSS', 'PolyA']

tss_polya = [[0, 0, 1], [0, 1, 1]]
print(tss_polya.shape)
df = pd.read_csv(gt_path, sep='\t')
for label in labels:
    output_path = f'{label}-caduceus_enhanced_v2.png'
    compute_metrics(tss_polya[labels.index(label)], df, label, output_path, max_k=250)