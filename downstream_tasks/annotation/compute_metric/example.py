#Example
gt_path = 'NC_060944.1.tsv'
labels = ['TSS', 'PolyA']

tss_polya = np.load('NC_060944.1-chromosome-tss_polya-caduceus_enhanced_v2-100k-forward.npy')
print(tss_polya.shape)
df = pd.read_csv(gt_path, sep='\t')
for label in labels:
    output_path = f'{label}-caduceus_enhanced_v2.png'
    compute_metrics(tss_polya, df, label, labels, output_path, max_k=250)