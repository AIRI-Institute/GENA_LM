#Example
gt_path = 'path/gt.tsv' #путь к файлу GT, который я дал. Если нужны будут тесты на другой хромосоме, то надо делать другой файл
label = 'TSS' #название класса предикта
output_path = 'path/fig.png' #название выходной картинки
predictions = [0, 1, 0, 0, 1, 0, 1] #массив размером с хромосому, где 1 выделены предикты выбранного класса
df = pd.read_csv(gt_path, sep='\t')
results = compute_overlaps(find_segments_ones(predictions), df, label, max_k=50) #здесь можно задать увеличение пика
plot_overlaps(results, label, output_path)