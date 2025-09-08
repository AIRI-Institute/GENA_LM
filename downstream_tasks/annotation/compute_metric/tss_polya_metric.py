import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import numpy as np
import pandas as pd
import bisect
from collections import defaultdict

def find_segments_ones(array):
    
    ones_idx = np.where(array == 1)[0]
    if ones_idx.size == 0:
        return []

    split_idx = np.where(np.diff(ones_idx) > 1)[0] + 1
    split_ones_idx = np.split(ones_idx, split_idx)

    return [(segment[0], segment[-1] + 1) for segment in split_ones_idx]

def merge_intervals(intervals):
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        last = merged[-1]
        if current[0] < last[1]:
            merged[-1] = (last[0], max(last[1], current[1]))
        else:
            merged.append(current)
    return merged

def count_overlapping_segments(segments1, segments2):

    if not segments1 or not segments2:
        return 0, []

    merged = merge_intervals(segments2)
    starts = [m[0] for m in merged]
    
    count = 0
    overlapping_intervals = set()
    
    for s, e in segments1:
        i = bisect.bisect_right(starts, e - 1e-10)
        if i > 0 and merged[i-1][1] > s and starts[i-1] < e:
            count += 1
            overlapping_intervals.add(merged[i-1])
    
    return count, list(overlapping_intervals)

def compute_overlaps(predictions, df, label, max_k=100):
    gt_label = [(x, x+1) for x in df[label].tolist()]
    label_to_gene = {(row[label], row[label]+1): row['gene'] for index, row in df.iterrows()}
    
    results = []
    for k in tqdm(range(max_k + 1)):
        expanded_preds = [(s - k, e + k) for s, e in predictions]
        count, overlapping_gt = count_overlapping_segments(expanded_preds, gt_label)
        
        overlapped_genes = set()
        for t in overlapping_gt:
            t_tuple = tuple(t)
            if t_tuple in label_to_gene:
                overlapped_genes.add(label_to_gene[t_tuple])
        
        num_genes = len(overlapped_genes)
        num_non_overlap_preds = len(predictions) - count
        results.append((k, num_genes, num_non_overlap_preds))
    
    return results

def plot_overlaps(results, label, output_path):
    k_values = [r[0] for r in results]
    num_genes = [r[1] for r in results]
    num_non_overlap_preds = [r[2] for r in results]
    
    max_value = max(max(num_genes), max(num_non_overlap_preds))
    y_step = np.ceil(max_value / 10) if max_value > 0 else 1

    plt.figure(figsize=(14, 6))
    plt.plot(k_values, num_genes, color='orange', label='Number of Overlapped Genes', linewidth=2)
    plt.plot(k_values, num_non_overlap_preds, color='blue', label='Number of Non-Overlapped Predictions', linewidth=2)
    
    plt.gca().xaxis.set_major_locator(MultipleLocator(1))
    plt.gca().yaxis.set_major_locator(MultipleLocator(y_step))

    plt.xlabel('k (Expansion Size)')
    plt.ylabel('Count')
    plt.xlim(0, len(k_values) - 1)
    plt.title(label)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')

# #Example
# gt_path = 'path/gt.tsv'
# label = 'TSS'
# output_path = 'path/fig.png'
# predictions = [0, 1, 0, 0, 1, 0, 1]
# df = pd.read_csv(gt_path, sep='\t')
# results = compute_overlaps(find_segments_ones(predictions), df, label, max_k=50)
# plot_overlaps(results, label, output_path)
