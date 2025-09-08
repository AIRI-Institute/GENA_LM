import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from tqdm import tqdm
import numpy as np
import bisect

def compute_metrics(tss_polya, df, label, output_path, max_k=250):
    def find_segments_ones(array):
        
        ones_idx = np.where(array == 1)[0]
        if ones_idx.size == 0:
            return []

        split_idx = np.where(np.diff(ones_idx) > 1)[0] + 1
        split_ones_idx = np.split(ones_idx, split_idx)

        return [(segment[0], segment[-1] + 1) for segment in split_ones_idx]

    def count_overlapping_segments(segments1, segments2):
        if not segments1 or not segments2:
            return 0, []

        segments2 = sorted(segments2, key=lambda x: x[0])
        merged = [segments2[0]]
        for current in segments2[1:]:
            last = merged[-1]
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)
        
        starts = [m[0] for m in merged]
        ends = [m[1] for m in merged]
        
        count = 0
        overlapping_intervals = set()
        
        for s, e in segments1:
            left = bisect.bisect_left(ends, s + 1)
            
            overlapped = False
            j = left
            while j < len(merged) and merged[j][0] < e:
                if merged[j][1] > s:
                    overlapping_intervals.add(merged[j])
                    overlapped = True
                j += 1
            
            if overlapped:
                count += 1
        
        return count, list(overlapping_intervals)

    def compute_overlaps(predictions, df, label, max_k=250):
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
        
        # First figure: Number of Overlapped Genes
        max_value_genes = max(num_genes) if num_genes else 1
        y_step_genes = np.ceil(max_value_genes / 10) if max_value_genes > 0 else 1

        plt.figure(figsize=(14, 6))
        plt.plot(k_values, num_genes, color='orange', label=f'Number of Overlapped Genes (k=50: {num_genes[k_values.index(50)] if 50 in k_values else "N/A"}, k=250: {num_genes[k_values.index(250)] if 250 in k_values else "N/A"})', linewidth=2)
        
        # Add red dots at k=50 and k=250
        for k in [50, 250]:
            if k in k_values:
                idx = k_values.index(k)
                plt.plot(k, num_genes[idx], 'ro', markersize=8)
        
        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(MultipleLocator(y_step_genes))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

        plt.xlabel('k (Expansion Size)')
        plt.ylabel('Count')
        plt.xlim(0, len(k_values) - 1)
        plt.ylim(0, max_value_genes + y_step_genes)
        plt.title(f"{label} - Overlapped Genes")
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        
        # Save first figure
        output_path1 = output_path.replace('.png', '_overlapped_genes.png')
        plt.savefig(output_path1, dpi=300, bbox_inches=None)
        plt.close()

        # Second figure: Number of Non-Overlapped Predictions
        max_value_non_overlap = max(num_non_overlap_preds) if num_non_overlap_preds else 1
        y_step_non_overlap = np.ceil(max_value_non_overlap / 10) if max_value_non_overlap > 0 else 1

        plt.figure(figsize=(14, 6))
        plt.plot(k_values, num_non_overlap_preds, color='blue', label=f'Number of Non-Overlapped Predictions (k=50: {num_non_overlap_preds[k_values.index(50)] if 50 in k_values else "N/A"}, k=250: {num_non_overlap_preds[k_values.index(250)] if 250 in k_values else "N/A"})', linewidth=2)
        
        # Add red dots at k=50 and k=250
        for k in [50, 250]:
            if k in k_values:
                idx = k_values.index(k)
                plt.plot(k, num_non_overlap_preds[idx], 'ro', markersize=8)
        
        plt.gca().xaxis.set_major_locator(MultipleLocator(5))
        plt.gca().yaxis.set_major_locator(MultipleLocator(y_step_non_overlap))
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x)}'))

        plt.xlabel('k (Expansion Size)')
        plt.ylabel('Count')
        plt.xlim(0, len(k_values) - 1)
        plt.ylim(0, max_value_non_overlap + y_step_non_overlap)
        plt.title(f"{label} - Non-Overlapped Predictions")
        plt.xticks(rotation=90)
        plt.legend()
        plt.grid(True)
        
        # Save second figure
        output_path2 = output_path.replace('.png', '_non_overlapped_preds.png')
        plt.savefig(output_path2, dpi=300, bbox_inches=None)
        plt.close()

    results = compute_overlaps(find_segments_ones(tss_polya), df, label, max_k)
    plot_overlaps(results, label, output_path)
