"""Read-only anchor metadata for ARMT, preserving each original dataset/cache type."""

import numpy as np
import torch

from downstream_tasks.expression_prediction.expression_dataset_final import (
    ExpressionDataset,
    ExpressionDatasetMode2,
    ExpressionDatasetFixedDesc,
    ExpressionDatasetMode2FixedDesc,
    MethylationDataset,
    ExpressionDatasetTSSWindow,
    ExpressionDatasetMode2TSSWindow,
    MethylationDatasetTSSWindow,
)


def tss_boundary_index(starts, ends, tss, strand):
    """Position of the first downstream DNA token, including the initial CLS slot."""
    starts, ends = np.asarray(starts), np.asarray(ends)
    if strand == "+":
        upstream = ends <= tss
    elif strand == "-":
        upstream = starts >= tss
    else:
        raise ValueError(f"Unknown strand: {strand}")
    # Upstream tokens must precede downstream tokens in transcription order.
    count = int(upstream.sum())
    if upstream[count:].any() or not upstream[:count].all():
        raise ValueError("Token coordinates are not in transcription order around TSS")
    return count + 1


def interval_anchor_index(starts, ends, row, num_before):
    """Use the middle available DNA token for num_before=0, otherwise true TSS.

    CLS/SEP and padding are excluded by the caller. With an even number of DNA
    tokens choose the second of the two middle tokens in transcription order.
    This is a segmentation anchor; genomic TSS/TES metadata and H5 stay unchanged.
    """
    if num_before == 0:
        if not len(starts):
            raise ValueError("Cannot select a central token from an empty cached window")
        return len(starts) // 2 + 1
    return tss_boundary_index(starts, ends, int(row["TSS"]), row["strand"])


class ARMTCenterTSSMixin:
    def __getitem__(self, idx):
        features = super().__getitem__(idx)
        n_rows = features["input_ids"].shape[0]
        if isinstance(self, ExpressionDatasetMode2):
            cell_id = self.all_keys[idx // self.num_gene_chunks]
            gene_indices = self._get_selected_gene_indices_for_cell(cell_id, idx % self.num_gene_chunks)
        else:
            gene_indices = [self.valid_indices[idx // self.n_cell_chunks]] * n_rows
        boundaries = []
        for row_idx, gene_index in enumerate(gene_indices):
            row = self.genes.iloc[gene_index]
            group = self.h5_cache[row["gene_id"]]
            length = int(features["attention_mask"][row_idx].sum()) - 2
            boundaries.append(interval_anchor_index(
                group["starts"][:length], group["ends"][:length], row, self.num_before))
        boundaries.extend([-1] * (n_rows - len(boundaries)))
        features["tss_token_index"] = torch.tensor(boundaries, dtype=torch.long)
        return features


class ARMTCenterExpressionDataset(ARMTCenterTSSMixin, ExpressionDatasetTSSWindow):
    pass


class ARMTCenterExpressionDatasetMode2(ARMTCenterTSSMixin, ExpressionDatasetMode2TSSWindow):
    pass


class ARMTCenterMethylationDataset(ARMTCenterTSSMixin, MethylationDatasetTSSWindow):
    pass


class ARMTCenterExpressionDatasetFixedDesc(ARMTCenterTSSMixin, ExpressionDatasetFixedDesc):
    pass


class ARMTCenterExpressionDatasetMode2FixedDesc(ARMTCenterTSSMixin, ExpressionDatasetMode2FixedDesc):
    pass


class ARMTCenterExpressionDatasetLegacy(ARMTCenterTSSMixin, ExpressionDataset):
    pass


class ARMTCenterExpressionDatasetMode2Legacy(ARMTCenterTSSMixin, ExpressionDatasetMode2):
    pass


class ARMTCenterMethylationDatasetLegacy(ARMTCenterTSSMixin, MethylationDataset):
    pass
