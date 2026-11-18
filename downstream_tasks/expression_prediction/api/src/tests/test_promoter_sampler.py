"""Focused tests for transcript TSS promoter sampling."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from gena_expression import AnnotatedSequence, PromoterSampler
from gena_expression.dna.genome import _GtfRecord, _InMemoryAnnotation, _TabixGtfAnnotation


class _FakeGenome:
    """Small in-memory Genome-compatible object for sampler tests."""

    def __init__(self, records: list[_GtfRecord]) -> None:
        self._sequences = {
            "chr1": ("ACGTTGCA" * 80)[:500],
            "chr2": ("TTAACCGG" * 40)[:300],
        }
        records_by_chrom: dict[str, list[_GtfRecord]] = {}
        for record in records:
            records_by_chrom.setdefault(record.chrom, []).append(record)
        self._annotation = _InMemoryAnnotation(records_by_chrom)

    def normalize_chrom(self, chrom: str) -> str:
        """Resolve prefixed and unprefixed test chromosome labels."""

        candidates = (chrom, f"chr{chrom.removeprefix('chr')}")
        for candidate in candidates:
            if candidate in self._sequences:
                return candidate
        raise ValueError(chrom)

    def chrom_length(self, chrom: str) -> int:
        """Return the synthetic contig length."""

        return len(self._sequences[self.normalize_chrom(chrom)])

    def sequence(
        self,
        chrom: str,
        start: int,
        end: int,
        *,
        strand: str = "+",
        include_features: bool = True,
        name: str | None = None,
    ) -> AnnotatedSequence:
        """Return one synthetic sequence in the requested orientation."""

        del include_features
        sequence = AnnotatedSequence(
            self._sequences[self.normalize_chrom(chrom)][start:end],
            name=name,
        )
        if strand == "-":
            sequence = sequence.reverse_complement(name=name)
        return sequence


def _record(
    start: int,
    end: int,
    *,
    strand: str = "+",
    transcript_id: str,
    gene_id: str = "gene",
    chrom: str = "chr1",
    feature_type: str = "transcript",
) -> _GtfRecord:
    """Build one internal 0-based annotation record."""

    return _GtfRecord(
        chrom=chrom,
        source="test",
        feature_type=feature_type,
        start=start,
        end=end,
        score=".",
        strand=strand,
        frame=".",
        attributes={"transcript_id": transcript_id, "gene_id": gene_id},
    )


class PromoterSamplerTests(unittest.TestCase):
    """Verify TSS discovery, overlap grouping, and repeated sampling."""

    def test_overlapping_windows_form_one_sampling_unit_and_iterator_restarts(self):
        records = [
            _record(50, 100, transcript_id="tx1", gene_id="gene1"),
            _record(50, 110, transcript_id="tx1b", gene_id="gene1"),
            _record(60, 120, transcript_id="tx2", gene_id="gene2"),
            _record(180, 220, strand="-", transcript_id="tx3", gene_id="gene3"),
            _record(259, 300, transcript_id="tx4", gene_id="gene4"),
        ]
        sampler = PromoterSampler(
            _FakeGenome(records),
            seq_len=10,
            tss_sampling_window=40,
            seed=7,
        )

        self.assertEqual(len(sampler), 3)
        first_cycle = [next(sampler) for _ in range(3)]
        restarted = next(sampler)

        self.assertEqual(
            [sequence.metadata["promoter_group_index"] for sequence in first_cycle],
            [0, 1, 2],
        )
        self.assertTrue(all(sequence.metadata["promoter_cycle"] == 0 for sequence in first_cycle))
        self.assertEqual(restarted.metadata["promoter_group_index"], 0)
        self.assertEqual(restarted.metadata["promoter_cycle"], 1)
        self.assertTrue(all(len(sequence) == 10 for sequence in first_cycle + [restarted]))

        group_bounds = [
            (sequence.metadata["promoter_group_start"], sequence.metadata["promoter_group_end"])
            for sequence in first_cycle
        ]
        self.assertEqual(group_bounds, [(30, 80), (199, 239), (239, 279)])
        self.assertLessEqual(group_bounds[0][1], group_bounds[1][0])
        self.assertLessEqual(group_bounds[1][1], group_bounds[2][0])

        tss_50 = next(window for window in sampler._groups[0].windows if window.tss == 50)
        self.assertEqual(tss_50.transcript_ids, ("tx1", "tx1b"))

        for sequence in first_cycle + [restarted]:
            self.assertGreaterEqual(sequence.metadata["genomic_start"], sequence.metadata["sampling_window_start"])
            self.assertLessEqual(sequence.metadata["genomic_end"], sequence.metadata["sampling_window_end"])

    def test_negative_strand_sequence_is_reverse_complemented_and_tss_is_local(self):
        genome = _FakeGenome([_record(20, 30, strand="-", transcript_id="minus")])
        sampler = PromoterSampler(genome, seq_len=10, tss_sampling_window=10, seed=3)

        sequence = next(sampler)

        expected_forward = genome._sequences["chr1"][24:34]
        expected = str(AnnotatedSequence(expected_forward).reverse_complement())
        self.assertEqual(str(sequence), expected)
        self.assertEqual(sequence.metadata["tss"], 29)
        self.assertEqual(sequence.metadata["strand"], "-")
        self.assertTrue(sequence.metadata["tss_in_sequence"])
        tss = sequence.feature("tss")
        self.assertEqual((tss.start, tss.end), (4, 5))

    def test_seed_reproduces_member_and_coordinate_sampling(self):
        records = [
            _record(80, 120, transcript_id="tx1"),
            _record(85, 130, transcript_id="tx2"),
            _record(200, 240, transcript_id="tx3"),
        ]
        first = PromoterSampler(_FakeGenome(records), 8, 30, seed=19)
        second = PromoterSampler(_FakeGenome(records), 8, 30, seed=19)

        first_coordinates = [
            (next_sequence.metadata["tss"], next_sequence.metadata["genomic_start"])
            for next_sequence in (next(first) for _ in range(8))
        ]
        second_coordinates = [
            (next_sequence.metadata["tss"], next_sequence.metadata["genomic_start"])
            for next_sequence in (next(second) for _ in range(8))
        ]
        self.assertEqual(first_coordinates, second_coordinates)

    def test_invalid_configuration_and_boundary_only_windows_fail_clearly(self):
        genome = _FakeGenome([_record(2, 30, transcript_id="boundary")])

        with self.assertRaisesRegex(ValueError, "greater than or equal"):
            PromoterSampler(genome, seq_len=20, tss_sampling_window=10)
        with self.assertRaisesRegex(ValueError, "no usable transcript TSS windows"):
            PromoterSampler(genome, seq_len=8, tss_sampling_window=10)

    def test_missing_annotation_fails_at_construction(self):
        genome = _FakeGenome([])
        genome._annotation = None

        with self.assertRaisesRegex(ValueError, "requires a Genome with an annotation"):
            PromoterSampler(genome, seq_len=8, tss_sampling_window=10)

    def test_tabix_backend_can_iterate_all_records(self):
        class _FakeTabix:
            contigs = ("chr1",)

            @staticmethod
            def fetch(chrom: str):
                self.assertEqual(chrom, "chr1")
                return iter(
                    [
                        'chr1\ttest\ttranscript\t11\t20\t.\t+\t.\tgene_id "g1"; transcript_id "t1";',
                        'chr1\ttest\texon\t11\t15\t.\t+\t.\tgene_id "g1"; transcript_id "t1";',
                    ]
                )

        annotation = _TabixGtfAnnotation(Path("annotation.gtf.gz"))
        with patch.object(annotation, "_handle", return_value=_FakeTabix()):
            records = list(annotation.iter_records())

        self.assertEqual([record.feature_type for record in records], ["transcript", "exon"])
        self.assertEqual((records[0].start, records[0].end), (10, 20))


if __name__ == "__main__":
    unittest.main()
