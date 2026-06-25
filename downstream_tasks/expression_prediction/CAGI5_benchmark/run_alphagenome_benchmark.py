import os
from alphagenome.data import gene_annotation
from alphagenome.data import genome
from alphagenome.data import transcript as transcript_utils
from alphagenome.visualization import plot_components
from alphagenome_research.model import dna_model
from alphagenome.models.variant_scorers import CenterMaskScorer
from alphagenome.models.variant_scorers import AggregationType
from alphagenome.models import variant_scorers
from alphagenome.models import dna_client
import numpy as np
import pandas as pd
import jax
import polars as pl
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


class AlphaGenomeCAGI5Benchmark:
    
    enformer_locus_to_ontology = {
        "F9": ["EFO:0001187"],
        "GP1BA": ["EFO:0002067"],
        "HBB": ["EFO:0002067"],
        "HBG1": ["EFO:0002067"],
        "HNF4A": ["UBERON:0002369"],
        "IRF4": ["CL:2000000", "CL:2000045", "EFO:0005720"],
        "IRF6": ["CL:0000312", "CL:1001606"],
        "LDLR": ["EFO:0001187"],
        "MSMB": ["UBERON:0002369"],
        "MYC": ["UBERON:0002369"],
        "PKLR": ["EFO:0002067"],
        "SORT1": ["EFO:0001187"],
        "TERT": ["UBERON:0002369"],
        "ZFAND3": ["CL:0002351", "UBERON:0001150", "UBERON:0001264"],
    }

    borzoi_locus_to_ontology = {
        "F9": ["EFO:0002067"],
        "GP1BA": ["EFO:0002067"],
        "HBB": ["EFO:0002067"],
        "HBG1": ["EFO:0002067"],
        "HNF4A": ["UBERON:0002113"],
        "IRF4": ["CL:2000000", "CL:2000045", "EFO:0005720"],
        "IRF6": ["CL:0000312", "CL:1001606"],
        "LDLR": ["UBERON:0002113"],
        "MSMB": ["UBERON:0002113"],
        "MYC": ["UBERON:0002113"],
        "PKLR": ["EFO:0002067"],
        "SORT1": ["EFO:0001187"],
        "TERT": ["UBERON:0002113"],
        "ZFAND3": ["CL:0002351", "UBERON:0001150", "UBERON:0001264"],
    }
    
    
    def __init__(
        self,
        model,
        locus_to_ontology,
        *,
        organism,
        scorer,
        interval_size=2**20,
        max_workers=20,
        score_rows_per_variant=305,
    ):
        self.model = model
        self.locus_to_ontology = locus_to_ontology
        self.organism = organism
        self.scorer = scorer
        self.interval_size = interval_size
        self.max_workers = max_workers
        self.score_rows_per_variant = score_rows_per_variant

    def make_intervals_and_variants(self, dataset: pl.DataFrame):
        intervals = []
        variants = []

        for var_str in dataset["variant"].to_list():
            variant = genome.Variant.from_str(var_str)
            interval = variant.reference_interval.resize(self.interval_size)
            variants.append(variant)
            intervals.append(interval)

        return intervals, variants

    def score_all_variants(self, dataset: pl.DataFrame) -> pl.DataFrame:
        intervals, variants = self.make_intervals_and_variants(dataset)

        prediction = self.model.score_variants(
            intervals,
            variants,
            variant_scorers=[self.scorer],
            organism=self.organism,
            max_workers=self.max_workers,
        )

        scores = variant_scorers.tidy_scores(prediction)[
            ["variant_id", "ontology_curie", "raw_score"]
        ]
        scores.variant_id = scores.variant_id.astype(str)

        if not isinstance(scores, pl.DataFrame):
            scores = pl.from_pandas(scores)

        scores = scores.with_columns(
            pl.col("variant_id").cast(pl.Utf8),
            pl.col("ontology_curie").cast(pl.Utf8),
            pl.col("raw_score").cast(pl.Float64),
        )

        expected_rows = len(dataset) * self.score_rows_per_variant
        if scores.height != expected_rows:
            print(
                f"Warning: expected {expected_rows} score rows "
                f"({len(dataset)} variants * {self.score_rows_per_variant}), "
                f"got {scores.height}."
            )

        return scores

    def parse_scores(
        self,
        dataset: pl.DataFrame,
        scores: pl.DataFrame,
        *,
        output_col="alphagenome_score",
    ) -> pl.DataFrame:
        ontology_map = pl.DataFrame(
            [
                {"Element": element, "ontology_curie": ontology}
                for element, ontologies in self.locus_to_ontology.items()
                for ontology in ontologies
            ]
        )
        dataset = dataset.with_columns(Element = pl.col.Element.map_elements(lambda el: el.split('.')[0].split('-')[0].split('rs')[0]))
        dataset_with_row_id = dataset.with_row_index("__row_id__")

        wanted_scores = (
            dataset_with_row_id
            .select("__row_id__", "Element", "variant")
            .join(ontology_map, on="Element", how="left")
        )
        
        wanted_scores = wanted_scores.filter(~pl.col("ontology_curie").is_null())

        #missing = wanted_scores.filter(pl.col("ontology_curie").is_null())
        #if missing.height:
        #    missing_elements = missing["Element"].unique().to_list()
        #    raise ValueError(f"No ontology mapping for elements: {missing_elements}")

        parsed = (
            wanted_scores
            .join(
                scores,
                left_on=["variant", "ontology_curie"],
                right_on=["variant_id", "ontology_curie"],
                how="left",
            )
            .group_by("__row_id__")
            .agg(
                pl.col("raw_score").mean().alias(output_col),
                pl.col("ontology_curie").alias("used_ontologies"),
                pl.col("raw_score").alias("ontology_scores"),
            )
        )

        missing_scores = parsed.filter(pl.col(output_col).is_null())
        if missing_scores.height:
            bad_rows = missing_scores["__row_id__"].to_list()[:10]
            raise ValueError(f"Missing AlphaGenome scores for dataset rows: {bad_rows}")

        return (
            dataset_with_row_id
            .join(parsed, on="__row_id__", how="left")
            .drop("__row_id__")
        )

    def run(self, dataset: pl.DataFrame):
        scores = self.score_all_variants(dataset)
        result = self.parse_scores(dataset, scores)
        return result
    
    
model = dna_model.create_from_huggingface('all_folds', device=jax.devices()[0], organism_settings={dna_model.Organism.HOMO_SAPIENS: dna_model.OrganismSettings(fasta_path='/home/jovyan/.cache/mpramnist/data/Kircher/hg38.fa'), 
                                                                                                   dna_model.Organism.MUS_MUSCULUS: (
            dna_model.OrganismSettings()
        )})
    
    
bench = AlphaGenomeCAGI5Benchmark(
    model=model,
    locus_to_ontology=AlphaGenomeCAGI5Benchmark.enformer_locus_to_ontology,
    organism=dna_model.Organism.HOMO_SAPIENS,
    scorer=CenterMaskScorer(
        requested_output=dna_model.OutputType.DNASE,
        width=501,
        aggregation_type=AggregationType.DIFF_SUM,
    ),
    max_workers=20,
)

dataset = pl.read_csv('/home/jovyan/.cache/mpramnist/data/Kircher/Kircher_GRCh38_ALL.tsv', separator='\t', infer_schema_length=10000)
variant_expr = pl.lit('chr') + pl.col('Chromosome').cast(pl.String) + pl.lit(':') + (pl.col('Position') + 1).cast(pl.String) + pl.lit(':') + pl.col('Ref').str.replace('-', '') + pl.lit('>') + pl.col('Alt').str.replace('-', '')
dataset = dataset.with_columns(variant = variant_expr)
dataset = dataset.filter(pl.col.Tags.__gt__(10))

result = bench.run(dataset)

result_for_csv = result.with_columns(
    pl.col("used_ontologies").list.join(";"),
    pl.col("ontology_scores")
      .list.eval(pl.element().cast(pl.Utf8))
      .list.join(";"),
)

result_for_csv.write_csv("alphagenome_cagi5_bench.tsv", separator="\t")