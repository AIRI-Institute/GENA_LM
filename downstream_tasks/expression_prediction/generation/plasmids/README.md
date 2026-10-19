# Plasmids

Place local GenBank plasmid records here. A separate FASTA is optional because
`PlasmidRecord` can read DNA from the GenBank `ORIGIN` block. If both files are
configured, the project verifies that their sequences match.

The feature named by `plasmid.replacement_feature_name` must occur exactly once
and must be a simple contiguous feature that does not overlap the reporter TSS.
