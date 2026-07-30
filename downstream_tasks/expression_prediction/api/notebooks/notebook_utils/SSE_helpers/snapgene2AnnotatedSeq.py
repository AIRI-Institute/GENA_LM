"""Convert SnapGene .dna files to variant_api AnnotatedSequence objects.

Use this as a standalone helper:

    from snapgene_to_annotated_sequence import snapgene_to_annotated_sequence

    seq = snapgene_to_annotated_sequence("plasmid.dna")

The module expects ``variant_api`` to be importable, because it returns the
``AnnotatedSequence`` class defined in ``variant_api.sequences``.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence
from xml.etree import ElementTree

from gena_expression.sequences import (AnnotatedSequence, 
                                        Feature, 
                                        CoordinateMap)


_SNAPGENE_COOKIE_PACKET = 0x09
_SNAPGENE_DNA_PACKET = 0x00
_SNAPGENE_PRIMERS_PACKET = 0x05
_SNAPGENE_NOTES_PACKET = 0x06
_SNAPGENE_FEATURES_PACKET = 0x0A


def snapgene_to_annotated_sequence(
    path: str | Path,
    *,
    name: str | None = None,
    include_primers: bool = True,
) -> AnnotatedSequence:
    """Read a SnapGene ``.dna`` file as an ``AnnotatedSequence``.

    SnapGene stores records as binary packets. This parser handles the core
    packets needed for annotation-preserving workflows: DNA bases, notes,
    sequence features, and primer binding sites. Coordinates are converted from
    SnapGene's 1-based inclusive ranges to 0-based half-open feature intervals.

    Multi-segment and circular-origin-spanning annotations are emitted as
    separate linked ``Feature`` rows because ``Feature`` represents one
    interval.
    """

    path = Path(path)
    packets = list(_iter_snapgene_packets(path.read_bytes()))
    if not packets:
        raise ValueError(f"SnapGene file is empty: {path}")

    packet_type, cookie = packets[0]
    if packet_type != _SNAPGENE_COOKIE_PACKET:
        raise ValueError("SnapGene file does not start with a cookie packet.")
    _validate_snapgene_cookie(cookie)

    sequence: str | None = None
    dna_flags: int | None = None
    notes: dict[str, str] = {}
    for packet_type, data in packets[1:]:
        if packet_type == _SNAPGENE_DNA_PACKET:
            if sequence is not None:
                raise ValueError("SnapGene file contains more than one DNA packet.")
            if not data:
                raise ValueError("SnapGene DNA packet is empty.")
            dna_flags = data[0]
            sequence = data[1:].decode("ascii")
        elif packet_type == _SNAPGENE_NOTES_PACKET:
            notes.update(_parse_snapgene_notes(data))

    if sequence is None:
        raise ValueError("SnapGene file does not contain a DNA packet.")

    features: list[Feature] = []
    for packet_type, data in packets[1:]:
        if packet_type == _SNAPGENE_FEATURES_PACKET:
            features.extend(_parse_snapgene_features(data, len(sequence)))
        elif include_primers and packet_type == _SNAPGENE_PRIMERS_PACKET:
            features.extend(_parse_snapgene_primers(data, len(sequence)))

    sequence_name = name or _snapgene_record_name(notes) or path.stem
    topology = "circular" if (dna_flags or 0) & 0x01 else "linear"
    metadata = {
        "source_format": "snapgene",
        "source_path": str(path),
        "topology": topology,
        "snapgene_flags": dna_flags,
        "snapgene_notes": notes,
    }
    return AnnotatedSequence(
        sequence,
        name=sequence_name,
        features=features,
        coordinate_map=CoordinateMap.from_length(len(sequence), source=str(path)),
        metadata=metadata,
    )


def _iter_snapgene_packets(data: bytes) -> Iterable[tuple[int, bytes]]:
    offset = 0
    while offset < len(data):
        if offset + 5 > len(data):
            raise ValueError("Unexpected end of SnapGene packet header.")
        packet_type = data[offset]
        length = int.from_bytes(data[offset + 1 : offset + 5], "big")
        offset += 5
        packet_data = data[offset : offset + length]
        if len(packet_data) != length:
            raise ValueError("Unexpected end of SnapGene packet data.")
        offset += length
        yield packet_type, packet_data


def _validate_snapgene_cookie(data: bytes) -> None:
    if len(data) < 8 or data[:8].decode("ascii", errors="replace") != "SnapGene":
        raise ValueError("The file is not a valid SnapGene file.")


def _parse_snapgene_notes(data: bytes) -> dict[str, str]:
    root = _parse_snapgene_xml(data)
    notes: dict[str, str] = {}
    for child in root:
        value = _decode_snapgene_text("".join(child.itertext()).strip())
        if value:
            notes[child.tag] = value
    return notes


def _parse_snapgene_features(data: bytes, sequence_length: int) -> list[Feature]:
    root = _parse_snapgene_xml(data)
    features: list[Feature] = []
    for feature_index, feature_node in enumerate(root.findall(".//Feature"), start=1):
        feature_type = _snapgene_attr(feature_node, "type", default="misc_feature")
        feature_name = _snapgene_attr(feature_node, "name")
        directionality = _snapgene_attr(feature_node, "directionality")
        strand = _snapgene_feature_strand(directionality)
        qualifiers = _parse_snapgene_qualifiers(feature_node)
        segments = [
            segment
            for segment in feature_node.findall("Segment")
            if _snapgene_attr(segment, "type", default="standard") != "gap"
        ]
        if not segments:
            raise ValueError(f"SnapGene feature {feature_name or feature_index!r} has no location segment.")

        for segment_index, segment in enumerate(segments, start=1):
            range_spec = _snapgene_attr(segment, "range", required=True)
            assert range_spec is not None
            segment_name = _snapgene_attr(segment, "name")
            intervals = _snapgene_range_to_intervals(range_spec, sequence_length)
            if not intervals:
                continue
            display_name = _snapgene_feature_name(
                feature_name=feature_name,
                segment_name=segment_name,
                qualifiers=qualifiers,
                feature_type=feature_type,
            )
            for interval_index, (start, end) in enumerate(intervals, start=1):
                features.append(
                    Feature(
                        name=display_name,
                        start=start,
                        end=end,
                        type=feature_type,
                        strand=strand,
                        source="snapgene",
                        metadata=_snapgene_feature_metadata(
                            feature_index=feature_index,
                            feature_name=feature_name,
                            segment_index=segment_index,
                            segment_count=len(segments),
                            segment_name=segment_name,
                            interval_index=interval_index,
                            interval_count=len(intervals),
                            range_spec=range_spec,
                            qualifiers=qualifiers,
                            extra={
                                "directionality": directionality,
                                "color": _snapgene_attr(feature_node, "color"),
                            },
                        ),
                    )
                )
    return features


def _parse_snapgene_primers(data: bytes, sequence_length: int) -> list[Feature]:
    root = _parse_snapgene_xml(data)
    features: list[Feature] = []
    min_match_length = 0
    min_melting_temperature = 0
    for params in root.findall(".//HybridizationParams"):
        min_match_length = _snapgene_int_attr(params, "minContinuousMatchLen", default=0)
        min_melting_temperature = _snapgene_int_attr(params, "minMeltingTemperature", default=0)

    for primer_index, primer in enumerate(root.findall(".//Primer"), start=1):
        primer_name = _snapgene_attr(primer, "name") or f"primer_{primer_index}"
        seen_simplified_sites: set[tuple[int, int, str]] = set()
        binding_sites = primer.findall("BindingSite")
        for site_index, site in enumerate(binding_sites, start=1):
            annealed_bases = _snapgene_attr(site, "annealedBases")
            if annealed_bases is not None and len(annealed_bases) < min_match_length:
                continue
            melting_temperature = _snapgene_attr(site, "meltingTemperature")
            if melting_temperature is not None and int(float(melting_temperature)) < min_melting_temperature:
                continue

            range_spec = _snapgene_attr(site, "location", required=True)
            assert range_spec is not None
            strand = "-" if _snapgene_attr(site, "boundStrand", default="0") == "1" else "+"
            intervals = _snapgene_range_to_intervals(range_spec, sequence_length, primer=True)
            for interval_index, (start, end) in enumerate(intervals, start=1):
                site_key = (start, end, strand)
                simplified = _snapgene_attr(site, "simplified", default="0") == "1"
                if simplified and site_key in seen_simplified_sites:
                    continue
                seen_simplified_sites.add(site_key)
                features.append(
                    Feature(
                        name=primer_name,
                        start=start,
                        end=end,
                        type="primer_bind",
                        strand=strand,
                        source="snapgene",
                        metadata={
                            "source_format": "snapgene",
                            "primer_index": primer_index,
                            "site_index": site_index,
                            "site_count": len(binding_sites),
                            "interval_index": interval_index,
                            "interval_count": len(intervals),
                            "range": range_spec,
                            "annealed_bases": annealed_bases,
                            "melting_temperature": melting_temperature,
                            "simplified": simplified,
                        },
                    )
                )
    return features


def _parse_snapgene_xml(data: bytes) -> ElementTree.Element:
    try:
        return ElementTree.fromstring(data.decode("utf-8"))
    except ElementTree.ParseError as exc:
        raise ValueError(f"Could not parse SnapGene XML packet: {exc}") from exc


def _snapgene_attr(
    node: ElementTree.Element,
    name: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    if name in node.attrib:
        return _decode_snapgene_text(node.attrib[name])
    if required:
        raise ValueError(f"Missing SnapGene XML attribute {name!r}.")
    return default


def _snapgene_int_attr(node: ElementTree.Element, name: str, *, default: int = 0) -> int:
    value = _snapgene_attr(node, name)
    return default if value is None else int(value)


def _decode_snapgene_text(text: str) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    return re.sub(r"\r\n|\r|\n", " ", text).strip()


def _parse_snapgene_qualifiers(feature_node: ElementTree.Element) -> dict[str, list[Any]]:
    qualifiers: dict[str, list[Any]] = {}
    for qualifier in feature_node.findall("Q"):
        name = _snapgene_attr(qualifier, "name", required=True)
        assert name is not None
        values: list[Any] = []
        for value_node in qualifier.findall("V"):
            if "text" in value_node.attrib:
                values.append(_decode_snapgene_text(value_node.attrib["text"]))
            elif "predef" in value_node.attrib:
                values.append(_decode_snapgene_text(value_node.attrib["predef"]))
            elif "int" in value_node.attrib:
                values.append(int(value_node.attrib["int"]))
        qualifiers[name] = values
    return qualifiers


def _snapgene_feature_name(
    *,
    feature_name: str | None,
    segment_name: str | None,
    qualifiers: Mapping[str, Sequence[Any]],
    feature_type: str,
) -> str:
    for key in ("label", "gene", "product", "note"):
        values = qualifiers.get(key)
        if values:
            return str(values[0])
    return feature_name or segment_name or feature_type or "feature"


def _snapgene_feature_strand(directionality: str | None) -> str | None:
    if directionality == "1":
        return "+"
    if directionality == "2":
        return "-"
    return None


def _snapgene_range_to_intervals(
    range_spec: str,
    sequence_length: int,
    *,
    primer: bool = False,
) -> list[tuple[int, int]]:
    try:
        left, right = (int(value) for value in range_spec.split("-", 1))
    except ValueError as exc:
        raise ValueError(f"Could not parse SnapGene feature range {range_spec!r}.") from exc

    start = left - 1
    end = right
    if primer:
        start += 1
        end += 1

    raw_intervals = [(start, sequence_length), (0, end)] if start >= end else [(start, end)]
    intervals: list[tuple[int, int]] = []
    for raw_start, raw_end in raw_intervals:
        clipped_start = max(0, min(sequence_length, raw_start))
        clipped_end = max(0, min(sequence_length, raw_end))
        if clipped_end > clipped_start:
            intervals.append((clipped_start, clipped_end))
    return intervals


def _snapgene_feature_metadata(
    *,
    feature_index: int,
    feature_name: str | None,
    segment_index: int,
    segment_count: int,
    segment_name: str | None,
    interval_index: int,
    interval_count: int,
    range_spec: str,
    qualifiers: Mapping[str, Sequence[Any]],
    extra: Mapping[str, Any],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "source_format": "snapgene",
        "feature_index": feature_index,
        "feature_name": feature_name,
        "segment_index": segment_index,
        "segment_count": segment_count,
        "segment_name": segment_name,
        "interval_index": interval_index,
        "interval_count": interval_count,
        "range": range_spec,
        "qualifiers": dict(qualifiers),
    }
    metadata.update({key: value for key, value in extra.items() if value is not None})
    return metadata


def _snapgene_record_name(notes: Mapping[str, str]) -> str | None:
    for key in ("Name", "SequenceName", "AccessionNumber"):
        value = notes.get(key)
        if value:
            return value
    comments = notes.get("Comments")
    return comments.split(" ", 1)[0] if comments else None
