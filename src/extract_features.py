"""Run CARD RGI on genome FASTA files and build isolate-level features."""

import argparse
import hashlib
import re
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.parse import quote

import pandas as pd


REQUIRED_RGI_COLUMNS = ("Best_Hit_ARO", "Model_type", "AMR Gene Family")
MUTATION_COLUMN_ALIASES = ("SNPs", "S-NP", "SNP", "SNPs_in_Best_Hit_ARO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate genome FASTAs, run CARD RGI, and create an isolate-level "
            "binary feature matrix."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/genomes.csv"),
        help="CSV with isolate_id, organism, and fasta_path columns.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/genome_antibiotic_features.csv"),
        help="Path for the generated feature matrix.",
    )
    parser.add_argument(
        "--rgi-output-dir",
        type=Path,
        default=Path("data/rgi_results"),
        help="Directory in which RGI per-isolate outputs are stored.",
    )
    parser.add_argument(
        "--rgi-executable",
        default="rgi",
        help="RGI executable or full path to it (default: rgi).",
    )
    return parser.parse_args()


def validate_fasta(fasta_path: Path) -> int:
    if not fasta_path.is_file():
        raise FileNotFoundError("FASTA file not found: {}".format(fasta_path))

    allowed_bases = set("ACGTRYSWKMBDHVN")
    record_count = 0
    current_header: Optional[str] = None
    current_length = 0

    with fasta_path.open("r", encoding="utf-8") as fasta:
        for line_number, raw_line in enumerate(fasta, start=1):
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None and current_length == 0:
                    raise ValueError(
                        "FASTA record {!r} has no sequence in {}.".format(
                            current_header, fasta_path
                        )
                    )
                current_header = line[1:].strip()
                if not current_header:
                    raise ValueError(
                        "Empty FASTA header at line {} in {}.".format(
                            line_number, fasta_path
                        )
                    )
                record_count += 1
                current_length = 0
                continue

            if current_header is None:
                raise ValueError(
                    "Sequence data appears before the first FASTA header at "
                    "line {} in {}.".format(line_number, fasta_path)
                )

            sequence = "".join(line.split()).upper()
            invalid_bases = set(sequence).difference(allowed_bases)
            if invalid_bases:
                raise ValueError(
                    "Invalid nucleotide symbols {} at line {} in {}.".format(
                        "".join(sorted(invalid_bases)), line_number, fasta_path
                    )
                )
            current_length += len(sequence)

    if current_header is None:
        raise ValueError("No FASTA records found in {}.".format(fasta_path))
    if current_length == 0:
        raise ValueError(
            "FASTA record {!r} has no sequence in {}.".format(
                current_header, fasta_path
            )
        )
    return record_count


def load_manifest(manifest_path: Path) -> pd.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError("Genome manifest not found: {}".format(manifest_path))

    manifest = pd.read_csv(manifest_path, dtype=str)
    required = {"isolate_id", "organism", "fasta_path"}
    missing = required.difference(manifest.columns)
    if missing:
        raise ValueError(
            "Genome manifest is missing columns: {}".format(", ".join(sorted(missing)))
        )
    if manifest.empty:
        raise ValueError("Genome manifest contains no rows.")
    manifest = manifest.loc[:, ["isolate_id", "organism", "fasta_path"]].copy()
    for column in manifest.columns:
        manifest[column] = manifest[column].fillna("").str.strip()
    if (manifest == "").any(axis=None):
        raise ValueError(
            "Genome manifest isolate_id, organism, and fasta_path values "
            "must not be empty."
        )
    if manifest["isolate_id"].duplicated().any():
        duplicates = manifest.loc[
            manifest["isolate_id"].duplicated(keep=False), "isolate_id"
        ].tolist()
        raise ValueError(
            "Genome manifest isolate_id values must be unique: {}".format(
                sorted(set(duplicates))
            )
        )

    manifest_directory = manifest_path.resolve().parent
    manifest["fasta_path"] = manifest["fasta_path"].map(
        lambda path: str(
            (manifest_directory / Path(path)).resolve()
            if not Path(path).is_absolute()
            else Path(path).resolve()
        )
    )
    return manifest


def safe_output_stem(isolate_id: str) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "_", isolate_id).strip("._-")
    return readable or "isolate"


def run_rgi(
    executable: str, fasta_path: Path, output_prefix: Path
) -> Path:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "main",
        "--input_sequence",
        str(fasta_path),
        "--output_file",
        str(output_prefix),
        "--input_type",
        "contig",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as error:
        raise RuntimeError(
            "RGI executable {!r} was not found. Install RGI and make it available "
            "on PATH, or pass --rgi-executable.".format(executable)
        ) from error
    except subprocess.CalledProcessError as error:
        details = (error.stderr or error.stdout or "").strip()
        raise RuntimeError(
            "RGI failed for {} with exit code {}. {}".format(
                fasta_path, error.returncode, details
            )
        ) from error

    tabular_path = Path(str(output_prefix) + ".txt")
    if not tabular_path.is_file():
        raise RuntimeError(
            "RGI completed without producing the expected tabular output: {}".format(
                tabular_path
            )
        )
    return tabular_path


def encode_feature_value(value: str) -> str:
    return quote(value.strip(), safe="")


def optional_column(frame: pd.DataFrame, aliases: Sequence[str]) -> Optional[str]:
    return next((column for column in aliases if column in frame.columns), None)


def features_from_rgi(rgi_path: Path) -> Dict[str, int]:
    if rgi_path.stat().st_size == 0:
        raise ValueError("RGI output is empty: {}".format(rgi_path))

    rgi_frame = pd.read_csv(rgi_path, sep="\t", dtype=str, keep_default_na=False)
    missing = set(REQUIRED_RGI_COLUMNS).difference(rgi_frame.columns)
    if missing:
        raise ValueError(
            "RGI output {} is missing required columns: {}".format(
                rgi_path, ", ".join(sorted(missing))
            )
        )
    if rgi_frame.empty:
        return {}

    mutation_column = optional_column(rgi_frame, MUTATION_COLUMN_ALIASES)
    other_mutation_column = optional_column(rgi_frame, ("Other_SNPs",))
    cutoff_column = optional_column(rgi_frame, ("Cut_Off", "Cutoff"))
    if not cutoff_column:
        raise ValueError(
            "RGI output {} is missing the detection cutoff/status column "
            "(expected Cut_Off).".format(rgi_path)
        )
    feature_values: Dict[str, int] = {}

    for _, hit in rgi_frame.iterrows():
        aro = hit["Best_Hit_ARO"].strip()
        family = hit["AMR Gene Family"].strip()
        model_type = hit["Model_type"].strip()
        if not aro or not family or not model_type:
            raise ValueError(
                "RGI output {} contains a hit with an empty ARO, family, or "
                "model type.".format(rgi_path)
            )

        feature_values["gene__{}".format(encode_feature_value(aro))] = 1
        feature_values["family__{}".format(encode_feature_value(family))] = 1
        feature_values["model__{}".format(encode_feature_value(model_type))] = 1

        cutoff = hit[cutoff_column].strip()
        if cutoff:
            feature_values["status__{}__{}".format(
                encode_feature_value(aro), encode_feature_value(cutoff)
            )] = 1

        mutation_values = []
        if mutation_column:
            mutation_values.append(hit[mutation_column].strip())
        if other_mutation_column:
            mutation_values.append(hit[other_mutation_column].strip())
        for mutation_text in mutation_values:
            for mutation in (
                part.strip() for part in re.split(r"[;,]", mutation_text)
            ):
                if mutation and mutation.lower() not in {"n/a", "na", "none", "-"}:
                    feature_name = "{}__{}".format(
                        encode_feature_value(aro), encode_feature_value(mutation)
                    )
                    feature_values["mutation__{}".format(feature_name)] = 1

    return feature_values


def build_feature_matrix(
    manifest: pd.DataFrame, rgi_paths: Dict[str, Path]
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for isolate in manifest.itertuples(index=False):
        row: Dict[str, object] = {
            "isolate_id": isolate.isolate_id,
            "organism": isolate.organism,
        }
        row.update(features_from_rgi(rgi_paths[isolate.isolate_id]))
        rows.append(row)

    frame = pd.DataFrame(rows).fillna(0)
    metadata_columns = ["isolate_id", "organism"]
    feature_columns = sorted(column for column in frame.columns if column not in metadata_columns)
    return frame[metadata_columns + feature_columns]


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    rgi_paths: Dict[str, Path] = {}
    for isolate in manifest.itertuples(index=False):
        fasta_path = Path(isolate.fasta_path)
        record_count = validate_fasta(fasta_path)
        output_prefix = (
            args.rgi_output_dir
            / "{}_{}".format(
                safe_output_stem(isolate.isolate_id),
                hashlib.sha256(isolate.isolate_id.encode("utf-8")).hexdigest()[:12],
            )
        )
        rgi_paths[isolate.isolate_id] = run_rgi(
            args.rgi_executable, fasta_path, output_prefix
        )
        print(
            "Processed {} ({} FASTA records) for {}.".format(
                isolate.isolate_id, record_count, isolate.organism
            )
        )

    feature_matrix = build_feature_matrix(manifest, rgi_paths)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    feature_matrix.to_csv(args.output, index=False)
    print(
        "Saved {} isolate rows and {} genomic features to {}.".format(
            len(feature_matrix), feature_matrix.shape[1] - 2, args.output
        )
    )


if __name__ == "__main__":
    main()
