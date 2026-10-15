"""Train an isolate-grouped AST model from NCBI isolate genotype and NLM data."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.sparse import csr_matrix, vstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit

TARGET_MAP = {"resistant": "R", "susceptible": "S"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join NCBI precomputed AMR genotypes to NLM AST labels and train a "
            "binary classifier with BioSamples held out as groups."
        )
    )
    parser.add_argument(
        "--isolates",
        type=Path,
        default=Path("data/isolates.csv"),
        help="NCBI isolate metadata CSV containing BioSample and AMR genotypes.",
    )
    parser.add_argument(
        "--ast",
        type=Path,
        default=Path("data/NLM.csv"),
        help="NLM AST CSV containing BioSample, Antibiotic, and Resistance phenotype.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/nlm_resistance_model.joblib"),
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--max-iter", type=int, default=300)
    return parser.parse_args()


def read_ast(ast_path: Path) -> Tuple[pd.DataFrame, Dict[str, int]]:
    if not ast_path.is_file():
        raise FileNotFoundError("NLM AST CSV not found: {}".format(ast_path))

    required_columns = {
        "#BioSample",
        "Organism group",
        "Antibiotic",
        "Resistance phenotype",
    }
    frame = pd.read_csv(
        ast_path,
        usecols=lambda column: column in required_columns,
        dtype={"#BioSample": str},
        escapechar="\\",
        low_memory=False,
    )
    missing = required_columns.difference(frame.columns)
    if missing:
        raise ValueError(
            "NLM CSV is missing columns: {}".format(", ".join(sorted(missing)))
        )
    if frame.empty:
        raise ValueError("NLM CSV contains no AST rows.")
    ast_rows_read = len(frame)

    for column in required_columns:
        frame[column] = frame[column].fillna("").astype(str).str.strip()
    frame["Resistance phenotype"] = frame["Resistance phenotype"].str.lower()
    frame = frame.rename(
        columns={
            "#BioSample": "biosample",
            "Organism group": "ast_organism",
            "Antibiotic": "antibiotic",
            "Resistance phenotype": "phenotype",
        }
    )
    blank_rows = int(
        (
            frame["biosample"].eq("")
            | frame["antibiotic"].eq("")
            | frame["phenotype"].eq("")
        ).sum()
    )
    frame = frame[
        frame["biosample"].ne("")
        & frame["antibiotic"].ne("")
        & frame["phenotype"].ne("")
    ].copy()

    supported = frame["phenotype"].isin(TARGET_MAP)
    unsupported_rows = int((~supported).sum())
    frame = frame.loc[supported].copy()
    if frame.empty:
        raise ValueError("NLM CSV contains no resistant/susceptible AST records.")
    frame["target"] = frame["phenotype"].map(TARGET_MAP)

    pair_labels = frame.groupby(["biosample", "antibiotic"])["target"].agg(
        size="size", classes="nunique"
    )
    conflicting_pairs = pair_labels.index[pair_labels["classes"] > 1]
    conflict_pair_count = len(conflicting_pairs)
    duplicate_pair_count = int(
        (pair_labels.loc[pair_labels["classes"] == 1, "size"] - 1).sum()
    )
    if conflict_pair_count:
        conflict_index = pd.MultiIndex.from_frame(
            frame[["biosample", "antibiotic"]]
        )
        frame = frame.loc[~conflict_index.isin(conflicting_pairs)].copy()
    frame = frame.drop_duplicates(["biosample", "antibiotic"]).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No unambiguous resistant/susceptible AST pairs remain.")

    return frame[["biosample", "ast_organism", "antibiotic", "target"]], {
        "ast_rows_read": int(ast_rows_read),
        "blank_ast_rows_excluded": blank_rows,
        "unsupported_phenotype_rows_excluded": unsupported_rows,
        "duplicate_ast_rows_collapsed": duplicate_pair_count,
        "conflicting_biosample_antibiotic_pairs_excluded": conflict_pair_count,
    }


def parse_genotypes(value: str, biosample: str) -> List[str]:
    features: Set[str] = set()
    for entry in value.split(","):
        entry = entry.strip()
        if not entry:
            continue
        # NCBI exports can contain either a gene plus a call state
        # (for example ``gyrA_D87Y=POINT``) or a gene-only call such as
        # ``blaCTX-M-15``. A gene-only call remains useful to the model and
        # should not reject an otherwise valid uploaded isolate.
        if "=" in entry:
            gene, state = (part.strip() for part in entry.rsplit("=", 1))
        else:
            gene, state = entry, ""
        if not gene or ("=" in entry and not state):
            raise ValueError(
                "Malformed AMR genotype {!r} for BioSample {}.".format(
                    entry, biosample
                )
            )
        features.add("gene::{}".format(gene))
        if state:
            features.add("call::{}::{}".format(gene, state))
    return sorted(features)


def read_isolate_genotypes(
    isolates_path: Path, biosamples: Set[str]
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    if not isolates_path.is_file():
        raise FileNotFoundError(
            "Isolate metadata CSV not found: {}".format(isolates_path)
        )

    required_columns = {
        "#Organism group",
        "BioSample",
        "AMR genotypes",
        "SNP cluster",
    }
    sample_records: Dict[str, Tuple[str, str, str]] = {}
    conflicting_samples: Set[str] = set()
    matched_rows = 0
    for chunk in pd.read_csv(
        isolates_path,
        usecols=lambda column: column in required_columns,
        dtype={"BioSample": str},
        chunksize=50000,
        low_memory=False,
    ):
        missing = required_columns.difference(chunk.columns)
        if missing:
            raise ValueError(
                "Isolate CSV is missing columns: {}".format(
                    ", ".join(sorted(missing))
                )
            )
        chunk["BioSample"] = chunk["BioSample"].fillna("").astype(str).str.strip()
        selected = chunk.loc[chunk["BioSample"].isin(biosamples)]
        matched_rows += len(selected)
        selected = selected[
            ["#Organism group", "BioSample", "AMR genotypes", "SNP cluster"]
        ]
        for row in selected.itertuples(index=False, name=None):
            organism, biosample, genotype, cluster = row
            biosample = str(biosample).strip()
            profile = (
                str(organism).strip(),
                "" if pd.isna(genotype) else str(genotype).strip(),
                "" if pd.isna(cluster) else str(cluster).strip(),
            )
            previous = sample_records.get(biosample)
            if previous is None:
                sample_records[biosample] = profile
            elif previous != profile:
                conflicting_samples.add(biosample)

    for biosample in conflicting_samples:
        sample_records.pop(biosample, None)

    genotype_features: Dict[str, Dict[str, Any]] = {}
    missing_genotype_samples = 0
    for biosample, (organism, genotype, cluster) in sample_records.items():
        if not organism:
            raise ValueError(
                "Isolate metadata has an empty organism for BioSample {}.".format(
                    biosample
                )
            )
        if not genotype or genotype.lower() in {"na", "nan", "none"}:
            missing_genotype_samples += 1
            continue
        genotype_features[biosample] = {
            "organism": organism,
            "genotypes": parse_genotypes(genotype, biosample),
            "split_group": (
                "cluster::{}".format(cluster)
                if cluster
                else "biosample::{}".format(biosample)
            ),
        }

    return genotype_features, {
        "ast_biosamples_without_isolate_metadata": len(
            biosamples.difference(sample_records).difference(conflicting_samples)
        ),
        "isolate_rows_matching_ast_biosamples": matched_rows,
        "biosamples_with_conflicting_isolate_genotypes_excluded": len(
            conflicting_samples
        ),
        "biosamples_with_missing_genotypes_excluded": missing_genotype_samples,
        "biosamples_with_usable_genotypes": len(genotype_features),
        "biosamples_with_snp_cluster": sum(
            bool(isolate["split_group"].startswith("cluster::"))
            for isolate in genotype_features.values()
        ),
        "biosamples_using_biosample_split_fallback": sum(
            bool(isolate["split_group"].startswith("biosample::"))
            for isolate in genotype_features.values()
        ),
    }


def row_features(
    biosample: str,
    antibiotic: str,
    genotype_features: Dict[str, Dict[str, Any]],
) -> Dict[str, float]:
    isolate = genotype_features[biosample]
    features = {name: 1.0 for name in isolate["genotypes"]}
    features["antibiotic::{}".format(antibiotic)] = 1.0
    features["organism::{}".format(isolate["organism"])] = 1.0
    return features


def transform_in_batches(
    vectorizer: DictVectorizer,
    rows: Iterable[Tuple[str, str]],
    genotype_features: Dict[str, Dict[str, Any]],
    batch_size: int,
) -> csr_matrix:
    blocks = []
    batch: List[Dict[str, float]] = []
    for biosample, antibiotic in rows:
        batch.append(row_features(biosample, antibiotic, genotype_features))
        if len(batch) >= batch_size:
            blocks.append(vectorizer.transform(iter(batch)))
            batch.clear()
    if batch:
        blocks.append(vectorizer.transform(iter(batch)))
    if not blocks:
        raise ValueError("Cannot build a feature matrix with no rows.")
    return csr_matrix(vstack(blocks, format="csr"))


def evaluate(
    classifier: LogisticRegression,
    matrix: csr_matrix,
    targets: pd.Series,
) -> Dict[str, Any]:
    predictions = classifier.predict(matrix)
    return {
        "accuracy": float(accuracy_score(targets, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(targets, predictions)
        ),
        "macro_f1": float(f1_score(targets, predictions, average="macro")),
        "classification_report": classification_report(
            targets,
            predictions,
            labels=["R", "S"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            targets, predictions, labels=["R", "S"]
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1.")
    if args.max_iter < 1:
        raise ValueError("--max-iter must be at least 1.")

    ast_frame, ast_counts = read_ast(args.ast)
    genotype_features, isolate_counts = read_isolate_genotypes(
        args.isolates, set(ast_frame["biosample"])
    )
    joined = ast_frame.loc[
        ast_frame["biosample"].isin(genotype_features)
    ].reset_index(drop=True)
    if joined.empty:
        raise ValueError(
            "No unambiguous AST records with usable genotype features were matched."
        )
    if joined["target"].nunique() < 2:
        raise ValueError("Joined AST data must contain both R and S labels.")
    if joined["biosample"].nunique() < 2:
        raise ValueError("At least two BioSamples are required for grouped evaluation.")

    joined["split_group"] = joined["biosample"].map(
        lambda biosample: genotype_features[str(biosample)]["split_group"]
    )
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.test_size, random_state=args.random_state
    )
    train_indices, test_indices = next(
        splitter.split(joined, joined["target"], groups=joined["split_group"])
    )
    train_frame = joined.iloc[train_indices].reset_index(drop=True)
    test_frame = joined.iloc[test_indices].reset_index(drop=True)
    train_groups = set(train_frame["split_group"])
    test_groups = set(test_frame["split_group"])
    if not train_groups.isdisjoint(test_groups):
        raise RuntimeError("SNP-cluster/BioSample leakage detected between splits.")
    if train_frame["target"].nunique() < 2 or test_frame["target"].nunique() < 2:
        raise ValueError(
            "The BioSample-grouped split must contain both R and S in train and "
            "test. Add data or change --random-state/--test-size."
        )

    vocabulary: Dict[str, float] = {}
    train_biosamples = set(train_frame["biosample"])
    for biosample in train_biosamples:
        isolate = genotype_features[biosample]
        for feature_name in isolate["genotypes"]:
            vocabulary[feature_name] = 1.0
        vocabulary["organism::{}".format(isolate["organism"])] = 1.0
    for antibiotic in train_frame["antibiotic"].unique():
        vocabulary["antibiotic::{}".format(antibiotic)] = 1.0

    vectorizer = DictVectorizer(sparse=True, dtype=np.float64)
    vectorizer.fit([vocabulary])
    del vocabulary

    x_train = transform_in_batches(
        vectorizer,
        train_frame[["biosample", "antibiotic"]].itertuples(index=False, name=None),
        genotype_features,
        args.batch_size,
    )
    x_test = transform_in_batches(
        vectorizer,
        test_frame[["biosample", "antibiotic"]].itertuples(index=False, name=None),
        genotype_features,
        args.batch_size,
    )

    classifier = LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=args.max_iter,
        tol=1e-3,
        random_state=args.random_state,
    )
    classifier.fit(x_train, train_frame["target"])
    metrics = evaluate(classifier, x_test, test_frame["target"])

    per_antibiotic: Dict[str, Any] = {}
    test_predictions = classifier.predict(x_test)
    evaluated = test_frame[["antibiotic", "target"]].copy()
    evaluated["prediction"] = test_predictions
    for antibiotic, group in evaluated.groupby("antibiotic"):
        if group["target"].nunique() < 2:
            continue
        per_antibiotic[antibiotic] = {
            "n": int(len(group)),
            "balanced_accuracy": float(
                balanced_accuracy_score(group["target"], group["prediction"])
            ),
            "macro_f1": float(
                f1_score(group["target"], group["prediction"], average="macro")
            ),
        }

    evaluation_iterations = classifier.n_iter_.tolist()
    evaluation_feature_count = len(vectorizer.feature_names_)
    del x_train, x_test, classifier, vectorizer

    final_vocabulary: Dict[str, float] = {}
    for isolate in genotype_features.values():
        for feature_name in isolate["genotypes"]:
            final_vocabulary[feature_name] = 1.0
        final_vocabulary["organism::{}".format(isolate["organism"])] = 1.0
    for antibiotic in joined["antibiotic"].unique():
        final_vocabulary["antibiotic::{}".format(antibiotic)] = 1.0

    final_vectorizer = DictVectorizer(sparse=True, dtype=np.float64)
    final_vectorizer.fit([final_vocabulary])
    del final_vocabulary
    all_features = transform_in_batches(
        final_vectorizer,
        joined[["biosample", "antibiotic"]].itertuples(index=False, name=None),
        genotype_features,
        args.batch_size,
    )
    final_classifier = LogisticRegression(
        solver="saga",
        class_weight="balanced",
        max_iter=args.max_iter,
        tol=1e-3,
        random_state=args.random_state,
    )
    final_classifier.fit(all_features, joined["target"])
    del all_features

    artifact = {
        "vectorizer": final_vectorizer,
        "classifier": final_classifier,
        "feature_columns": [
            "organism",
            "antibiotic",
            "AMR genotypes (gene presence and call state)",
        ],
        "target_classes": ["R", "S"],
        "metrics": metrics,
        "per_antibiotic_test_metrics": per_antibiotic,
        "split": {
            "strategy": (
                "GroupShuffleSplit by SNP cluster; BioSample used when "
                "SNP cluster is missing"
            ),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "train_rows": int(len(train_frame)),
            "test_rows": int(len(test_frame)),
            "train_biosamples": int(train_frame["biosample"].nunique()),
            "test_biosamples": int(test_frame["biosample"].nunique()),
            "train_split_groups": len(train_groups),
            "test_split_groups": len(test_groups),
        },
        "data_counts": {**ast_counts, **isolate_counts},
        "training": {
            "algorithm": "class-balanced logistic regression (SAGA)",
            "max_iter": args.max_iter,
            "fit_rows": int(len(joined)),
            "fit_biosamples": int(joined["biosample"].nunique()),
            "n_iter": final_classifier.n_iter_.tolist(),
            "evaluation_model_n_iter": evaluation_iterations,
            "feature_count": len(final_vectorizer.feature_names_),
            "evaluation_model_feature_count": evaluation_feature_count,
            "input_sources": {
                "isolates": str(args.isolates.resolve()),
                "ast": str(args.ast.resolve()),
            },
        },
        "scikit_learn_version": sklearn.__version__,
        "note": (
            "Research baseline using precomputed AMR genotypes. Not validated "
            "for clinical treatment decisions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)

    summary = {
        "data_counts": artifact["data_counts"],
        "joined_rows": len(joined),
        "joined_biosamples": int(joined["biosample"].nunique()),
        "antibiotics": int(joined["antibiotic"].nunique()),
        "split": artifact["split"],
        "feature_count": artifact["training"]["feature_count"],
        "n_iter": artifact["training"]["n_iter"],
        "final_model_fit_rows": artifact["training"]["fit_rows"],
        "metrics": metrics,
        "per_antibiotic_test_metrics": per_antibiotic,
        "saved_model": str(args.output),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
