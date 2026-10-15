"""Join extracted CARD features to AST labels and train an isolate-grouped model."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

TARGET_CLASSES = ("R", "S")
GENOMIC_FEATURE_PREFIXES = (
    "gene__",
    "family__",
    "model__",
    "status__",
    "mutation__",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Join CARD RGI-derived features to measured R/S AST labels and train "
            "a model with isolates held out as groups."
        )
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("data/genome_antibiotic_features.csv"),
        help="Feature matrix from extract_features.py.",
    )
    parser.add_argument(
        "--ast",
        type=Path,
        default=Path("data/ast_labels.csv"),
        help="AST CSV containing isolate_id, antibiotic, and phenotype.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models/resistance_model.joblib"),
    )
    parser.add_argument("--id-column", default="isolate_id")
    parser.add_argument("--organism-column", default="organism")
    parser.add_argument("--antibiotic-column", default="antibiotic")
    parser.add_argument("--target-column", default="phenotype")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_and_join_data(
    features_path: Path,
    ast_path: Path,
    id_column: str,
    organism_column: str,
    antibiotic_column: str,
    target_column: str,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, list, int, int]:
    if not features_path.is_file():
        raise FileNotFoundError(
            "Feature matrix not found: {}. Run src/extract_features.py first.".format(
                features_path
            )
        )
    if not ast_path.is_file():
        raise FileNotFoundError("AST labels CSV not found: {}".format(ast_path))

    feature_frame = pd.read_csv(features_path, dtype={id_column: str})
    ast_frame = pd.read_csv(ast_path, dtype={id_column: str})
    feature_required = {id_column, organism_column}
    missing_feature_columns = feature_required.difference(feature_frame.columns)
    if missing_feature_columns:
        raise ValueError(
            "Feature matrix is missing columns: {}".format(
                ", ".join(sorted(missing_feature_columns))
            )
        )
    ast_required = {id_column, antibiotic_column, target_column}
    missing_ast_columns = ast_required.difference(ast_frame.columns)
    if missing_ast_columns:
        raise ValueError(
            "AST CSV is missing columns: {}".format(
                ", ".join(sorted(missing_ast_columns))
            )
        )
    if feature_frame.empty:
        raise ValueError("Feature matrix contains no isolates.")
    if ast_frame.empty:
        raise ValueError("AST CSV contains no rows.")

    for column in (id_column, organism_column):
        feature_frame[column] = feature_frame[column].fillna("").astype(str).str.strip()
        if feature_frame[column].eq("").any():
            raise ValueError(
                "Feature matrix column {!r} contains empty values.".format(column)
            )
    if feature_frame[id_column].duplicated().any():
        raise ValueError(
            "Feature matrix must contain exactly one row per isolate; duplicate "
            "identifiers found."
        )

    genomic_columns = [
        column
        for column in feature_frame.columns
        if column.startswith(GENOMIC_FEATURE_PREFIXES)
    ]
    if not genomic_columns:
        raise ValueError(
            "Feature matrix has no genomic columns with recognized prefixes: {}."
            .format(", ".join(GENOMIC_FEATURE_PREFIXES))
        )
    for column in genomic_columns:
        feature_frame[column] = pd.to_numeric(feature_frame[column], errors="raise")
        if feature_frame[column].isna().any():
            raise ValueError(
                "Genomic feature {!r} contains missing values.".format(column)
            )
        if not feature_frame[column].isin([0, 1]).all():
            raise ValueError(
                "Genomic feature {!r} must contain only 0 or 1.".format(column)
            )

    ast_frame[id_column] = ast_frame[id_column].fillna("").astype(str).str.strip()
    ast_frame[antibiotic_column] = (
        ast_frame[antibiotic_column].fillna("").astype(str).str.strip()
    )
    ast_frame[target_column] = (
        ast_frame[target_column].fillna("").astype(str).str.strip().str.upper()
    )
    if (
        ast_frame[id_column].eq("").any()
        or ast_frame[antibiotic_column].eq("").any()
        or ast_frame[target_column].eq("").any()
    ):
        raise ValueError(
            "AST isolate, antibiotic, and phenotype values must not be empty."
        )

    unsupported_label_rows = int(
        (~ast_frame[target_column].isin(TARGET_CLASSES)).sum()
    )
    ast_frame = ast_frame[ast_frame[target_column].isin(TARGET_CLASSES)].copy()
    if ast_frame.empty:
        raise ValueError("AST labels contain no supported R/S phenotype rows.")
    if ast_frame.duplicated([id_column, antibiotic_column]).any():
        raise ValueError(
            "AST CSV contains duplicate isolate-antibiotic pairs; resolve them "
            "before training."
        )

    known_isolates = set(feature_frame[id_column])
    unmatched_ast_rows = int((~ast_frame[id_column].isin(known_isolates)).sum())
    joined = ast_frame.merge(
        feature_frame, on=id_column, how="inner", validate="many_to_one"
    )
    if joined.empty:
        raise ValueError(
            "No AST rows matched isolate IDs in the genomic feature matrix."
        )
    if joined[target_column].nunique() < 2:
        raise ValueError("Joined AST data must contain both R and S labels.")

    feature_columns = [antibiotic_column, organism_column] + genomic_columns
    return (
        joined[feature_columns],
        joined[target_column],
        joined[id_column],
        genomic_columns,
        unmatched_ast_rows,
        unsupported_label_rows,
    )


def split_by_isolate(
    features: pd.DataFrame,
    target_values: pd.Series,
    groups: pd.Series,
    test_size: float,
    random_state: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    if not 0 < test_size < 1:
        raise ValueError("--test-size must be greater than 0 and less than 1.")
    if groups.nunique() < 2:
        raise ValueError("At least two distinct isolates are required for a group split.")

    splitter = GroupShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    train_indices, test_indices = next(
        splitter.split(features, target_values, groups=groups)
    )
    x_train = features.iloc[train_indices]
    x_test = features.iloc[test_indices]
    y_train = target_values.iloc[train_indices]
    y_test = target_values.iloc[test_indices]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise ValueError(
            "The isolate-grouped split must contain both R and S in training "
            "and test sets. Add data or use a different --random-state/test-size."
        )
    if not set(groups.iloc[train_indices]).isdisjoint(set(groups.iloc[test_indices])):
        raise RuntimeError("Isolate leakage detected between train and test sets.")

    return x_train, x_test, y_train, y_test


def build_pipeline(
    antibiotic_column: str,
    organism_column: str,
    genomic_columns: list,
    random_state: int,
) -> Pipeline:
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                [antibiotic_column, organism_column],
            ),
            ("genomic_features", "passthrough", genomic_columns),
        ],
        remainder="drop",
    )
    classifier = RandomForestClassifier(
        n_estimators=300,
        max_features="sqrt",
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )
    return Pipeline(
        steps=[("preprocessor", preprocessing), ("classifier", classifier)]
    )


def evaluate_model(
    model: Pipeline, features: pd.DataFrame, target_values: pd.Series
) -> Dict[str, Any]:
    predictions = model.predict(features)
    return {
        "accuracy": float(accuracy_score(target_values, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target_values, predictions)
        ),
        "macro_f1": float(f1_score(target_values, predictions, average="macro")),
        "classification_report": classification_report(
            target_values,
            predictions,
            labels=list(TARGET_CLASSES),
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            target_values, predictions, labels=list(TARGET_CLASSES)
        ).tolist(),
    }


def main() -> None:
    args = parse_args()
    (
        features,
        target_values,
        groups,
        genomic_columns,
        unmatched_ast_rows,
        unsupported_label_rows,
    ) = (
        load_and_join_data(
            args.features,
            args.ast,
            args.id_column,
            args.organism_column,
            args.antibiotic_column,
            args.target_column,
        )
    )
    x_train, x_test, y_train, y_test = split_by_isolate(
        features, target_values, groups, args.test_size, args.random_state
    )
    model = build_pipeline(
        args.antibiotic_column,
        args.organism_column,
        genomic_columns,
        args.random_state,
    )
    model.fit(x_train, y_train)
    metrics = evaluate_model(model, x_test, y_test)

    artifact = {
        "model": model,
        "feature_columns": features.columns.tolist(),
        "genomic_feature_columns": genomic_columns,
        "id_column": args.id_column,
        "organism_column": args.organism_column,
        "antibiotic_column": args.antibiotic_column,
        "target_column": args.target_column,
        "target_classes": list(TARGET_CLASSES),
        "metrics": metrics,
        "split": {
            "strategy": "GroupShuffleSplit by {}".format(args.id_column),
            "test_size": args.test_size,
            "random_state": args.random_state,
            "train_isolates": int(groups.loc[x_train.index].nunique()),
            "test_isolates": int(groups.loc[x_test.index].nunique()),
        },
        "scikit_learn_version": sklearn.__version__,
        "note": (
            "Research prototype. Not validated for clinical treatment decisions."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.output)

    print(
        json.dumps(
            {
                "matched_ast_rows": len(features),
                "unmatched_ast_rows": unmatched_ast_rows,
                "unsupported_label_rows_excluded": unsupported_label_rows,
                "isolates": int(groups.nunique()),
                "phenotype_counts": target_values.value_counts().to_dict(),
                "split": artifact["split"],
                "metrics": metrics,
            },
            indent=2,
        )
    )
    print("Saved model bundle to {}".format(args.output))


if __name__ == "__main__":
    main()
