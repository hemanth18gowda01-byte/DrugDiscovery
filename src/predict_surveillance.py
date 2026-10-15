"""Predict an AST resistance category from isolate metadata and AMR genotypes."""

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import joblib

from train_surveillance import parse_genotypes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Predict resistant/susceptible AST from the trained surveillance "
            "model's organism, antibiotic, and AMR genotype inputs."
        )
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/nlm_resistance_model.joblib"),
        help="Trained model bundle path.",
    )
    parser.add_argument("--organism", required=True, help="NCBI organism group.")
    parser.add_argument("--antibiotic", required=True, help="Antibiotic name.")
    parser.add_argument(
        "--genotypes",
        default="",
        help=(
            "Comma-separated NCBI AMR genotype calls, such as "
            '"blaTEM-1=COMPLETE,gyrA_S83F=POINT".'
        ),
    )
    return parser.parse_args()


def predict(
    artifact: Dict[str, Any],
    organism: str,
    antibiotic: str,
    genotype_text: str,
) -> Dict[str, Any]:
    organism = organism.strip()
    antibiotic = antibiotic.strip()
    if not organism:
        raise ValueError("--organism must not be empty.")
    if not antibiotic:
        raise ValueError("--antibiotic must not be empty.")

    genotype_features = parse_genotypes(genotype_text, "prediction")
    features = {name: 1.0 for name in genotype_features}
    features["organism::{}".format(organism)] = 1.0
    features["antibiotic::{}".format(antibiotic)] = 1.0

    matrix = artifact["vectorizer"].transform([features])
    classifier = artifact["classifier"]
    prediction = str(classifier.predict(matrix)[0])
    probabilities = classifier.predict_proba(matrix)[0]
    probability_by_class = {
        str(label): float(probability)
        for label, probability in zip(classifier.classes_, probabilities)
    }
    return {
        "prediction": prediction,
        "prediction_label": (
            "resistant" if prediction == "R" else "susceptible"
        ),
        "probabilities": probability_by_class,
        "organism": organism,
        "antibiotic": antibiotic,
        "genotype_calls_supplied": len(genotype_features) // 2,
        "note": artifact.get("note", ""),
    }


def main() -> None:
    args = parse_args()
    if not args.model.is_file():
        raise FileNotFoundError(
            "Model bundle not found: {}. Train the model first or pass --model."
            .format(args.model)
        )
    artifact = joblib.load(args.model)
    required_keys = {"vectorizer", "classifier", "target_classes"}
    missing_keys = required_keys.difference(artifact)
    if missing_keys:
        raise ValueError(
            "Model bundle is missing required fields: {}".format(
                ", ".join(sorted(missing_keys))
            )
        )

    result = predict(
        artifact, args.organism, args.antibiotic, args.genotypes
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
