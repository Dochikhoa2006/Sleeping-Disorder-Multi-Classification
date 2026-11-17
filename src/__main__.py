"""Discover and launch project tools without eagerly loading ML dependencies."""

import argparse
from importlib import import_module
import sys

COMMANDS = {
    "predict": ("predict", "Predict classes for patient inputs"),
    "train": ("train", "Train and compare classification models"),
    "evaluate": ("evaluate", "Evaluate a model on unseen labeled data"),
    "audit": ("audit", "Validate every patient record without a model"),
    "convert": ("convert", "Convert CSV, JSON, and JSON Lines inputs"),
    "doctor": ("doctor", "Check installation and model availability"),
    "diary": ("diary", "Record and summarize a local sleep diary"),
    "web": ("web", "Start the local browser app and JSON API"),
    "model-info": ("model_info", "Inspect a trusted saved model"),
    "generate": ("generate", "Generate marked synthetic patient fixtures"),
    "split": ("split_data", "Create profile-disjoint train/test CSVs"),
    "verify-split": ("verify_split", "Verify split integrity and separation"),
    "merge": ("merge_data", "Merge validated CSV batches"),
    "sample": ("sample_data", "Sample complete patient-profile groups"),
    "deduplicate": ("deduplicate", "Remove equivalent CSV profiles"),
    "overlap": ("overlap", "Audit shared profiles across datasets"),
    "label-audit": ("label_audit", "Find conflicting labels on equivalent profiles"),
    "coverage": ("coverage", "Check fitted categorical encoder coverage"),
    "importance": ("importance", "Measure raw-feature permutation importance"),
    "drift": ("drift", "Compare patient input distributions"),
    "compare-models": ("compare_models", "Compare predictions from saved models"),
    "compare-exports": ("compare_exports", "Compare saved prediction CSVs by ID"),
    "compare-diaries": ("compare_diaries", "Preview changes between diary backups"),
    "receipt": ("receipt", "Create prediction provenance receipts"),
    "verify-receipt": ("verify_receipt", "Verify and replay a prediction receipt"),
    "score-exports": ("score_exports", "Score predictions against supplied actual labels"),
    "evaluation-report": ("evaluation_report", "Render evaluation JSON as printable HTML"),
    "benchmark": ("benchmark", "Benchmark model inference on synthetic inputs"),
    "schema": ("schema", "Export the patient input JSON Schema"),
    "clean-batch": ("clean_batch", "Export valid rows with a rejection report"),
}


def main() -> None:
    listing = "Commands:\n" + "\n".join(f"  {name:20} {description}" for name, (_, description) in COMMANDS.items())
    parser = argparse.ArgumentParser(description=__doc__, epilog=listing,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=COMMANDS, metavar="COMMAND")
    parser.add_argument("arguments", nargs=argparse.REMAINDER, metavar="ARG")
    if len(sys.argv) == 1:
        parser.print_help()
        return
    args = parser.parse_args()
    module_name = COMMANDS[args.command][0]
    original = sys.argv
    try:
        sys.argv = [f"python -m src {args.command}", *args.arguments]
        import_module(f"src.{module_name}").main()
    finally:
        sys.argv = original


if __name__ == "__main__":
    main()
