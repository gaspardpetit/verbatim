import argparse
import json
import logging

from verbatim.eval.diarization_metrics import calculate_metrics, format_metrics

LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Compare hypothesis JSON file with reference JSON file and calculate metrics")
    parser.add_argument("ref_json", help="Path to reference JSON file")
    parser.add_argument("hyp_json", help="Path to hypothesis JSON file")
    args = parser.parse_args()

    # Load reference data
    with open(args.ref_json, "r", encoding="utf-8") as f:
        ref_data = json.load(f)

    # Load hypothesis data
    with open(args.hyp_json, "r", encoding="utf-8") as f:
        hyp_data = json.load(f)

    # Calculate metrics
    print("Calculating metrics...")
    metrics = calculate_metrics(hyp_data.copy(), ref_data)
    print(format_metrics(metrics))


if __name__ == "__main__":
    main()
