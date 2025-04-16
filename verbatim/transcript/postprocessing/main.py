import argparse
import json
import logging
from pathlib import Path

from verbatim.transcript.postprocessing.config import Config
from verbatim.transcript.postprocessing.processor import DiarizationProcessor
from verbatim.eval.diarizationlm.metrics import calculate_metrics, format_metrics, format_improvements

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process diarized transcripts with LLM")
    parser.add_argument("input_json", type=Path, help="Path to input JSON file")
    parser.add_argument("--output-json", type=Path, help="Path to output JSON file")
    parser.add_argument("--ref-json", type=Path, help="Path to reference JSON file for evaluation")
    parser.add_argument("--chunk-size", type=int, default=3, help="Number of utterances per chunk")
    parser.add_argument("--model", type=str, default=Config.MODEL_NAME, help="Name of the Ollama model to use")

    args = parser.parse_args()

    output_path = args.output_json or args.input_json.with_suffix(".dlm.json")

    # Initialize configuration and processor
    config = Config()
    if args.model:
        config.MODEL_NAME = args.model

    processor = DiarizationProcessor(config)

    # Load input data
    with open(args.input_json) as f:
        input_data = json.load(f)

    # Process with LLM
    print("\nProcessing JSON with LLM...")
    processed_data = processor.process_json(input_path=args.input_json, output_path=output_path, chunk_size=args.chunk_size)

    # Evaluate if reference provided
    if args.ref_json:
        with open(args.ref_json) as f:
            ref_data = json.load(f)

        print("\nEvaluating results...")
        before_metrics = calculate_metrics(input_data, ref_data)
        after_metrics = calculate_metrics(processed_data, ref_data)

        print(format_metrics(before_metrics, prefix="Before LLM postprocessing"))
        print(format_metrics(after_metrics, prefix="After LLM postprocessing"))
        print(format_improvements(before_metrics, after_metrics))


if __name__ == "__main__":
    main()
