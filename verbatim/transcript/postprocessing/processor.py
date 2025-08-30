import logging
from typing import Tuple, List, Dict
from pathlib import Path
import json
import re

from openai import OpenAI

from verbatim.transcript.postprocessing.config import Config

LOG = logging.getLogger(__name__)

class DiarizationProcessor:
    def __init__(self, config: Config):
        """Initialize the processor with configuration"""
        self.config = config
        self.client = OpenAI(base_url=config.OLLAMA_BASE_URL, api_key=config.API_KEY)

    def extract_text_and_spk(self, completion: str) -> Tuple[str, str]:
        """Extract text and speaker labels from completion string"""
        spk = "1"
        previous_spk = "1"
        result_text = []
        result_spk = []

        for word in completion.split():
            if word.startswith("<speaker:"):
                if not word.endswith(">"):
                    word += ">"
                spk = word[len("<speaker:"):-len(">")]
                try:
                    spk_int = int(spk)
                    if not spk or spk_int < 1 or spk_int > 10:
                        raise ValueError(f"Unexpected speaker token: {word}")
                    previous_spk = spk
                except ValueError:
                    LOG.warning(f"Skipping meaningless speaker token: {word}")
                    spk = previous_spk
            else:
                result_text.append(word)
                result_spk.append(spk)

        return " ".join(result_text), " ".join(result_spk)

    def process_chunk(self, text: str) -> Tuple[str, str]:
        """Process a single chunk of text"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.MODEL_NAME,
                messages=[
                    {"role": "system", "content": self.config.SYSTEM_PROMPT},
                    {"role": "user", "content": f"{text} -->"}
                ],
                temperature=0.1
            )

            completion = response.choices[0].message.content or ""

            # Create and log pretty diff
            print("\nProcessing chunk:")
            print("=" * 80)
            print("Before:")
            print(text)  # Original formatted text
            print("\nAfter:")
            print(completion)

            return self.extract_text_and_spk(completion)

        except Exception as e:
            LOG.error(f"Error processing chunk: {e}")
            raise

    def clean_speaker_tag(self, tag: str) -> str:
        """Clean up speaker tags by removing repeated numbers"""
        # Extract the first number from the tag
        match = re.search(r'<speaker:(\d+)', tag)
        if match:
            number = match.group(1)
            return f"<speaker:{number}>"
        return tag

    def format_chunk(self, utterances: List[Dict]) -> str:
        """Format a chunk of utterances into diarized text"""
        # Join utterances with proper speaker tags
        text_parts = []
        for utt in utterances:
            # Clean up any repeated speaker numbers in the input
            speaker_tag = self.clean_speaker_tag(f"<speaker:{utt['hyp_spk']}>")
            text_parts.append(f"{speaker_tag} {utt['hyp_text']}")

        return "\n".join(text_parts)

    def process_json(self, input_path: Path, output_path: Path, chunk_size: int = 3) -> Dict:
        """Process entire JSON file and save results"""
        with open(input_path) as f:
            data = json.load(f)

        output_utterances = []
        current_chunk = []

        for utterance in data["utterances"]:
            current_chunk.append(utterance)

            if len(current_chunk) >= chunk_size:
                chunk_text = self.format_chunk(current_chunk)
                text, spk = self.process_chunk(chunk_text)

                output_utterances.append({
                    "utterance_id": f"utt{len(output_utterances)}",
                    "hyp_text": text,
                    "hyp_spk": spk
                })
                current_chunk = []

        # Process remaining utterances
        if current_chunk:
            chunk_text = self.format_chunk(current_chunk)
            text, spk = self.process_chunk(chunk_text)
            output_utterances.append({
                "utterance_id": f"utt{len(output_utterances)}",
                "hyp_text": text,
                "hyp_spk": spk
            })

        output_data = {"utterances": output_utterances}

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        return output_data
