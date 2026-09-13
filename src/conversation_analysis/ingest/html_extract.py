# Ported from chatgpt-archive-clean/chatgptarchive/data_extraction.py @ f9b2b636 (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.
# Changes vs original: verbatim; requires the `regex` package (see requirements.txt)

import json
import logging
import argparse
import regex
import html

def extract_json_with_balanced_brackets(html_file_path, output_json_path):
    try:
        with open(html_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Regex pattern to match var jsonData = [ ... ];
        pattern = r'var\s+jsonData\s*=\s*(\[(?:[^][]+|(?1))*\]);'

        match = regex.search(pattern, content, regex.DOTALL)
        if not match:
            logging.error("Could not find 'var jsonData' with balanced brackets.")
            return

        json_text = match.group(1)

        # Unescape HTML entities
        json_text = html.unescape(json_text)

        # Remove trailing commas
        json_text = regex.sub(r',\s*([\]}])', r'\1', json_text)

        # Validate JSON
        json_text = json_text.strip()
        if not (json_text.startswith('[') and json_text.endswith(']')):
            logging.error("Extracted JSON does not start with '[' and end with ']'.")
            return

        # Parse JSON
        try:
            conversations = json.loads(json_text)
        except json.JSONDecodeError as e:
            logging.error(f"JSON decoding failed: {e}")
            with open('raw_chat.json', 'w', encoding='utf-8') as raw_file:
                raw_file.write(json_text)
            logging.info("Raw JSON data saved to 'raw_chat.json' for manual inspection.")
            return

        # Save to output file
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(conversations, outfile, ensure_ascii=False, indent=4)

        logging.info(f"Successfully parsed and saved JSON to {output_json_path}")

    except Exception as e:
        logging.error(f"Error parsing HTML: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract JSON data from HTML with balanced brackets.')
    parser.add_argument('--input', required=True, help='Path to input HTML file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
    extract_json_with_balanced_brackets(args.input, args.output)
