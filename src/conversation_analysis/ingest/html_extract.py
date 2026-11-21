# Ported from chatgpt-archive-clean/chatgptarchive/data_extraction.py @ f9b2b636 (master) during the conversation-analysis
# consolidation (2026-09-12). See docs/PROVENANCE.md.

import argparse
import html
import json
import logging
import regex
import sys


def _assigned_array(text):
    """Read the assigned array without treating brackets in strings as syntax."""
    text = text.lstrip()
    if not text.startswith('['):
        raise ValueError("jsonData must be an array")
    depth = 0
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if quoted:
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == '[':
            depth += 1
        elif character == ']':
            depth -= 1
            if depth == 0:
                if not text[index + 1:].lstrip().startswith(';'):
                    raise ValueError("jsonData array must end with a semicolon")
                return text[:index + 1]
    raise ValueError("Could not find the end of the jsonData array")


def _without_trailing_commas(text):
    """Remove structural trailing commas only, leaving JSON strings untouched."""
    result = []
    quoted = False
    escaped = False
    for index, character in enumerate(text):
        if quoted:
            result.append(character)
            if escaped:
                escaped = False
            elif character == '\\':
                escaped = True
            elif character == '"':
                quoted = False
            continue
        if character == '"':
            quoted = True
        elif character == ',':
            following = index + 1
            while following < len(text) and text[following].isspace():
                following += 1
            if following < len(text) and text[following] in '}]':
                continue
        result.append(character)
    return ''.join(result)


def extract_json_with_balanced_brackets(html_file_path, output_json_path):
    try:
        with open(html_file_path, 'r', encoding='utf-8') as source:
            content = source.read()
        match = regex.search(r'var\s+jsonData\s*=\s*', content)
        if not match:
            logging.error("Could not find 'var jsonData'.")
            return False
        assigned = content[match.end():]
        last_json = None
        conversations = None
        # Literal script JSON takes precedence. Retain legacy entity recovery
        # only when the literal form could not be parsed.
        for entity_recovery, candidate in ((False, assigned), (True, html.unescape(assigned))):
            if entity_recovery and candidate == assigned:
                continue
            try:
                json_text = _assigned_array(candidate)
            except ValueError:
                continue
            last_json = json_text
            try:
                conversations = json.loads(json_text)
                recovered_commas = False
            except json.JSONDecodeError:
                try:
                    conversations = json.loads(_without_trailing_commas(json_text))
                    recovered_commas = True
                except json.JSONDecodeError:
                    continue
            if entity_recovery or recovered_commas:
                logging.warning("Used legacy HTML-entity or trailing-comma recovery.")
            break
        if conversations is None:
            logging.error("Could not parse the jsonData array.")
            if last_json is not None:
                with open('raw_chat.json', 'w', encoding='utf-8') as raw_file:
                    raw_file.write(last_json)
                logging.info("Raw JSON data saved to 'raw_chat.json' for manual inspection.")
            return False
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(conversations, outfile, ensure_ascii=False, indent=4)
        logging.info(f"Successfully parsed and saved JSON to {output_json_path}")
        return True
    except Exception as error:
        logging.error(f"Error parsing HTML: {error}")
        return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract JSON data from HTML with balanced brackets.')
    parser.add_argument('--input', required=True, help='Path to input HTML file')
    parser.add_argument('--output', required=True, help='Path to output JSON file')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
    sys.exit(0 if extract_json_with_balanced_brackets(args.input, args.output) else 1)
