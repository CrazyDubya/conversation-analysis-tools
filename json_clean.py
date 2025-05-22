"""
JSON Sanitization Utility

This script can be used to perform an initial sanity check and validation
on JSON files before attempting to parse them with the main parser.
"""

import json
import sys
import os
import re


def validate_json_file(file_path):
    """
    Check if a file contains valid JSON.

    Args:
        file_path: Path to the JSON file

    Returns:
        tuple: (is_valid, error_message)
    """
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            # Just check first few bytes to detect array or object structure
            first_char = f.read(1).strip()
            if first_char not in ["[", "{"]:
                return False, f"File must start with '[' or '{{', got '{first_char}'"

            # Reset and try to parse the full file
            f.seek(0)
            json.load(f)
            return True, "File contains valid JSON"
    except json.JSONDecodeError as e:
        return False, f"Invalid JSON: {str(e)}"
    except Exception as e:
        return False, f"Error checking file: {str(e)}"


def examine_file_structure(file_path):
    """
    Examine a file to determine its structure and characteristics.

    Args:
        file_path: Path to the JSON file

    Returns:
        dict: Information about the file structure
    """
    info = {
        "file_path": file_path,
        "file_size": os.path.getsize(file_path),
        "first_char": None,
        "structure": "unknown",
        "array_items": 0,
        "sample_lines": [],
        "has_claude_markers": False,
        "has_chatgpt_markers": False,
    }

    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            # Get first character to determine basic structure
            info["first_char"] = f.read(1).strip()
            if info["first_char"] == "[":
                info["structure"] = "array"
            elif info["first_char"] == "{":
                info["structure"] = "object"

            # Reset and scan for key markers
            f.seek(0)
            line_count = 0
            brace_level = 0

            for line in f:
                line_count += 1

                # Add sample lines (up to 10)
                if line_count <= 10:
                    info["sample_lines"].append(line.strip())

                # Check for platform markers
                if '"chat_messages":' in line:
                    info["has_claude_markers"] = True
                if '"mapping":' in line:
                    info["has_chatgpt_markers"] = True

                # Count opening/closing braces to estimate array items
                if info["structure"] == "array":
                    for char in line:
                        if char == "{":
                            if brace_level == 0:
                                info["array_items"] += 1
                            brace_level += 1
                        elif char == "}":
                            brace_level -= 1

    except Exception as e:
        info["error"] = str(e)

    return info


def fix_json_array(file_path, output_path=None):
    """
    Attempt to fix common issues in JSON array files.

    Args:
        file_path: Path to the JSON file to fix
        output_path: Path to save the fixed file (default: adds _fixed suffix)

    Returns:
        tuple: (success, message, fixed_path)
    """
    if output_path is None:
        base, ext = os.path.splitext(file_path)
        output_path = f"{base}_fixed{ext}"

    try:
        # Check if file starts with '[' and ends with ']'
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            first_char = f.read(1)
            if first_char != "[":
                return False, "File doesn't start with '['", None

            # Check the last character
            f.seek(0, os.SEEK_END)
            pos = f.tell() - 1
            while pos > 0:
                f.seek(pos)
                char = f.read(1)
                if not char.isspace():
                    last_char = char
                    break
                pos -= 1

            if last_char != "]":
                return False, "File doesn't end with ']'", None

        # Process the file to fix common issues
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            with open(output_path, "w", encoding="utf-8") as out:
                # Start the array
                out.write("[\n")

                # Variables to track parsing state
                buffer = ""
                item_count = 0
                brace_level = 0
                in_string = False
                escape_next = False

                # Skip the opening '['
                f.read(1)

                # Process character by character
                while True:
                    char = f.read(1)
                    if not char:  # End of file
                        break

                    # Update parsing state
                    if escape_next:
                        escape_next = False
                        buffer += char
                    elif char == "\\":
                        escape_next = True
                        buffer += char
                    elif char == '"':
                        in_string = not in_string
                        buffer += char
                    elif not in_string:
                        if char == "{":
                            brace_level += 1
                            buffer += char
                        elif char == "}":
                            brace_level -= 1
                            buffer += char

                            # If we've reached the end of an object
                            if brace_level == 0:
                                # Write the complete object
                                out.write(buffer)
                                item_count += 1

                                # Add comma if not the last item
                                next_non_space = ""
                                pos = f.tell()
                                while True:
                                    next_char = f.read(1)
                                    if not next_char:
                                        break
                                    if not next_char.isspace():
                                        next_non_space = next_char
                                        break

                                f.seek(pos)  # Reset position

                                if next_non_space and next_non_space != "]":
                                    out.write(",\n")
                                else:
                                    out.write("\n")

                                buffer = ""
                        elif char in [",", "\n", " ", "\t", "\r"]:
                            # Only add commas between items
                            if brace_level > 0:
                                buffer += char
                        else:
                            buffer += char
                    else:
                        buffer += char

                # Close the array
                out.write("]")

        # Verify the fixed file
        valid, error = validate_json_file(output_path)
        if valid:
            return (
                True,
                f"Successfully fixed and validated JSON file. Items: {item_count}",
                output_path,
            )
        else:
            return False, f"Fixed file is still invalid: {error}", output_path

    except Exception as e:
        return False, f"Error fixing JSON file: {str(e)}", None


def main():
    """Main function for running as a command-line script."""
    if len(sys.argv) < 2:
        print("Usage: python json_sanitize.py <command> <file_path> [output_path]")
        print("Commands:")
        print("  validate    - Check if file contains valid JSON")
        print("  examine     - Examine file structure and characteristics")
        print("  fix-array   - Attempt to fix issues in a JSON array file")
        return

    command = sys.argv[1]

    if len(sys.argv) < 3:
        print("Error: File path is required")
        return

    file_path = sys.argv[2]

    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        return

    if command == "validate":
        valid, message = validate_json_file(file_path)
        print(f"JSON valid: {valid}")
        print(f"Message: {message}")

    elif command == "examine":
        info = examine_file_structure(file_path)
        print("FILE STRUCTURE INFORMATION:")
        print(f"File path: {info['file_path']}")
        print(f"File size: {info['file_size']} bytes")
        print(f"First character: {info['first_char']}")
        print(f"Structure: {info['structure']}")
        print(f"Estimated array items: {info['array_items']}")
        print(f"Has Claude markers: {info['has_claude_markers']}")
        print(f"Has ChatGPT markers: {info['has_chatgpt_markers']}")
        print("\nSample lines:")
        for i, line in enumerate(info["sample_lines"]):
            print(f"{i + 1}: {line[:100]}{'...' if len(line) > 100 else ''}")

    elif command == "fix-array":
        output_path = sys.argv[3] if len(sys.argv) > 3 else None
        success, message, fixed_path = fix_json_array(file_path, output_path)
        print(f"Success: {success}")
        print(f"Message: {message}")
        if fixed_path:
            print(f"Fixed file: {fixed_path}")

    else:
        print(f"Unknown command: {command}")


if __name__ == "__main__":
    main()
