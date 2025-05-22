
import json
import os

# Sample size in characters
SAMPLE_SIZE = 5000

# File paths
claude_path = '/Users/pup/Desktop/Arch/claude/conversations.json'
chatgpt_path = '/Users/pup/Desktop/Arch/chatgpt/conversations.json'
output_path = '/Users/pup/Desktop/Arch/json_samples.txt'

# Function to get file size
def get_file_size(path):
    return os.path.getsize(path)

# Function to extract samples from a file
def extract_samples(file_path, file_name):
    """Extract beginning, middle, and end samples from a file"""
    try:
        file_size = get_file_size(file_path)
        
        # Read beginning
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            begin_sample = f.read(SAMPLE_SIZE)
        
        # Read middle (seek to approximate middle)
        middle_pos = max(0, file_size // 2 - SAMPLE_SIZE // 2)
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(middle_pos)
            # Read a line to avoid starting in the middle of a line
            f.readline()
            middle_sample = f.read(SAMPLE_SIZE)
        
        # Read end
        end_pos = max(0, file_size - SAMPLE_SIZE - 1000)  # Extra buffer
        with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
            f.seek(end_pos)
            # Read a line to align
            f.readline()
            end_sample = f.read(SAMPLE_SIZE)
        
        # Check structure type
        structure_type = "Unknown"
        if begin_sample.strip().startswith('['):
            structure_type = "JSON Array"
        elif begin_sample.strip().startswith('{'):
            structure_type = "JSON Object"
        
        # Try to parse a small part to understand keys
        sample_keys = []
        try:
            if structure_type == "JSON Array" and '{' in begin_sample:
                # Find first complete object in array
                obj_start = begin_sample.find('{')
                if obj_start > -1:
                    # Simple approach to find object end - not perfect but functional for most JSON
                    bracket_level = 0
                    in_string = False
                    escape_next = False
                    obj_end = -1
                    
                    for i, char in enumerate(begin_sample[obj_start:], obj_start):
                        if escape_next:
                            escape_next = False
                            continue
                            
                        if char == '\\':
                            escape_next = True
                        elif char == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                bracket_level += 1
                            elif char == '}':
                                bracket_level -= 1
                                if bracket_level == 0:
                                    obj_end = i + 1
                                    break
                    
                    if obj_end > obj_start:
                        obj_text = begin_sample[obj_start:obj_end]
                        try:
                            obj = json.loads(obj_text)
                            sample_keys = list(obj.keys())
                        except:
                            sample_keys = ["Failed to parse keys from array item"]
            elif structure_type == "JSON Object":
                # Try to parse top level keys only
                try:
                    # Find a complete object by balance of braces
                    bracket_level = 0
                    in_string = False
                    escape_next = False
                    obj_end = -1
                    
                    for i, char in enumerate(begin_sample):
                        if escape_next:
                            escape_next = False
                            continue
                            
                        if char == '\\':
                            escape_next = True
                        elif char == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                bracket_level += 1
                            elif char == '}':
                                bracket_level -= 1
                                if bracket_level == 0:
                                    obj_end = i + 1
                                    break
                    
                    if obj_end > 0:
                        obj_text = begin_sample[:obj_end]
                        obj = json.loads(obj_text)
                        sample_keys = list(obj.keys())
                except:
                    sample_keys = ["Failed to parse keys from object"]
        except Exception as e:
            sample_keys = [f"Error analyzing keys: {str(e)}"]
            
        return {
            "file_name": file_name,
            "file_size": file_size,
            "structure_type": structure_type,
            "sample_keys": sample_keys,
            "begin_sample": begin_sample,
            "middle_sample": middle_sample,
            "end_sample": end_sample
        }
    except Exception as e:
        return {
            "file_name": file_name,
            "error": str(e)
        }

# Extract samples from both files
claude_result = extract_samples(claude_path, "Claude")
chatgpt_result = extract_samples(chatgpt_path, "ChatGPT")

# Write results to output file
with open(output_path, 'w', encoding='utf-8') as f:
    # Claude results
    f.write("=" * 80 + "\n")
    f.write(f"FILE: {claude_result['file_name']} conversations.json\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"File Size: {claude_result.get('file_size', 'ERROR')} bytes\n")
    f.write(f"Structure Type: {claude_result.get('structure_type', 'ERROR')}\n")
    f.write(f"Sample Keys: {', '.join(claude_result.get('sample_keys', ['ERROR']))}\n\n")
    
    f.write("-" * 40 + "\n")
    f.write("BEGIN SAMPLE:\n")
    f.write("-" * 40 + "\n")
    f.write(claude_result.get('begin_sample', 'ERROR READING FILE'))
    f.write("\n\n")
    
    f.write("-" * 40 + "\n")
    f.write("MIDDLE SAMPLE:\n")
    f.write("-" * 40 + "\n")
    f.write(claude_result.get('middle_sample', 'ERROR READING FILE'))
    f.write("\n\n")
    
    f.write("-" * 40 + "\n")
    f.write("END SAMPLE:\n")
    f.write("-" * 40 + "\n")
    f.write(claude_result.get('end_sample', 'ERROR READING FILE'))
    f.write("\n\n\n")
    
    # ChatGPT results
    f.write("=" * 80 + "\n")
    f.write(f"FILE: {chatgpt_result['file_name']} conversations.json\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"File Size: {chatgpt_result.get('file_size', 'ERROR')} bytes\n")
    f.write(f"Structure Type: {chatgpt_result.get('structure_type', 'ERROR')}\n")
    f.write(f"Sample Keys: {', '.join(chatgpt_result.get('sample_keys', ['ERROR']))}\n\n")
    
    f.write("-" * 40 + "\n")
    f.write("BEGIN SAMPLE:\n")
    f.write("-" * 40 + "\n")
    f.write(chatgpt_result.get('begin_sample', 'ERROR READING FILE'))
    f.write("\n\n")
    
    f.write("-" * 40 + "\n")
    f.write("MIDDLE SAMPLE:\n")
    f.write("-" * 40 + "\n")
    f.write(chatgpt_result.get('middle_sample', 'ERROR READING FILE'))
    f.write("\n\n")
    
    f.write("-" * 40 + "\n")
    f.write("END SAMPLE:\n")
    f.write("-" * 40 + "\n")
    f.write(chatgpt_result.get('end_sample', 'ERROR READING FILE'))

print(f"Samples extracted and written to {output_path}")
