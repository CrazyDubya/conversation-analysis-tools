
import os

# Paths to the JSON files
claude_path = '/Users/pup/Desktop/Arch/claude/conversations.json'
chatgpt_path = '/Users/pup/Desktop/Arch/chatgpt/conversations.json'
output_path = '/Users/pup/Desktop/Arch/json_samples.txt'

# Sample size (characters)
SAMPLE_SIZE = 15000

def extract_samples(file_path, file_name):
    """Extract samples from beginning, middle, and end of a file"""
    # Get file size
    file_size = os.path.getsize(file_path)
    
    samples = {
        'begin': '',
        'middle': '',
        'end': ''
    }
    
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        # Beginning sample
        samples['begin'] = f.read(SAMPLE_SIZE)
        
        # Middle sample - seek to the middle
        middle_pos = max(0, file_size // 2 - SAMPLE_SIZE // 2)
        f.seek(middle_pos)
        # Read a line to align with JSON structure (avoid starting mid-line)
        f.readline()
        samples['middle'] = f.read(SAMPLE_SIZE)
        
        # End sample - seek to near the end
        end_pos = max(0, file_size - SAMPLE_SIZE - 1000)  # Extra 1000 chars buffer
        f.seek(end_pos)
        # Read a line to align with JSON structure
        f.readline()
        samples['end'] = f.read(SAMPLE_SIZE)
    
    return samples

# Extract samples from both files
try:
    claude_samples = extract_samples(claude_path, 'Claude')
    chatgpt_samples = extract_samples(chatgpt_path, 'ChatGPT')
    
    # Write all samples to the output file
    with open(output_path, 'w', encoding='utf-8') as out_file:
        # Claude samples
        out_file.write('=' * 80 + '\n')
        out_file.write('CLAUDE CONVERSATIONS.JSON SAMPLES\n')
        out_file.write('=' * 80 + '\n\n')
        
        out_file.write('-' * 40 + '\n')
        out_file.write('BEGIN SAMPLE:\n')
        out_file.write('-' * 40 + '\n')
        out_file.write(claude_samples['begin'])
        out_file.write('\n\n')
        
        out_file.write('-' * 40 + '\n')
        out_file.write('MIDDLE SAMPLE:\n')
        out_file.write('-' * 40 + '\n')
        out_file.write(claude_samples['middle'])
        out_file.write('\n\n')
        
        out_file.write('-' * 40 + '\n')
        out_file.write('END SAMPLE:\n')
        out_file.write('-' * 40 + '\n')
        out_file.write(claude_samples['end'])
        out_file.write('\n\n')
        
        # ChatGPT samples
        out_file.write('=' * 80 + '\n')
        out_file.write('CHATGPT CONVERSATIONS.JSON SAMPLES\n')
        out_file.write('=' * 80 + '\n\n')
        
        out_file.write('-' * 40 + '\n')
        out_file.write('BEGIN SAMPLE:\n')
        out_file.write('-' * 40 + '\n')
        out_file.write(chatgpt_samples['begin'])
        out_file.write('\n\n')
        
        out_file.write('-' * 40 + '\n')
        out_file.write('MIDDLE SAMPLE:\n')
        out_file.write('-' * 40 + '\n')
        out_file.write(chatgpt_samples['middle'])
        out_file.write('\n\n')
        
        out_file.write('-' * 40 + '\n')
        out_file.write('END SAMPLE:\n')
        out_file.write('-' * 40 + '\n')
        out_file.write(chatgpt_samples['end'])
    
    print(f"Samples extracted and saved to {output_path}")
    
except Exception as e:
    print(f"Error: {str(e)}")
