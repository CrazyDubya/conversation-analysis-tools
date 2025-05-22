
import json

def sample_start(file_path, size=1000):
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        return f.read(size)

# Sample Claude file
try:
    claude_sample = sample_start('/Users/pup/Desktop/Arch/claude/conversations.json')
    print("=== CLAUDE SAMPLE (FIRST 300 CHARS) ===")
    print(claude_sample[:300])
    
    # Try to determine if it's an array or object
    claude_type = "array" if claude_sample.strip().startswith("[") else "object" if claude_sample.strip().startswith("{") else "unknown"
    print(f"Claude file appears to be a JSON {claude_type}")
except Exception as e:
    print(f"Error sampling Claude file: {e}")

print("\n" + "="*50 + "\n")

# Sample ChatGPT file
try:
    chatgpt_sample = sample_start('/Users/pup/Desktop/Arch/chatgpt/conversations.json')
    print("=== CHATGPT SAMPLE (FIRST 300 CHARS) ===")
    print(chatgpt_sample[:300])
    
    # Try to determine if it's an array or object
    chatgpt_type = "array" if chatgpt_sample.strip().startswith("[") else "object" if chatgpt_sample.strip().startswith("{") else "unknown"
    print(f"ChatGPT file appears to be a JSON {chatgpt_type}")
except Exception as e:
    print(f"Error sampling ChatGPT file: {e}")

# Write the full samples to output files
with open('/Users/pup/Desktop/Arch/claude_sample.txt', 'w', encoding='utf-8', errors='replace') as f:
    f.write(claude_sample)
    
with open('/Users/pup/Desktop/Arch/chatgpt_sample.txt', 'w', encoding='utf-8', errors='replace') as f:
    f.write(chatgpt_sample)
    
print("\nSamples written to claude_sample.txt and chatgpt_sample.txt")
