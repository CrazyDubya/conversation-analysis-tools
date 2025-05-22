
import sys
import json

# Read the first 5000 bytes of the file
with open('/Users/pup/Desktop/Arch/claude/conversations.json', 'rb') as f:
    data = f.read(5000)
    
# Write to a sample file
with open('/Users/pup/Desktop/Arch/claude_sample.txt', 'wb') as f:
    f.write(data)

# Read the first 5000 bytes of the ChatGPT file
with open('/Users/pup/Desktop/Arch/chatgpt/conversations.json', 'rb') as f:
    data = f.read(5000)
    
# Write to a sample file
with open('/Users/pup/Desktop/Arch/chatgpt_sample.txt', 'wb') as f:
    f.write(data)

print("Sample files created successfully")
