import os

# Define file paths
claude_path = '/Users/pup/Desktop/Arch/claude/conversations.json'
chatgpt_path = '/Users/pup/Desktop/Arch/chatgpt/conversations.json'
output_path = '/Users/pup/Desktop/Arch/json_samples.txt'

# Sample size for each section (characters)
SAMPLE_SIZE = 5000


def get_file_size(filepath):
    """Get the size of a file in bytes"""
    return os.path.getsize(filepath)


def read_file_section(filepath, start_pos, length):
    """Read a section of a file from a specific position"""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        f.seek(start_pos)
        return f.read(length)


def main():
    # Get file sizes
    claude_size = get_file_size(claude_path)
    chatgpt_size = get_file_size(chatgpt_path)

    # Calculate positions for samples
    claude_positions = {
        'begin': 0,
        'middle': max(0, claude_size // 2 - SAMPLE_SIZE // 2),
        'end': max(0, claude_size - SAMPLE_SIZE)
    }

    chatgpt_positions = {
        'begin': 0,
        'middle': max(0, chatgpt_size // 2 - SAMPLE_SIZE // 2),
        'end': max(0, chatgpt_size - SAMPLE_SIZE)
    }

    # Read samples
    claude_samples = {
        'begin': read_file_section(claude_path, claude_positions['begin'], SAMPLE_SIZE),
        'middle': read_file_section(claude_path, claude_positions['middle'], SAMPLE_SIZE),
        'end': read_file_section(claude_path, claude_positions['end'], SAMPLE_SIZE)
    }

    chatgpt_samples = {
        'begin': read_file_section(chatgpt_path, chatgpt_positions['begin'], SAMPLE_SIZE),
        'middle': read_file_section(chatgpt_path, chatgpt_positions['middle'], SAMPLE_SIZE),
        'end': read_file_section(chatgpt_path, chatgpt_positions['end'], SAMPLE_SIZE)
    }

    # Write all samples to output file
    with open(output_path, 'w', encoding='utf-8', errors='replace') as out:
        # Claude samples
        out.write('=' * 80 + '\n')
        out.write(f'CLAUDE CONVERSATIONS.JSON (Total size: {claude_size} bytes)\n')
        out.write('=' * 80 + '\n\n')

        out.write('-' * 40 + '\n')
        out.write('BEGIN SAMPLE:\n')
        out.write('-' * 40 + '\n')
        out.write(claude_samples['begin'])
        out.write('\n\n')

        out.write('-' * 40 + '\n')
        out.write('MIDDLE SAMPLE (position approx. {}):\n'.format(claude_positions['middle']))
        out.write('-' * 40 + '\n')
        out.write(claude_samples['middle'])
        out.write('\n\n')

        out.write('-' * 40 + '\n')
        out.write('END SAMPLE (position approx. {}):\n'.format(claude_positions['end']))
        out.write('-' * 40 + '\n')
        out.write(claude_samples['end'])
        out.write('\n\n')

        # ChatGPT samples
        out.write('=' * 80 + '\n')
        out.write(f'CHATGPT CONVERSATIONS.JSON (Total size: {chatgpt_size} bytes)\n')
        out.write('=' * 80 + '\n\n')

        out.write('-' * 40 + '\n')
        out.write('BEGIN SAMPLE:\n')
        out.write('-' * 40 + '\n')
        out.write(chatgpt_samples['begin'])
        out.write('\n\n')

        out.write('-' * 40 + '\n')
        out.write('MIDDLE SAMPLE (position approx. {}):\n'.format(chatgpt_positions['middle']))
        out.write('-' * 40 + '\n')
        out.write(chatgpt_samples['middle'])
        out.write('\n\n')

        out.write('-' * 40 + '\n')
        out.write('END SAMPLE (position approx. {}):\n'.format(chatgpt_positions['end']))
        out.write('-' * 40 + '\n')
        out.write(chatgpt_samples['end'])

    print(f"Samples extracted and saved to {output_path}")


if __name__ == "__main__":
    main()