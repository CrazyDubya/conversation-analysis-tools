#!/usr/bin/env python3
"""Generate PNG screenshots from terminal command output."""

import subprocess
import re
from PIL import Image, ImageDraw, ImageFont
import os

# Terminal color scheme (dark theme)
COLORS = {
    'background': (30, 30, 30),
    'text': (204, 204, 204),
    'green': (78, 201, 176),
    'red': (244, 71, 71),
    'yellow': (229, 192, 123),
    'blue': (97, 175, 239),
    'magenta': (198, 120, 221),
    'cyan': (86, 182, 194),
    'white': (255, 255, 255),
}

def strip_ansi(text):
    """Remove ANSI escape codes from text."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def create_terminal_screenshot(output_text, filename, title="Terminal Output"):
    """Create a PNG image that looks like a terminal screenshot."""

    # Clean the text
    clean_text = strip_ansi(output_text)
    lines = clean_text.split('\n')

    # Calculate dimensions
    try:
        # Try to use a monospace font
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 14)
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 14)
    except:
        font = ImageFont.load_default()
        title_font = font

    # Get character dimensions
    char_width = 8
    char_height = 18

    # Calculate image size
    max_line_length = max(len(line) for line in lines) if lines else 80
    padding = 20
    title_bar_height = 30

    width = max(max_line_length * char_width + padding * 2, 600)
    height = len(lines) * char_height + padding * 2 + title_bar_height

    # Create image
    img = Image.new('RGB', (width, height), COLORS['background'])
    draw = ImageDraw.Draw(img)

    # Draw title bar
    draw.rectangle([0, 0, width, title_bar_height], fill=(50, 50, 50))

    # Draw window buttons (macOS style)
    button_y = title_bar_height // 2
    draw.ellipse([10, button_y - 6, 22, button_y + 6], fill=(255, 95, 86))  # Close
    draw.ellipse([30, button_y - 6, 42, button_y + 6], fill=(255, 189, 46))  # Minimize
    draw.ellipse([50, button_y - 6, 62, button_y + 6], fill=(39, 201, 63))   # Maximize

    # Draw title
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_width) // 2, 7), title, fill=COLORS['white'], font=title_font)

    # Draw text content with basic syntax highlighting
    y = title_bar_height + padding
    for line in lines:
        x = padding

        # Determine color based on content
        if line.strip().startswith('✅') or 'PASSED' in line or 'success' in line.lower():
            color = COLORS['green']
        elif line.strip().startswith('❌') or 'FAIL' in line or 'error' in line.lower():
            color = COLORS['red']
        elif line.strip().startswith('🧪') or line.strip().startswith('💡'):
            color = COLORS['cyan']
        elif line.strip().startswith('⏱️') or line.strip().startswith('📊'):
            color = COLORS['yellow']
        elif line.strip().startswith('🚀') or line.strip().startswith('🐌'):
            color = COLORS['magenta']
        elif '=' * 10 in line:
            color = COLORS['blue']
        elif line.strip().startswith('$'):
            color = COLORS['green']
        else:
            color = COLORS['text']

        draw.text((x, y), line, fill=color, font=font)
        y += char_height

    # Save image
    img.save(filename, 'PNG')
    print(f"Created: {filename}")
    return filename


def run_command(cmd):
    """Run a command and capture output."""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        return result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return "Command timed out"
    except Exception as e:
        return f"Error: {e}"


def main():
    """Generate all screenshots."""
    screenshots_dir = "/home/user/WikipediaANIReview/screenshots"
    os.makedirs(screenshots_dir, exist_ok=True)

    os.chdir("/home/user/WikipediaANIReview")

    # 1. Async Test Suite
    print("Capturing async test suite...")
    output = run_command("python test_async.py 2>&1")
    create_terminal_screenshot(
        output,
        f"{screenshots_dir}/01_async_tests.png",
        "Async Test Suite Results"
    )

    # 2. Performance Demo
    print("Capturing performance demo...")
    output = run_command("timeout 30 python demo_async_performance.py 2>&1 || true")
    create_terminal_screenshot(
        output,
        f"{screenshots_dir}/02_performance_demo.png",
        "Performance Demo"
    )

    # 3. Database Initialization
    print("Capturing database init...")
    output = "$ python ani_review.py init\n" + run_command("python ani_review.py init 2>&1")
    create_terminal_screenshot(
        output,
        f"{screenshots_dir}/03_database_init.png",
        "Database Initialization"
    )

    # 4. CLI Help - Main
    print("Capturing CLI help...")
    output = "$ python ani_review.py --help\n" + run_command("python ani_review.py --help 2>&1")
    create_terminal_screenshot(
        output,
        f"{screenshots_dir}/04_cli_help_main.png",
        "Main CLI Help"
    )

    # 5. CLI Help - Async
    print("Capturing async CLI help...")
    output = "$ python ani_review_async.py --help\n" + run_command("python ani_review_async.py --help 2>&1")
    create_terminal_screenshot(
        output,
        f"{screenshots_dir}/05_cli_help_async.png",
        "Async CLI Help"
    )

    # 6. Report Command
    print("Capturing report...")
    output = "$ python ani_review.py report\n" + run_command("python ani_review.py report 2>&1")
    create_terminal_screenshot(
        output,
        f"{screenshots_dir}/06_report_output.png",
        "Analysis Report"
    )

    # 7. NO GIL Report
    print("Capturing NO GIL report...")
    output = run_command("python ani_review_async.py nogil-report 2>&1")
    create_terminal_screenshot(
        output,
        f"{screenshots_dir}/07_nogil_report.png",
        "NO GIL Opportunity Analysis"
    )

    print("\nAll screenshots generated!")
    print(f"Location: {screenshots_dir}/")


if __name__ == "__main__":
    main()
