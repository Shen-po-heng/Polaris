"""
Diagnostic: print short lines from a PDF to understand what text pdfplumber extracts.
Usage: python diagnose_pdf.py <path/to/paper.pdf> [max_lines]
"""
import sys
import io
import statistics
from pathlib import Path

# Force UTF-8 output on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

import pdfplumber
from approach_b_heuristic import _group_chars_by_line

def diagnose(pdf_path: str, max_lines: int = 80):
    with pdfplumber.open(pdf_path) as pdf:
        all_sizes = [
            c["size"] for page in pdf.pages
            for c in (page.chars or []) if c.get("size")
        ]
        body_size = statistics.median(all_sizes) if all_sizes else 10.0
        print(f"Body font size (median): {body_size:.2f} pt")
        print(f"Total chars with size info: {len(all_sizes)}")
        print()

        count = 0
        for page_num, page in enumerate(pdf.pages, 1):
            chars = page.chars or []
            for line_chars in _group_chars_by_line(chars):
                line_text = "".join(c["text"] for c in line_chars).strip()
                if not line_text or len(line_text) > 100:
                    continue

                sizes = [c["size"] for c in line_chars if c.get("size")]
                avg_size = statistics.mean(sizes) if sizes else body_size
                fonts = list({c.get("fontname", "?") for c in line_chars})
                is_bold = any("bold" in (f or "").lower() for f in fonts)
                size_diff = avg_size - body_size

                flag = ""
                if size_diff > 0.5:
                    flag += "[LARGER] "
                if is_bold:
                    flag += "[BOLD] "

                print(f"p{page_num:02d} | sz={avg_size:.1f}({size_diff:+.1f}) | {flag}{repr(line_text)}")
                count += 1
                if count >= max_lines:
                    print(f"\n... (stopped after {max_lines} lines)")
                    return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_pdf.py <pdf_path> [max_lines]")
        sys.exit(1)
    max_l = int(sys.argv[2]) if len(sys.argv) > 2 else 80
    diagnose(sys.argv[1], max_l)
