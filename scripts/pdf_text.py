import sys
import fitz

doc = fitz.open(sys.argv[1])
with open(sys.argv[2], "w", encoding="utf-8") as f:
    for i, page in enumerate(doc, 1):
        f.write(f"\n===== page {i} =====\n")
        f.write(page.get_text())
print(len(doc), "pages")
