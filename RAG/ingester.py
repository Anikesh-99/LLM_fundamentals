import os
from pypdf import PdfReader

directory_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

def convert_pdf_to_txt(pdf_path, output_txt_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += (page.extract_text() or "") + "\n"   # extract_text() can return None on image-only pages
    with open(output_txt_path, "w", encoding="utf-8") as f:
        f.write(text)

if __name__ == "__main__":
    for filename in os.listdir(directory_path):
        full_path = os.path.join(directory_path, filename)
        if not full_path.lower().endswith(".pdf"):
            continue
        base = os.path.splitext(os.path.basename(full_path))[0]
        convert_pdf_to_txt(full_path, os.path.join(directory_path, base + ".txt"))
        print(f"converted: {filename}")
