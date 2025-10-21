import pdfplumber

def extract_text_from_pdf(pdf_path):
    """Extracts all text from a multi-page PDF and returns it as a string."""
    all_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                all_text.append(text)
    return "\n\n".join(all_text)

