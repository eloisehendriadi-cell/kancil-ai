from the_pdf_reader import extract_text_from_pdf
from the_summarizer import summarize_text

pdf_text = extract_text_from_pdf("sample.pdf")

# Print raw text or summarized version
print("📄 PDF Extracted Text (first 1000 chars):\n")
print(pdf_text[:1000])  # Just to preview

print("\n📝 Summary:\n")
print(summarize_text(pdf_text))

