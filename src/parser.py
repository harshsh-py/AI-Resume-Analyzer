from typing import Optional, Dict, Any
import re
import io
from PyPDF2 import PdfReader

SECTION_HEADERS = [
    r'education', r'experience', r'work experience', r'projects',
    r'skills', r'technical skills', r'certifications', r'achievements',
    r'summary', r'profile'
]

def read_pdf(file_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(file_bytes))
    text_pages = []
    for page in reader.pages:
        try:
            text_pages.append(page.extract_text() or "")
        except Exception:
            continue
    return "\n".join(text_pages)

def clean_text(text: str) -> str:
    text = re.sub(r'[\r\t]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def split_sections(text: str) -> Dict[str, str]:
    lowered = text.lower()
    # Build regex to find headers
    pattern = r'(?mi)^(?:' + '|'.join(SECTION_HEADERS) + r')\s*:?\s*$'
    matches = list(re.finditer(pattern, lowered))
    sections = {}
    if not matches:
        sections["general"] = text
        return sections
    # Walk through matches to slice text
    spans = [(m.group(0).strip(), m.start(), m.end()) for m in matches]
    spans.append(("__END__", len(lowered), len(lowered)))
    for i in range(len(spans)-1):
        header = re.sub(r':\s*$', '', spans[i][0]).strip()
        start = spans[i][2]
        end = spans[i+1][1]
        sections[header] = text[start:end].strip()
    return sections

def parse_resume(file_bytes: bytes, filetype: str) -> Dict[str, Any]:
    if filetype == "pdf":
        raw = read_pdf(file_bytes)
    elif filetype == "txt":
        raw = file_bytes.decode("utf-8", errors="ignore")
    else:
        # Try naive decode
        raw = file_bytes.decode("utf-8", errors="ignore")
    raw = clean_text(raw)
    sections = split_sections(raw)
    return {"raw_text": raw, "sections": sections}
