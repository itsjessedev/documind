import re
from typing import List, Tuple
from io import BytesIO
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from .models import DocumentType


class DocumentProcessor:
    """Processes various document formats and extracts text content."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text(self, content: bytes, filename: str) -> Tuple[str, DocumentType]:
        """Extract text from document based on file extension."""
        ext = filename.lower().split('.')[-1]

        if ext == 'pdf':
            return self._extract_pdf(content), DocumentType.PDF
        elif ext == 'docx':
            return self._extract_docx(content), DocumentType.DOCX
        elif ext == 'txt':
            return content.decode('utf-8', errors='ignore'), DocumentType.TXT
        elif ext in ['html', 'htm']:
            return self._extract_html(content), DocumentType.HTML
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_pdf(self, content: bytes) -> str:
        """Extract text from PDF."""
        reader = PdfReader(BytesIO(content))
        text_parts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                text_parts.append(text)
        return '\n\n'.join(text_parts)

    def _extract_docx(self, content: bytes) -> str:
        """Extract text from DOCX."""
        doc = DocxDocument(BytesIO(content))
        return '\n\n'.join([para.text for para in doc.paragraphs if para.text.strip()])

    def _extract_html(self, content: bytes) -> str:
        """Extract text from HTML."""
        soup = BeautifulSoup(content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text(separator='\n', strip=True)

    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks for better retrieval."""
        # Clean the text
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [text] if text else []

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                # Keep overlap
                overlap_text = ' '.join(current_chunk)
                while len(overlap_text) > self.chunk_overlap and current_chunk:
                    current_chunk.pop(0)
                    overlap_text = ' '.join(current_chunk)
                current_length = len(overlap_text)

            current_chunk.append(sentence)
            current_length += sentence_length + 1

        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks


# Sample documents for demo
SAMPLE_DOCUMENTS = {
    "company_policy.txt": """
    Employee Handbook - Company Policies

    Section 1: Working Hours
    Standard working hours are 9 AM to 5 PM, Monday through Friday.
    Flexible working arrangements may be available upon manager approval.
    Remote work is permitted up to 2 days per week for eligible employees.

    Section 2: Leave Policy
    Full-time employees receive 15 days of paid vacation annually.
    Sick leave: 10 days per year, can be carried over up to 5 days.
    Parental leave: 12 weeks paid for primary caregivers, 4 weeks for secondary.

    Section 3: Benefits
    Health insurance is provided through BlueCross starting from day one.
    401(k) matching up to 4% of salary after 90 days of employment.
    Professional development budget of $2,000 per year.

    Section 4: Code of Conduct
    All employees must maintain professional behavior at all times.
    Harassment of any kind will not be tolerated.
    Conflicts of interest must be disclosed to HR immediately.
    """,

    "product_guide.txt": """
    Product User Guide - CloudSync Pro

    Getting Started
    CloudSync Pro is an enterprise file synchronization solution.
    Installation requires administrator privileges on Windows, Mac, or Linux.
    Minimum requirements: 4GB RAM, 500MB disk space, internet connection.

    Features Overview
    1. Real-time Sync: Files sync automatically across all devices.
    2. Version History: Access up to 30 days of file versions.
    3. Selective Sync: Choose which folders to sync on each device.
    4. Conflict Resolution: Smart merging prevents data loss.

    Security Features
    - End-to-end encryption using AES-256
    - Two-factor authentication support
    - Admin controls for team permissions
    - Audit logs for compliance requirements

    Troubleshooting
    If sync fails, check internet connection first.
    Clear cache: Settings > Advanced > Clear Cache
    Contact support: support@cloudsync.example.com
    """,

    "technical_spec.txt": """
    Technical Specification Document
    Project: API Gateway v2.0

    Architecture Overview
    The API Gateway serves as the single entry point for all client requests.
    Built on Node.js with Express framework for high performance.
    Uses Redis for rate limiting and caching.

    Authentication
    JWT tokens with 1-hour expiration for access tokens.
    Refresh tokens valid for 7 days, stored securely.
    OAuth 2.0 support for third-party integrations.

    Rate Limiting
    Default: 1000 requests per minute per API key.
    Premium tier: 10,000 requests per minute.
    Burst allowance: 20% above limit for 30 seconds.

    Endpoints
    GET /api/v2/users - List users (paginated)
    POST /api/v2/users - Create new user
    GET /api/v2/users/:id - Get user details
    PUT /api/v2/users/:id - Update user
    DELETE /api/v2/users/:id - Soft delete user

    Error Codes
    400 - Bad Request (invalid parameters)
    401 - Unauthorized (invalid or expired token)
    403 - Forbidden (insufficient permissions)
    429 - Too Many Requests (rate limit exceeded)
    500 - Internal Server Error
    """
}
