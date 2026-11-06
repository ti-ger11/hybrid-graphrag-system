"""Layout-aware PDF parsing with a pluggable adapter interface.

Provides an abstract `DocumentParser` protocol and a concrete
`UnstructuredPDFParser` that uses Unstructured's hi_res strategy for
accurate extraction of tables, multi-column layouts, and section headers.

To add a new parser (e.g. LlamaParse), implement the `DocumentParser`
protocol and inject it into the `IngestionPipeline`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


# ── Data Structures ──────────────────────────────────────────────────────


@dataclass
class ParsedElement:
    """A single structural element extracted from a PDF document.

    Attributes:
        text: The textual content of the element.
        element_type: Structural category (e.g. "Title", "NarrativeText",
                       "Table", "ListItem", "Header", "Footer").
        page_number: 1-indexed page where the element appears (None if unknown).
        metadata: Additional parser-specific metadata (coordinates, HTML for
                   tables, etc.).
    """

    text: str
    element_type: str
    page_number: int | None = None
    metadata: dict = field(default_factory=dict)


# ── Parser Interface ─────────────────────────────────────────────────────


@runtime_checkable
class DocumentParser(Protocol):
    """Interface that all document parsers must satisfy.

    Implementing this protocol allows swapping parsing backends without
    modifying downstream chunking or indexing code.
    """

    def parse(self, file_path: Path) -> list[ParsedElement]:
        """Parse a document file into a list of structural elements.

        Args:
            file_path: Absolute path to the document file.

        Returns:
            Ordered list of ParsedElement objects reflecting the document's
            reading order.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the file format is unsupported.
        """
        ...


# ── Concrete Implementation: Unstructured ────────────────────────────────


class UnstructuredPDFParser:
    """Layout-aware PDF parser using Unstructured's hi_res strategy.

    The hi_res strategy routes pages through layout detection models (YOLO-based)
    to correctly handle:
    - Multi-column layouts
    - Complex tables (extracted as HTML via `infer_table_structure`)
    - Sidebars, headers, footers
    - Proper reading order reconstruction

    For simpler/faster parsing (at the cost of layout accuracy), set
    `strategy="fast"`.
    """

    def __init__(
        self,
        strategy: str = "hi_res",
        languages: list[str] | None = None,
        infer_tables: bool = True,
    ) -> None:
        self.strategy = strategy
        self.languages = languages or ["eng"]
        self.infer_tables = infer_tables

    def parse(self, file_path: Path) -> list[ParsedElement]:
        """Parse a PDF into layout-aware structural elements.

        Args:
            file_path: Path to the PDF file.

        Returns:
            Ordered list of ParsedElement objects.

        Raises:
            FileNotFoundError: If the PDF file does not exist.
            ValueError: If the file is not a PDF.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        if file_path.suffix.lower() != ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {file_path.suffix}")

        logger.info(
            "Parsing PDF with Unstructured (strategy=%s): %s",
            self.strategy,
            file_path.name,
        )

        # Lazy import to avoid heavy Unstructured dependencies at module load
        try:
            from unstructured.partition.pdf import partition_pdf

            raw_elements = partition_pdf(
                filename=str(file_path),
                strategy=self.strategy,
                infer_table_structure=self.infer_tables,
                languages=self.languages,
            )

            parsed: list[ParsedElement] = []
            for el in raw_elements:
                text = str(el).strip()
                if not text:
                    continue

                # Extract metadata safely
                meta: dict = {}
                if hasattr(el.metadata, "to_dict"):
                    meta = el.metadata.to_dict()
                elif hasattr(el.metadata, "__dict__"):
                    meta = {
                        k: v
                        for k, v in el.metadata.__dict__.items()
                        if not k.startswith("_") and v is not None
                    }

                page_num = getattr(el.metadata, "page_number", None)

                # Preserve HTML table representation if available
                if el.category == "Table" and hasattr(el.metadata, "text_as_html"):
                    meta["html"] = el.metadata.text_as_html

                parsed.append(
                    ParsedElement(
                        text=text,
                        element_type=el.category,
                        page_number=page_num,
                        metadata=meta,
                    )
                )

            if parsed:
                logger.info(
                    "Extracted %d structural elements from %s using Unstructured",
                    len(parsed),
                    file_path.name,
                )
                return parsed
        except Exception as e:
            logger.warning(
                "Unstructured parsing unavailable or failed (%s). Falling back to pypdf.",
                e,
            )

        # Graceful fallback to pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(str(file_path), strict=False)
            parsed = []
            for i, page in enumerate(reader.pages, start=1):
                try:
                    page_text = (page.extract_text() or "").strip()
                except Exception as page_err:
                    logger.warning("Failed to extract text from page %d of %s: %s", i, file_path.name, page_err)
                    page_text = ""
                if page_text:
                    parsed.append(
                        ParsedElement(
                            text=page_text,
                            element_type="NarrativeText",
                            page_number=i,
                            metadata={"source": "pypdf", "page_number": i},
                        )
                    )
            logger.info("Extracted %d pages from %s using pypdf fallback", len(parsed), file_path.name)
            return parsed
        except Exception as e:
            logger.error("Failed to parse PDF with pypdf: %s", e)
            raise
