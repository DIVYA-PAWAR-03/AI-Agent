"""
Configuration management for the PDF Document Summarization & Insight Agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if it exists
load_dotenv(override=False)


def resolve_dataset_dir(custom_path: Optional[Path | str] = None) -> Path:
    """
    Locates the PDF dataset directory reliably without guessing arbitrary paths.
    Checks explicit override, environment variable, default data/pdfs, and repository Details folder.
    Raises FileNotFoundError with instructions if no dataset is found.
    """
    if custom_path:
        p = Path(custom_path)
        if not p.exists() or not p.is_dir():
            raise FileNotFoundError(
                f"Specified PDF dataset directory '{custom_path}' does not exist or is not a directory. "
                "Please place your PDF files into './data/pdfs/' or provide a valid directory."
            )
        return p.resolve()

    env_path = os.getenv("PDF_DATASET_DIR")
    if env_path:
        p = Path(env_path)
        if p.exists() and p.is_dir():
            return p.resolve()
        raise FileNotFoundError(
            f"Configured PDF_DATASET_DIR '{env_path}' does not exist or is not a directory. "
            "Please verify your .env file or place your PDF files into './data/pdfs/'."
        )

    # Standard default per task requirements
    default_dir = Path("./data/pdfs")
    if default_dir.exists() and default_dir.is_dir() and any(default_dir.glob("*.pdf")):
        return default_dir.resolve()

    # Known repository dataset folder from Phase 00 inspection
    details_dir = Path("./Details/Pdf_Dataset-20261010T065421Z-1-001/Pdf_Dataset")
    if details_dir.exists() and details_dir.is_dir() and any(details_dir.glob("*.pdf")):
        return details_dir.resolve()

    # If default_dir exists even if empty, return it
    if default_dir.exists() and default_dir.is_dir():
        return default_dir.resolve()

    raise FileNotFoundError(
        "PDF dataset directory could not be located. "
        "Please place your PDF files into './data/pdfs/' or specify the path "
        "using the PDF_DATASET_DIR environment variable in your .env file."
    )


class AppConfig(BaseModel):
    """Application configuration schema with fallback defaults."""

    app_name: str = "PDF Document Summarization & Insight Agent"
    app_version: str = "0.1.0"
    app_env: str = Field(default_factory=lambda: os.getenv("APP_ENV", "development"))
    log_level: str = Field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # LLM Provider Configuration
    llm_provider: str = Field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY")
    )
    gemini_model: str = Field(
        default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    )

    # Embeddings & Vector Store
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
    )
    chroma_persist_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db"))
    )

    # Extracted Data Storage Directory
    extracted_data_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("EXTRACTED_DATA_DIR", "./data/extracted"))
    )

    # Dataset Location (defaults to data/pdfs per Phase 02 requirements)
    pdf_dataset_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("PDF_DATASET_DIR", "./data/pdfs"))
    )

    def validate_llm_credentials(self) -> Tuple[bool, str]:
        """
        Validates LLM credentials without raising unhandled exceptions at import time.
        Returns a tuple of (is_valid, message).
        """
        if not self.gemini_api_key or self.gemini_api_key.strip() in (
            "",
            "your_gemini_api_key_here",
            "YOUR_API_KEY",
        ):
            return (
                False,
                "GEMINI_API_KEY is not configured or contains placeholder text. "
                "Please add a valid GEMINI_API_KEY to your .env file to enable LLM operations.",
            )
        return True, "GEMINI_API_KEY is configured."

    def get_setup_status(self) -> Dict[str, Any]:
        """Inspects and returns a dictionary of the environment and setup status."""
        is_key_valid, key_msg = self.validate_llm_credentials()

        try:
            resolved_pdf_dir = resolve_dataset_dir(self.pdf_dataset_dir)
            pdf_exists = True
            pdf_count = len(list(resolved_pdf_dir.glob("*.pdf")))
            pdf_dir_str = str(resolved_pdf_dir)
        except FileNotFoundError:
            pdf_exists = False
            pdf_count = 0
            pdf_dir_str = str(self.pdf_dataset_dir)

        return {
            "app_name": self.app_name,
            "version": self.app_version,
            "environment": self.app_env,
            "llm_provider": self.llm_provider,
            "gemini_model": self.gemini_model,
            "gemini_api_key_configured": is_key_valid,
            "gemini_status_message": key_msg,
            "embedding_model": self.embedding_model_name,
            "chroma_persist_dir": str(self.chroma_persist_dir),
            "extracted_data_dir": str(self.extracted_data_dir),
            "pdf_dataset_dir": pdf_dir_str,
            "pdf_dataset_found": pdf_exists,
            "pdf_count": pdf_count,
        }

    def ensure_directories(self) -> None:
        """Creates necessary local runtime directories if they do not exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        self.extracted_data_dir.mkdir(parents=True, exist_ok=True)
        self.pdf_dataset_dir.mkdir(parents=True, exist_ok=True)


# Global singleton instance for easy import across modules
config = AppConfig()
