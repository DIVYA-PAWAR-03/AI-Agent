"""
Configuration management for the PDF Document Summarization & Insight Agent.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file if it exists
load_dotenv(override=False)


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

    # Dataset Location
    pdf_dataset_dir: Path = Field(
        default_factory=lambda: Path(
            os.getenv(
                "PDF_DATASET_DIR",
                "./Details/Pdf_Dataset-20261010T065421Z-1-001/Pdf_Dataset",
            )
        )
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

        pdf_exists = self.pdf_dataset_dir.exists() and self.pdf_dataset_dir.is_dir()
        pdf_count = 0
        if pdf_exists:
            pdf_count = len(list(self.pdf_dataset_dir.glob("*.pdf")))

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
            "pdf_dataset_dir": str(self.pdf_dataset_dir),
            "pdf_dataset_found": pdf_exists,
            "pdf_count": pdf_count,
        }

    def ensure_directories(self) -> None:
        """Creates necessary local runtime directories if they do not exist."""
        self.chroma_persist_dir.mkdir(parents=True, exist_ok=True)


# Global singleton instance for easy import across modules
config = AppConfig()
