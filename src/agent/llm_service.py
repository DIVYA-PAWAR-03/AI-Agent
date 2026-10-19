"""
Dedicated, reusable LLM Service for Document Analysis and Structured Output Generation.
Supports Google Gemini with robust rate-limiting, error handling, and schema validation.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import re
import time
from typing import Any, Dict, List, Optional

from src.config import AppConfig, config as default_config
from src.models.schemas import (
    DocumentCatalogEntry,
    DocumentExtractionRecord,
    FactOrMetric,
    TwoLineInsight,
)
from src.utils.logger import setup_logger

logger = setup_logger("llm_service")


class LLMServiceError(Exception):
    """Base exception for LLM service errors."""


class LLMConfigurationError(LLMServiceError):
    """Raised when the LLM service is invoked without required credentials."""


class LLMRateLimitError(LLMServiceError):
    """Raised when rate limits are exceeded after retries."""


class LLMOutputParsingError(LLMServiceError):
    """Raised when LLM response cannot be parsed into the expected schema."""


class LLMAnalysisError(LLMServiceError):
    """Raised when document analysis fails."""


# Standard primary domain categories per project specifications
STANDARD_DOMAINS = [
    "Artificial Intelligence & Machine Learning",
    "Physics & Quantum Computing",
    "Biology & Life Sciences",
    "Economics & Finance",
    "History & Humanities",
    "Earth & Environmental Sciences",
]


class LLMService:
    """
    Service wrapper around LLM providers (initially Google Gemini).
    Encapsulates prompt construction, rate-limit backoff, and Pydantic validation.
    """

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        client: Optional[Any] = None,
    ):
        self.config = config or default_config
        self._client = client
        self._model = None

        if self.is_configured and self._client is None:
            self._init_client()

    @property
    def is_configured(self) -> bool:
        """Checks if a valid, non-placeholder API key is available."""
        is_valid, _ = self.config.validate_llm_credentials()
        return is_valid

    def _init_client(self) -> None:
        """Initializes Google Gemini GenerativeModel client."""
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.config.gemini_api_key)
            self._client = genai
            self._model = genai.GenerativeModel(
                model_name=self.config.gemini_model,
                generation_config={
                    "response_mime_type": "application/json",
                    "temperature": 0.2,
                },
            )
            logger.info("Initialized Gemini client with model: %s", self.config.gemini_model)
        except Exception as exc:
            logger.warning("Could not initialize Gemini client: %s", exc)

    def _build_analysis_prompt(self, record: DocumentExtractionRecord) -> str:
        """Constructs an evidence-focused extraction prompt for the document."""
        # Compile document text with page markers
        page_chunks: List[str] = []
        for p in record.pages:
            if p.text.strip():
                page_chunks.append(f"--- [Page {p.page_number}] ---\n{p.text}")

        document_content = "\n\n".join(page_chunks)
        # Cap text if excessively large to avoid payload issues, while preserving substantial body
        if len(document_content) > 120_000:
            document_content = document_content[:120_000] + "\n\n[... Remaining content truncated ...]"

        prompt = f"""You are a senior document intelligence analyst. Analyze the following document thoroughly and extract structured insights based STRICTLY and ONLY on the text provided.

DOCUMENT FILENAME: {record.filename}
DOCUMENT ID: {record.doc_id}

RULES & INTEGRITY CONSTRAINTS:
1. Grounding: Every statement must be directly supported by the text. NEVER invent facts, metrics, statistics, recommendations, or conclusions.
2. If recommendations, risks, or limitations are NOT explicitly mentioned, return empty lists [] for those fields. Do not fabricate hypothetical advice.
3. Two-Line Insight: Must be dense, informative, and exactly two distinct sentences:
   - Line 1: Summarize the core topic, problem, or theoretical premise.
   - Line 2: Highlight the primary empirical finding, methodology, or impactful conclusion.
4. Facts & Metrics: Extract specific numerical facts, percentages, dates, metrics, and named entities with page numbers where found. Distinguish explicit statements from inferences.
5. Domain Category: Select the best matching primary domain among:
   - Artificial Intelligence & Machine Learning
   - Physics & Quantum Computing
   - Biology & Life Sciences
   - Economics & Finance
   - History & Humanities
   - Earth & Environmental Sciences

OUTPUT FORMAT: Return a valid JSON object matching this exact schema:
{{
  "title": "Document Title",
  "main_topic": "Core subject matter",
  "purpose": "Primary purpose or objective of the document",
  "category": "Selected Domain Category",
  "two_line_insight": {{
    "doc_id": "{record.doc_id}",
    "line_1": "First insight sentence.",
    "line_2": "Second insight sentence."
  }},
  "summary": "Concise, structured executive summary (2-4 paragraphs).",
  "key_findings": ["Finding 1 with evidence", "Finding 2 with evidence"],
  "conclusions": ["Conclusion 1", "Conclusion 2"],
  "facts_and_metrics": [
    {{
      "fact": "Specific factual statistic or date",
      "page_number": 1,
      "is_explicit_fact": true
    }}
  ],
  "recommendations": [],
  "risks_and_limitations": [],
  "explicit_vs_interpretation_notes": "Note distinguishing confirmed factual data from inferences or interpretations."
}}

DOCUMENT TEXT:
{document_content}
"""
        return prompt

    def _clean_json_response(self, raw_text: str) -> str:
        """Strips markdown code fences or conversational text around JSON output."""
        cleaned = raw_text.strip()
        # Remove markdown code fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    def _parse_and_validate_response(
        self,
        raw_output: str,
        record: DocumentExtractionRecord,
    ) -> DocumentCatalogEntry:
        """Parses LLM JSON string and validates into DocumentCatalogEntry schema."""
        cleaned_json = self._clean_json_response(raw_output)

        try:
            data = json.loads(cleaned_json)
        except json.JSONDecodeError as err:
            raise LLMOutputParsingError(
                f"Failed to decode LLM response as JSON: {err}. Output sample: {cleaned_json[:200]}"
            ) from err

        try:
            # Ensure required linkage fields are present
            data["doc_id"] = record.doc_id
            data["filename"] = record.filename
            data["file_hash"] = record.file_hash
            data["analyzed_at"] = datetime.now(timezone.utc).isoformat()
            data["model_used"] = self.config.gemini_model
            data["status"] = "success"
            data["error"] = None

            # Validate two_line_insight doc_id linkage
            if "two_line_insight" in data and isinstance(data["two_line_insight"], dict):
                data["two_line_insight"]["doc_id"] = record.doc_id

            return DocumentCatalogEntry(**data)
        except Exception as err:
            raise LLMOutputParsingError(
                f"Failed to validate catalog entry against Pydantic schema: {err}"
            ) from err

    def analyze_document(
        self,
        record: DocumentExtractionRecord,
        max_retries: int = 3,
        retry_delay_seconds: float = 4.0,
    ) -> DocumentCatalogEntry:
        """
        Analyzes a single document extraction record using the configured LLM.
        Handles rate limits with exponential backoff and retries.
        """
        if not self.is_configured:
            _, msg = self.config.validate_llm_credentials()
            raise LLConfigurationError(msg)

        if self._model is None and self._client is not None:
            # Client provided manually or initialized
            if hasattr(self._client, "GenerativeModel"):
                self._model = self._client.GenerativeModel(
                    model_name=self.config.gemini_model,
                    generation_config={"response_mime_type": "application/json"},
                )

        if self._model is None:
            raise LLMConfigurationError("LLM model client is not initialized.")

        prompt = self._build_analysis_prompt(record)

        last_error: Optional[Exception] = None
        current_delay = retry_delay_seconds

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(
                    "Calling LLM for %s (attempt %d/%d)",
                    record.doc_id,
                    attempt,
                    max_retries,
                )
                response = self._model.generate_content(prompt)
                raw_text = response.text if hasattr(response, "text") else str(response)

                entry = self._parse_and_validate_response(raw_text, record)
                return entry

            except LLMOutputParsingError as parse_err:
                logger.warning(
                    "Parsing error for %s (attempt %d/%d): %s",
                    record.doc_id,
                    attempt,
                    max_retries,
                    parse_err,
                )
                last_error = parse_err
                if attempt < max_retries:
                    time.sleep(1.0)

            except Exception as exc:
                err_str = str(exc).lower()
                is_rate_limit = (
                    "429" in err_str
                    or "resourceexhausted" in err_str
                    or "quota" in err_str
                    or "rate limit" in err_str
                )

                if is_rate_limit:
                    logger.warning(
                        "Rate limit encountered for %s. Backing off for %.1fs (attempt %d/%d)",
                        record.doc_id,
                        current_delay,
                        attempt,
                        max_retries,
                    )
                    time.sleep(current_delay)
                    current_delay *= 2.0
                    last_error = LLMRateLimitError(f"Rate limit exceeded: {exc}")
                else:
                    logger.warning(
                        "LLM call error for %s: %s (attempt %d/%d)",
                        record.doc_id,
                        exc,
                        attempt,
                        max_retries,
                    )
                    last_error = exc
                    if attempt < max_retries:
                        time.sleep(current_delay)

        # If all attempts exhausted, return failed record rather than crashing the batch
        now_iso = datetime.now(timezone.utc).isoformat()
        return DocumentCatalogEntry(
            doc_id=record.doc_id,
            filename=record.filename,
            file_hash=record.file_hash,
            title=record.filename,
            main_topic="Analysis failed",
            purpose="Analysis failed",
            category="Uncategorized",
            two_line_insight=TwoLineInsight(
                doc_id=record.doc_id,
                line_1="Analysis could not be completed for this document.",
                line_2=f"Reason: {last_error}",
            ),
            summary="Document analysis failed during LLM processing.",
            key_findings=[],
            conclusions=[],
            facts_and_metrics=[],
            recommendations=[],
            risks_and_limitations=[],
            explicit_vs_interpretation_notes=None,
            analyzed_at=now_iso,
            model_used=self.config.gemini_model,
            status="failed",
            error=str(last_error),
        )


LLConfigurationError = LLMConfigurationError
