"""
Conversational memory management for multi-turn Q&A and active document tracking.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from src.models.schemas import Citation, RetrievedPassage


class ConversationMessage(BaseModel):
    """A single turn in the conversation."""

    role: str = Field(description="'user' or 'assistant'")
    content: str = Field(description="Message text")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    operation: Optional[str] = None
    referenced_doc_id: Optional[str] = None
    citations: List[Citation] = Field(default_factory=list)


class ConversationSession:
    """
    Maintains bounded conversation history and active document context for multi-turn reasoning.
    """

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self.messages: List[ConversationMessage] = []
        self.active_doc_id: Optional[str] = None
        self.last_retrieved_passages: List[RetrievedPassage] = []

    def set_active_document(self, doc_id: Optional[str]) -> None:
        """Sets or updates the currently active focus document."""
        if doc_id:
            self.active_doc_id = doc_id

    def add_user_query(self, query: str) -> None:
        """Appends user query to conversation history."""
        self.messages.append(
            ConversationMessage(
                role="user",
                content=query,
                referenced_doc_id=self.active_doc_id,
            )
        )
        self._trim_history()

    def add_assistant_response(
        self,
        answer: str,
        operation: str,
        doc_id: Optional[str] = None,
        citations: Optional[List[Citation]] = None,
    ) -> None:
        """Appends assistant answer to history with metadata and citations."""
        target_doc = doc_id or self.active_doc_id
        if target_doc:
            self.set_active_document(target_doc)

        self.messages.append(
            ConversationMessage(
                role="assistant",
                content=answer,
                operation=operation,
                referenced_doc_id=target_doc,
                citations=citations or [],
            )
        )
        self._trim_history()

    def _trim_history(self) -> None:
        """Keeps conversation within bounds to prevent token overflow."""
        if len(self.messages) > self.max_turns * 2:
            self.messages = self.messages[-self.max_turns * 2 :]

    def format_history_for_prompt(self, max_items: int = 4) -> str:
        """Formats the most recent dialogue turns for LLM prompt context."""
        recent = self.messages[-max_items * 2 :]
        if not recent:
            return ""

        formatted_lines: List[str] = ["RECENT CONVERSATION HISTORY:"]
        for msg in recent:
            speaker = "User" if msg.role == "user" else "Assistant"
            doc_note = f" (Focus: {msg.referenced_doc_id})" if msg.referenced_doc_id else ""
            formatted_lines.append(f"{speaker}{doc_note}: {msg.content}")

        return "\n".join(formatted_lines)

    def clear(self) -> None:
        """Resets conversation state."""
        self.messages.clear()
        self.active_doc_id = None
        self.last_retrieved_passages.clear()
