"""Request and response models for the gateway API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class Target(BaseModel):
    """One provider to try. ``model`` falls back to the provider's configured default."""

    provider: str
    model: str | None = None


class ChatRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    targets: list[Target] | None = Field(
        default=None, description="Providers to try in order; defaults to DEFAULT_ROUTE."
    )
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=8192)
    cache: bool = Field(default=True, description="Cache the answer (only when temperature is 0).")

    @model_validator(mode="after")
    def _needs_a_non_system_message(self) -> ChatRequest:
        if all(m.role == "system" for m in self.messages):
            raise ValueError("messages must contain at least one user or assistant message")
        return self


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class Attempt(BaseModel):
    provider: str
    model: str
    ok: bool
    status_code: int | None = None
    error: str | None = None
    latency_ms: float


class ChatResponse(BaseModel):
    content: str
    provider: str
    model: str
    usage: Usage
    total_tokens: int
    cost_usd: float | None = Field(description="None when the model has no price configured.")
    latency_ms: float
    cached: bool
    attempts: list[Attempt]
