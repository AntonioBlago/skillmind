"""Configuration management for SkillMind."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class EmbeddingConfig(BaseModel):
    """Embedding model configuration."""

    provider: str = Field(default="sentence-transformers", description="sentence-transformers | openai")
    model: str = Field(default="all-MiniLM-L6-v2", description="Model name")
    dimension: int = Field(default=384, description="Embedding dimension")


class StoreConfig(BaseModel):
    """Vector store backend configuration."""

    backend: str = Field(default="chroma", description="chroma | pinecone | supabase | qdrant | faiss")

    # Chroma
    chroma_path: str = Field(default=".skillmind/chroma", description="ChromaDB persistence path")

    # Pinecone
    pinecone_api_key: str = Field(default="", description="Pinecone API key")
    pinecone_index: str = Field(default="skillmind", description="Pinecone index name")
    pinecone_environment: str = Field(default="", description="Pinecone environment")

    # Supabase
    supabase_url: str = Field(default="", description="Supabase project URL")
    supabase_key: str = Field(default="", description="Supabase anon/service key")
    supabase_table: str = Field(default="memories", description="Table name")

    # Qdrant
    qdrant_url: str = Field(default="http://localhost:6333", description="Qdrant server URL")
    qdrant_api_key: str = Field(default="", description="Qdrant API key")
    qdrant_collection: str = Field(default="skillmind", description="Collection name")

    # FAISS
    faiss_path: str = Field(default=".skillmind/faiss", description="FAISS index path")


class CustomPattern(BaseModel):
    """A user-defined detection pattern."""

    pattern: str = Field(..., description="Regex pattern to match in user messages")
    memory_type: str = Field(..., description="Memory type to assign: user, feedback, project, reference, skill")
    topic: str = Field(default="", description="Force this topic when pattern matches (empty = auto-detect)")
    description: str = Field(default="", description="What this pattern detects (for reference)")


class ListenerConfig(BaseModel):
    """Listener configuration."""

    watch_git: bool = Field(default=True, description="Watch git events")
    watch_files: bool = Field(default=True, description="Watch file changes")
    auto_learn: bool = Field(default=True, description="Auto-extract memories from conversations")
    consolidation_interval: int = Field(default=86400, description="Consolidation interval in seconds")
    custom_patterns: list[CustomPattern] = Field(default_factory=list, description="User-defined detection patterns")
    review_mode: str = Field(
        default="review",
        description="How auto-detected memories are handled: "
                    "'review' = queue for approval (default), "
                    "'auto' = store directly without review, "
                    "'off' = don't auto-detect at all"
    )


class SanitizerConfig(BaseModel):
    """Sanitizer / anonymization configuration."""

    enabled: bool = Field(default=True, description="Enable automatic sanitization of sensitive data")
    redact_api_keys: bool = Field(default=True, description="Redact API keys, tokens, secrets")
    redact_personal: bool = Field(default=True, description="Redact emails, phones, IBANs")
    redact_paths: bool = Field(default=True, description="Redact env vars, private keys")
    redact_names: list[str] = Field(default_factory=list, description="Specific names to anonymize")
    allowlist: list[str] = Field(default_factory=list, description="Patterns to never redact")
    custom_patterns: list[list[str]] = Field(default_factory=list, description="Extra [pattern, label] pairs")


class SkillMindConfig(BaseModel):
    """Root configuration."""

    version: str = "1.0"
    project_name: str = "default"
    data_dir: str = ".skillmind"
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    store: StoreConfig = Field(default_factory=StoreConfig)
    sanitizer: SanitizerConfig = Field(default_factory=SanitizerConfig)
    listener: ListenerConfig = Field(default_factory=ListenerConfig)
    context_max_tokens: int = Field(default=4000, description="Max tokens for context injection")

    @classmethod
    def load(cls, path: str | Path | None = None) -> SkillMindConfig:
        """Load config from YAML file, falling back to defaults. Also loads .env."""
        # Load .env if present
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        if path is None:
            path = Path(".skillmind/config.yml")
        path = Path(path)

        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f) or {}
            return cls(**data)

        return cls()

    def save(self, path: str | Path | None = None) -> None:
        """Save config to YAML."""
        if path is None:
            path = Path(f"{self.data_dir}/config.yml")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False, sort_keys=False)

    def resolve_env(self) -> SkillMindConfig:
        """Override config values from environment variables."""
        env_map = {
            "SKILLMIND_BACKEND": "store.backend",
            "PINECONE_API_KEY": "store.pinecone_api_key",
            "PINECONE_ENVIRONMENT": "store.pinecone_environment",
            "SUPABASE_URL": "store.supabase_url",
            "SUPABASE_KEY": "store.supabase_key",
            "QDRANT_URL": "store.qdrant_url",
            "QDRANT_API_KEY": "store.qdrant_api_key",
        }
        for env_var, config_path in env_map.items():
            val = os.environ.get(env_var)
            if val:
                parts = config_path.split(".")
                obj: Any = self
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], val)
        return self
