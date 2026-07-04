import os
from pathlib import Path
from typing import Optional

import httpx
import typer

from .config import API_KEY_ENV_VARS, build_llm_config, load_settings
from .exceptions import ChunkProcessingError
from .glossary import load_glossary
from .llm.factory import get_client
from .logging_utils import configure_logging
from .pipeline import dry_run_stats, run_pipeline

app = typer.Typer(add_completion=False)
doctor_app = typer.Typer(add_completion=False)

LANG_ALIASES = {"jp": "ja"}
SUPPORTED_LANGS = {"th", "ja", "en"}
ON_ERROR_CHOICES = {"abort", "skip"}


def _normalize_lang(value: str) -> str:
    code = LANG_ALIASES.get(value.lower(), value.lower())
    if code not in SUPPORTED_LANGS:
        raise typer.BadParameter(f"unsupported language '{value}' (use th, ja, en, or jp as alias for ja)")
    return code


@app.command()
def translate(
    input_path: Path = typer.Argument(
        ..., exists=True, file_okay=True, dir_okay=False, readable=True,
        help="DOCX/PPTX/XLSX/PDF file to translate",
    ),
    source_lang: str = typer.Option(..., "--from", help="Source language: th, ja/jp, en"),
    target_lang: str = typer.Option(..., "--to", help="Target language: th, ja/jp, en"),
    provider: str = typer.Option("ollama", "--provider", help="ollama, openai, anthropic, thaillm"),
    model: Optional[str] = typer.Option(None, "--model", help="Override the provider's default model"),
    output: Optional[Path] = typer.Option(None, "--output", help="Output file path"),
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to a config YAML file"),
    chunk_tokens: Optional[int] = typer.Option(None, "--chunk-tokens", help="Token budget per chunk"),
    max_retries: Optional[int] = typer.Option(None, "--max-retries", help="Retries per chunk"),
    on_error: Optional[str] = typer.Option(None, "--on-error", help="abort or skip"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    glossary: Optional[Path] = typer.Option(
        None, "--glossary", help="JSON file of {source_term: target_term} for consistent terminology"
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Extract and chunk only; print stats without calling any provider"
    ),
    verbose: bool = typer.Option(False, "-v", "--verbose"),
):
    """Translate a document between Thai, Japanese, and English."""
    configure_logging(verbose)

    src_lang = _normalize_lang(source_lang)
    tgt_lang = _normalize_lang(target_lang)

    resolved_on_error = on_error or "abort"
    if resolved_on_error not in ON_ERROR_CHOICES:
        raise typer.BadParameter(f"--on-error must be one of {sorted(ON_ERROR_CHOICES)}")

    settings = load_settings(str(config_path) if config_path else None)
    resolved_chunk_tokens = chunk_tokens or settings.get("chunk_token_budget", 3000)

    if dry_run:
        stats = dry_run_stats(str(input_path), resolved_chunk_tokens)
        typer.echo(f"doc_type: {stats['doc_type']}")
        typer.echo(f"segments: {stats['segment_count']}")
        typer.echo(f"chunks: {stats['chunk_count']}")
        typer.echo(f"estimated input tokens: {stats['estimated_input_tokens']}")
        raise typer.Exit(code=0)

    glossary_terms = load_glossary(str(glossary)) if glossary else None

    llm_config = build_llm_config(
        settings, provider, model_override=model, temperature_override=temperature
    )
    client = get_client(llm_config)

    try:
        out_path = run_pipeline(
            str(input_path),
            src_lang,
            tgt_lang,
            client,
            output_path=str(output) if output else None,
            chunk_token_budget=resolved_chunk_tokens,
            max_retries=max_retries or settings.get("max_retries", 3),
            on_error=resolved_on_error,
            output_pattern=settings.get("output_pattern", "{stem}.{target_lang}{ext}"),
            font_paths=settings.get("fonts"),
            glossary=glossary_terms,
        )
    except ChunkProcessingError as exc:
        typer.secho(f"Translation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc

    typer.secho(f"Saved: {out_path}", fg=typer.colors.GREEN)


@doctor_app.command()
def doctor(
    config_path: Optional[Path] = typer.Option(None, "--config", help="Path to a config YAML file"),
):
    """Check provider connectivity/config and font availability."""
    settings = load_settings(str(config_path) if config_path else None)
    providers_settings = settings.get("providers", {})

    typer.echo("Providers:")
    for provider_name in ("ollama", "openai", "anthropic", "thaillm"):
        provider_settings = providers_settings.get(provider_name, {})
        model = provider_settings.get("model", "(none configured)")

        if provider_name == "ollama":
            base_url = provider_settings.get("base_url") or os.environ.get("OLLAMA_HOST", "http://localhost:11434")
            try:
                response = httpx.get(f"{base_url}/api/tags", timeout=3.0)
                status = "reachable" if response.status_code == 200 else f"HTTP {response.status_code}"
            except httpx.HTTPError as exc:
                status = f"unreachable ({exc})"
            typer.echo(f"  ollama: model={model} base_url={base_url} -> {status}")
        else:
            env_var = API_KEY_ENV_VARS.get(provider_name)
            has_key = bool(os.environ.get(env_var)) if env_var else False
            typer.echo(f"  {provider_name}: model={model} api_key_set={has_key}")

    typer.echo("Fonts:")
    for lang, path in settings.get("fonts", {}).items():
        exists = Path(path).exists()
        marker = "found" if exists else "MISSING (run: python scripts/fetch_fonts.py)"
        typer.echo(f"  {lang}: {path} -> {marker}")


if __name__ == "__main__":
    app()


def doctor_main() -> None:
    doctor_app()
