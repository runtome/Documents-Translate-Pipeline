# Documents-Translate-Pipeline

Translate DOCX, PPTX, XLSX, and PDF documents between Thai, Japanese, and English, using
a local Ollama model or a hosted provider (OpenAI, Anthropic, thaillm.or.th).

Rather than sending a whole document to an LLM as one blob of text, the pipeline extracts
each paragraph/cell/text block into a small "segment" with a stable ID, batches segments into
token-budgeted chunks wrapped in `<SEG id="...">` tags, translates each chunk, validates that
the response contains exactly the same set of IDs, and writes translations back in place using
the original document objects — so formatting, tables, images, and layout are preserved and a
failed chunk can be retried without reprocessing the whole file.

## Setup

Requires Python 3.11 or 3.12 (3.14 is not yet supported — some native dependencies don't ship
wheels for it yet).

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"
python scripts/fetch_fonts.py   # downloads NotoSansThai/NotoSansJP for PDF output
cp .env.example .env            # fill in API keys for any hosted provider you'll use
```

Local Ollama needs no API key, just a running server (`ollama serve`) and a pulled model
(`ollama pull llama3.1`).

## Usage

```bash
translate INPUT_PATH --from LANG --to LANG [OPTIONS]
```

`LANG` is `th`, `ja`, or `en` (`jp` is accepted as an alias for `ja`).

```bash
# Local, free, no API key
translate report.docx --from en --to th --provider ollama --model llama3.1

# Hosted providers (reads API key from .env)
translate slides.pptx --from ja --to en --provider openai --model gpt-4o-mini
translate sheet.xlsx --from th --to ja --provider anthropic --model claude-sonnet-5
translate manual.pdf --from en --to th --provider thaillm
```

Options:

| Flag | Description |
|---|---|
| `--output PATH` | Output file path (default: `{stem}.{target_lang}{ext}` next to the input) |
| `--config PATH` | Path to a config YAML file (default: `config/default.yaml`) |
| `--chunk-tokens N` | Token budget per LLM request (default: 3000) |
| `--max-segments-per-chunk N` | Max segments per chunk regardless of token budget (default: 40) |
| `--max-retries N` | Retries per chunk on transport errors or malformed responses (default: 3) |
| `--on-error {abort,skip}` | `abort` the whole run, or `skip` a failed chunk and leave its segments untranslated (default: `abort`) |
| `--temperature FLOAT` | Sampling temperature |
| `--glossary PATH` | JSON file of `{"source term": "target term"}` for consistent terminology |
| `--dry-run` | Extract and chunk only; print segment/chunk/token counts without calling any provider |
| `-v` / `--verbose` | Debug logging |

Check provider connectivity and font availability:

```bash
translate-doctor
```

## Configuration

Non-secret defaults live in `config/default.yaml` (chunk size, retries, default model per
provider, font paths). Secrets (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `THAILLM_API_KEY`,
`OLLAMA_HOST`) go in `.env`, which is gitignored — copy `.env.example` to get started.
CLI flags override environment variables, which override `config/default.yaml`.

## Testing

```bash
pytest
```

The suite runs fully offline against a fake, deterministic LLM client — no network or API keys
required. Provider-specific behavior (error wrapping, response parsing) is tested against
mocked transports in `tests/test_llm_providers_mocked.py`.

## Notes on small/local models

The `<SEG id="...">` tags sent to the LLM use short per-chunk numeric ids (`1`, `2`, ...), not
the segments' internal UUIDs — small local models (e.g. a 3B Ollama model) reliably corrupt long
hex strings when asked to copy dozens of them back verbatim, which fails the response validation.
Segments are also capped at `--max-segments-per-chunk` (default 40) per request regardless of
token budget, since tag-count tracking degrades with weaker models well before the token budget
is exhausted. If you still see `chunk N failed: missing=[...] extra=[...]` errors with a small
model, lower `--max-segments-per-chunk` further (e.g. to 15-20) or switch to a larger model.

## Known v1 limitations

- **Run-level formatting**: DOCX/PPTX paragraphs are translated as a whole and written into the
  paragraph's dominant (longest) run; other runs are cleared. A sentence with one bold word in
  the middle will lose that inline formatting distinction after translation.
- **PDF redaction background**: translated PDF text blocks are redacted with a white fill before
  the translation is inserted, which assumes a white page background.
- **PDF text overflow**: if translated text doesn't fit its original bounding box even after
  shrinking the font to a 5pt floor, it's inserted at the floor size anyway and a warning is
  logged with the affected segment IDs — there's no automatic reflow.
- **thaillm.or.th**: its exact API contract wasn't available at build time. The adapter assumes
  an OpenAI-chat-like request/response shape. If the real contract differs, add `chat_path` /
  `auth_header` / `auth_scheme` / `response_path` keys directly under `providers.thaillm` in
  `config/default.yaml` rather than editing code — see `src/doctranslate/llm/thaillm_client.py`.
- **PDF scope**: digital/text PDFs only; scanned/image-only PDFs would need OCR first, which is
  out of scope.
