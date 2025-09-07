# Legal Document Change Analyzer

A FastAPI-based service that compares two versions of legal documents (PDFs) and provides AI-powered analysis of changes, classifying them by legal significance.

## Setup

### Preliminaries

Add an OpenAI API key to a `.env` file (`OPENAI_API_KEY=sk-...`) in the project root or as an environment variable.

### Install dependencies
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Start the API server

```bash
source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```

### Send PDF comparison request

```bash
curl -X POST http://localhost:8000/pdf-diff \
  -F "file1=@path/to/first.pdf" \
  -F "file2=@path/to/second.pdf" \
  -o output.json
```

Example files can be found in the examples/ folder, so you can use `file1=@example/1.pdf` and `file2=@example/2.pdf`.

## Change Types

- **unchanged** - Content identical in both versions
- **modified** - Content changed in place
- **added** - New content in the new version (file2)
- **removed** - Content deleted from the old version (file1)
- **moved** - Content relocated to a different position
- **moved_and_modified** - Content both relocated and changed (similarity > 0.7)

## Classification Categories

- **Critical** - Alters material rights, obligations, or legal meaning
- **Minor** - Clarifies intent or improves language without changing legal substance
- **Formatting** - Only presentation changes (punctuation, capitalization, layout)

## Major Libraries

- **pymupdf4llm** - PDF to Markdown extraction
- **diff_match_patch** - Character-level text comparison
- **patiencediff** - Block-level structural comparison
- **FastAPI** - Web API framework
- **OpenAI API** - AI-powered change classification and impact analysis

## API Response

Returns JSON with analyzed changes including:
- Change type and location
- Content differences with inline diff
- Classification (Critical/Minor/Formatting)
- Impact analysis for critical changes (legal implications, affected party, severity)