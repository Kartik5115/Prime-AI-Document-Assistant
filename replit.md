# Document Assistant

An AI-powered PDF Q&A application built with Streamlit and Google Gemini 1.5 Flash. Upload any PDF and ask questions about its content in a chat interface.

## Run & Operate

- `streamlit run main.py` — run the Document Assistant (port 8000)
- `pnpm --filter @workspace/api-server run dev` — run the API server (port 5000)
- `pnpm run typecheck` — full typecheck across all packages
- `pnpm run build` — typecheck + build all packages

## Stack

- Python 3.10+, Streamlit 1.32+
- Google Gemini 1.5 Flash (`google-genai` SDK)
- PyPDF2 — PDF text extraction
- python-dotenv — local env variable loading
- pnpm workspaces, Node.js 24, TypeScript 5.9 (shared backend)
- API: Express 5 | DB: PostgreSQL + Drizzle ORM

## Where things live

- `main.py` — Streamlit app entry point (Document Assistant)
- `requirements.txt` — Python dependencies
- `.streamlit/config.toml` — Streamlit server & Prime Track dark theme
- `lib/api-spec/openapi.yaml` — API contract source of truth
- `artifacts/api-server/src/` — Express route handlers

## Architecture decisions

- PDF text is extracted once on upload and stored in `st.session_state` to avoid re-processing on every interaction.
- Conversation history (last 10 turns) is injected into each Gemini prompt for multi-turn awareness without overrunning context limits.
- Gemini temperature is fixed at 0.2 to keep answers factual and grounded in the document.
- The Gemini client is cached with `@st.cache_resource` so only one connection is created per server session.
- GEMINI_API_KEY is stored as a Replit Secret — never in the filesystem.

## Product

- Upload any PDF in the left sidebar
- Extracted text is stored in session state; word/char counts are shown
- Chat interface powered by Gemini 1.5 Flash answers questions based solely on document content
- "Clear Document" button resets the session for a new PDF

## User preferences

- Prime Track dark mode aesthetic (deep navy/black background, violet accent #7C3AED, monospace font)
- Wide layout dashboard with sidebar for controls

## Gotchas

- Port 8080 is occupied by the mockup-sandbox artifact — Streamlit runs on port 8000
- PyPDF2 cannot extract text from image-based or encrypted PDFs; users see a clear error message
- Run `pip install -r requirements.txt` if starting fresh on a new machine

## Pointers

- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
