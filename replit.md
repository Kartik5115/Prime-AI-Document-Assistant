# Document Assistant

An AI-powered PDF Q&A application built with **pure Python** — designed for an
Electronics & Communication (ECE) and AI/ML portfolio.

## Stack

| Layer       | Technology                          |
|-------------|-------------------------------------|
| UI          | Streamlit                           |
| AI Model    | Google Gemini 2.5 Flash             |
| PDF Parsing | PyPDF2                              |
| Language    | Python 3.11                         |

## Run

```bash
streamlit run main.py
```

## File Structure

```
.
├── main.py                  # Application entry point (all logic here)
├── requirements.txt         # Python dependencies
├── .streamlit/
│   └── config.toml          # Dark theme + server config
└── replit.md                # This file
```

## Architecture

- **PDF extraction** — PyPDF2 reads uploaded files once; text is stored in
  `st.session_state` so re-uploads are never needed during a session.
- **Conversation history** — the last 10 turns are injected into every Gemini
  prompt for multi-turn awareness without overrunning the context window.
- **Gemini client** — cached with `@st.cache_resource`; one connection per
  server session. Prefers Replit-managed credentials; falls back to
  `GEMINI_API_KEY` secret.
- **Temperature** — fixed at 0.2 to keep answers factual and document-grounded.

## Environment Secrets

| Key                              | Purpose                                   |
|----------------------------------|-------------------------------------------|
| `GEMINI_API_KEY`                 | Your Google AI Studio key (fallback)      |
| `AI_INTEGRATIONS_GEMINI_BASE_URL`| Replit-managed Gemini proxy (auto-set)    |
| `AI_INTEGRATIONS_GEMINI_API_KEY` | Proxy dummy key (auto-set)                |

## User Preferences

- Prime Track dark mode — deep navy `#0F172A` bg, violet `#7C3AED` accent
- Wide layout dashboard, sidebar for controls, monospace font
