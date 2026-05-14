"""
Document Assistant
==================
AI-powered PDF Q&A tool using Google Gemini 1.5 Flash.

Upload any PDF document and ask questions about its content.
The assistant extracts text from the PDF and uses Gemini to provide
accurate, context-aware answers based solely on the document.

Usage:
    streamlit run main.py

Requirements:
    GEMINI_API_KEY environment variable must be set (via .env or system env).
"""

import os

import streamlit as st
from PyPDF2 import PdfReader
from google import genai
from google.genai import types
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# Bootstrap
# ─────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Prime Track Dark Mode — Custom CSS
# ─────────────────────────────────────────────

st.markdown(
    """
    <style>
        /* ── Root & scrollbar ── */
        :root {
            --prime: #7C3AED;
            --prime-glow: rgba(124, 58, 237, 0.25);
            --bg: #0A0A0F;
            --surface: #12121A;
            --surface-2: #1A1A28;
            --border: rgba(124, 58, 237, 0.18);
            --text: #E2E8F0;
            --muted: #64748B;
            --success: #10B981;
        }

        html, body, [data-testid="stAppViewContainer"] {
            background-color: var(--bg) !important;
        }

        ::-webkit-scrollbar { width: 6px; height: 6px; }
        ::-webkit-scrollbar-track { background: var(--surface); }
        ::-webkit-scrollbar-thumb { background: var(--prime); border-radius: 4px; }

        /* ── Sidebar ── */
        [data-testid="stSidebar"] {
            background-color: var(--surface) !important;
            border-right: 1px solid var(--border) !important;
        }

        [data-testid="stSidebar"] * {
            color: var(--text) !important;
        }

        /* ── Main area ── */
        [data-testid="stMain"] {
            background-color: var(--bg) !important;
        }

        /* ── Headings & text ── */
        h1, h2, h3, h4, h5, h6 { color: var(--text) !important; }
        p, span, label { color: var(--text) !important; }

        /* ── File uploader ── */
        [data-testid="stFileUploaderDropzone"] {
            background-color: var(--surface-2) !important;
            border: 1px dashed var(--prime) !important;
            border-radius: 10px !important;
        }

        /* ── Chat messages ── */
        [data-testid="stChatMessage"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 12px !important;
            margin-bottom: 8px !important;
            padding: 12px 16px !important;
        }

        /* ── Chat input ── */
        [data-testid="stChatInput"] textarea {
            background-color: var(--surface-2) !important;
            border: 1px solid var(--border) !important;
            color: var(--text) !important;
            border-radius: 12px !important;
        }

        [data-testid="stChatInput"] textarea:focus {
            border-color: var(--prime) !important;
            box-shadow: 0 0 0 3px var(--prime-glow) !important;
        }

        /* ── Metric cards ── */
        [data-testid="stMetric"] {
            background-color: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            padding: 14px !important;
        }

        [data-testid="stMetricValue"] { color: var(--prime) !important; font-weight: 700 !important; }
        [data-testid="stMetricLabel"] { color: var(--muted) !important; }

        /* ── Success / info banners ── */
        [data-testid="stAlert"] {
            border-radius: 10px !important;
            border-left: 3px solid var(--prime) !important;
        }

        /* ── Divider ── */
        hr { border-color: var(--border) !important; }

        /* ── Spinner text ── */
        .stSpinner > div > div { border-top-color: var(--prime) !important; }

        /* ── Glow tag for the header ── */
        .prime-tag {
            display: inline-block;
            background: linear-gradient(135deg, var(--prime), #9333EA);
            color: #fff !important;
            font-size: 0.7rem;
            font-weight: 700;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            padding: 2px 10px;
            border-radius: 999px;
            margin-left: 10px;
            vertical-align: middle;
            box-shadow: 0 0 10px var(--prime-glow);
        }

        .header-line {
            border-bottom: 1px solid var(--border);
            padding-bottom: 12px;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    """Extract all text from a PDF file object.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Concatenated text content from all pages.
    """
    reader = PdfReader(uploaded_file)
    pages = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages.append(text.strip())
    return "\n\n".join(pages)


def build_prompt(document_text: str, question: str, history: list[dict]) -> str:
    """Construct the full prompt sent to Gemini.

    The prompt instructs the model to answer solely from the document and
    includes the conversation history for multi-turn awareness.

    Args:
        document_text: Extracted text from the PDF.
        question: The user's latest question.
        history: Prior messages [{role, content}, ...].

    Returns:
        Formatted prompt string.
    """
    history_block = ""
    if history:
        turns = []
        for msg in history[-10:]:  # keep last 10 turns to stay within context
            role_label = "User" if msg["role"] == "user" else "Assistant"
            turns.append(f"{role_label}: {msg['content']}")
        history_block = "\n".join(turns) + "\n"

    return f"""You are a precise document assistant. Answer the user's question using ONLY the information found in the document below.
If the answer is not present in the document, say so clearly — do not fabricate information.
Be concise, accurate, and cite relevant parts of the document when helpful.

--- DOCUMENT START ---
{document_text}
--- DOCUMENT END ---

{history_block}User: {question}
Assistant:"""


@st.cache_resource(show_spinner=False)
def get_gemini_client() -> genai.Client:
    """Initialise and cache the Gemini client.

    Prefers the Replit-managed integration endpoint (no user quota consumed).
    Falls back to the user-supplied GEMINI_API_KEY if the integration vars
    are not present.

    Returns:
        Authenticated google.genai Client instance.

    Raises:
        ValueError: If neither integration vars nor GEMINI_API_KEY are set.
    """
    integration_base_url = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL", "").strip()
    integration_api_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY", "").strip()

    if integration_base_url and integration_api_key:
        return genai.Client(
            api_key=integration_api_key,
            http_options={"base_url": integration_base_url, "api_version": ""},
        )

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "No Gemini credentials found. "
            "Add GEMINI_API_KEY to your environment secrets."
        )
    return genai.Client(api_key=api_key)


def query_gemini(document_text: str, question: str, history: list[dict]) -> str:
    """Send a question to Gemini 1.5 Flash and return the answer.

    Args:
        document_text: Full extracted PDF text.
        question: Current user question.
        history: Prior conversation turns.

    Returns:
        Model response as a plain string.
    """
    client = get_gemini_client()
    prompt = build_prompt(document_text, question, history)

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2,        # lower = more factual / less creative
            max_output_tokens=1024,
        ),
    )
    return response.text.strip()


# ─────────────────────────────────────────────
# Session-state initialisation
# ─────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []

if "document_text" not in st.session_state:
    st.session_state.document_text: str = ""

if "pdf_meta" not in st.session_state:
    st.session_state.pdf_meta: dict = {}


# ─────────────────────────────────────────────
# Sidebar — PDF Upload & Document Info
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<h2 style="margin-bottom:4px;">📄 Document Assistant</h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<span class="prime-tag">Prime Track</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown("### Upload PDF")
    uploaded_file = st.file_uploader(
        label="Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to start asking questions about it.",
        label_visibility="collapsed",
    )

    if uploaded_file is not None:
        file_changed = (
            st.session_state.pdf_meta.get("name") != uploaded_file.name
            or st.session_state.pdf_meta.get("size") != uploaded_file.size
        )

        if file_changed:
            with st.spinner("Extracting text from PDF…"):
                try:
                    extracted = extract_pdf_text(uploaded_file)
                    if not extracted.strip():
                        st.error("No readable text found. The PDF may be image-based or encrypted.")
                    else:
                        st.session_state.document_text = extracted
                        st.session_state.pdf_meta = {
                            "name": uploaded_file.name,
                            "size": uploaded_file.size,
                        }
                        # Clear previous conversation when a new doc is loaded
                        st.session_state.messages = []
                        st.rerun()
                except Exception as exc:
                    st.error(f"Failed to read PDF: {exc}")

    # ── Document stats ──
    if st.session_state.document_text:
        st.markdown("---")
        st.markdown("### Document Info")
        word_count = len(st.session_state.document_text.split())
        char_count = len(st.session_state.document_text)

        col1, col2 = st.columns(2)
        col1.metric("Words", f"{word_count:,}")
        col2.metric("Chars", f"{char_count:,}")

        st.caption(f"**File:** {st.session_state.pdf_meta.get('name', '—')}")

        if st.button("🗑️ Clear Document", use_container_width=True):
            st.session_state.document_text = ""
            st.session_state.pdf_meta = {}
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption("Powered by **Gemini 2.5 Flash**")
    st.caption("Model: `gemini-2.5-flash` · Temp: 0.2")


# ─────────────────────────────────────────────
# Main Area — Chat Interface
# ─────────────────────────────────────────────

# Header
st.markdown(
    """
    <div class="header-line">
        <h1 style="font-size:1.8rem; margin:0;">
            Document Assistant
            <span class="prime-tag">Prime Track</span>
        </h1>
        <p style="margin:6px 0 0; color:#64748B; font-size:0.9rem;">
            Upload a PDF in the sidebar, then ask any question about it.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── No document yet ──
if not st.session_state.document_text:
    st.info("👈 Upload a PDF in the sidebar to get started.")

    st.markdown("#### What you can do")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**📋 Summarise**\nGet a concise summary of any document instantly.")
    c2.markdown("**🔍 Extract data**\nPull out names, dates, figures, or facts.")
    c3.markdown("**💬 Deep-dive Q&A**\nAsk follow-up questions for full context.")

else:
    # ── Render existing messages ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Chat input ──
    if question := st.chat_input("Ask a question about your document…"):
        # Show user message immediately
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        # Generate and stream assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = query_gemini(
                        document_text=st.session_state.document_text,
                        question=question,
                        history=st.session_state.messages[:-1],  # exclude current question
                    )
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except ValueError as exc:
                    err = str(exc)
                    st.error(f"⚠️ Configuration error: {err}")
                except Exception as exc:
                    st.error(f"⚠️ Gemini API error: {exc}")
