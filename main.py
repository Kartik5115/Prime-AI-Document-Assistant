"""
Document Assistant — ECE & AI/ML Portfolio Project
====================================================
An AI-powered PDF Q&A tool built with pure Python.

Stack:
    - Streamlit  : web UI
    - google-genai: Gemini 2.5 Flash inference
    - PyPDF2     : PDF text extraction

Usage:
    streamlit run main.py

Environment:
    GEMINI_API_KEY              — your Google AI Studio key (fallback)
    AI_INTEGRATIONS_GEMINI_BASE_URL — Replit-managed Gemini proxy (preferred)
    AI_INTEGRATIONS_GEMINI_API_KEY  — Replit proxy dummy key
"""

import os
import streamlit as st
from PyPDF2 import PdfReader
from google import genai
from google.genai import types

# ── Constants ────────────────────────────────────────────────────────────────

MODEL_ID = "gemini-2.5-flash"
TEMPERATURE = 0.2          # low → more factual answers
MAX_OUTPUT_TOKENS = 8192
MAX_HISTORY_TURNS = 10     # keep last N turns to manage context length

# ── Page config (must be first Streamlit call) ────────────────────────────────

st.set_page_config(
    page_title="Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — Prime Track dark theme ───────────────────────────────────────

st.markdown("""
<style>
    /* Global background */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #0F172A !important;
    }
    [data-testid="stMain"] { background-color: #0F172A !important; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1E293B !important;
        border-right: 1px solid rgba(124,58,237,0.2) !important;
    }
    [data-testid="stSidebar"] * { color: #E2E8F0 !important; }

    /* Headings & body text */
    h1, h2, h3, h4, h5, h6, p, span, label { color: #E2E8F0 !important; }

    /* File uploader drop zone */
    [data-testid="stFileUploaderDropzone"] {
        background-color: #1E293B !important;
        border: 1px dashed #7C3AED !important;
        border-radius: 10px !important;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #1E293B !important;
        border: 1px solid rgba(124,58,237,0.15) !important;
        border-radius: 12px !important;
        margin-bottom: 8px !important;
    }

    /* Chat input */
    [data-testid="stChatInput"] textarea {
        background-color: #1E293B !important;
        border: 1px solid rgba(124,58,237,0.3) !important;
        color: #E2E8F0 !important;
        border-radius: 12px !important;
    }
    [data-testid="stChatInput"] textarea:focus {
        border-color: #7C3AED !important;
        box-shadow: 0 0 0 3px rgba(124,58,237,0.2) !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #1E293B !important;
        border: 1px solid rgba(124,58,237,0.15) !important;
        border-radius: 10px !important;
        padding: 12px !important;
    }
    [data-testid="stMetricValue"] { color: #7C3AED !important; font-weight: 700 !important; }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 5px; }
    ::-webkit-scrollbar-track { background: #1E293B; }
    ::-webkit-scrollbar-thumb { background: #7C3AED; border-radius: 4px; }

    /* Divider */
    hr { border-color: rgba(124,58,237,0.2) !important; }

    /* Badge */
    .badge {
        display: inline-block;
        background: linear-gradient(135deg, #7C3AED, #9333EA);
        color: #fff !important;
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        padding: 2px 9px;
        border-radius: 999px;
        margin-left: 8px;
        vertical-align: middle;
    }
</style>
""", unsafe_allow_html=True)


# ── Gemini client (cached for the server session) ────────────────────────────

@st.cache_resource(show_spinner=False)
def get_gemini_client() -> genai.Client:
    """
    Return an authenticated Gemini client.

    Priority:
        1. Replit-managed integration (no personal quota consumed).
        2. User-supplied GEMINI_API_KEY as fallback.

    Raises:
        ValueError: when no credentials are available.
    """
    base_url = os.environ.get("AI_INTEGRATIONS_GEMINI_BASE_URL", "").strip()
    proxy_key = os.environ.get("AI_INTEGRATIONS_GEMINI_API_KEY", "").strip()

    if base_url and proxy_key:
        # api_version="" strips the /v1beta prefix expected by the proxy
        return genai.Client(
            api_key=proxy_key,
            http_options={"base_url": base_url, "api_version": ""},
        )

    user_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if user_key:
        return genai.Client(api_key=user_key)

    raise ValueError(
        "No Gemini credentials found. "
        "Set GEMINI_API_KEY in your environment secrets."
    )


# ── PDF helpers ───────────────────────────────────────────────────────────────

def extract_pdf_text(uploaded_file) -> str:
    """
    Extract plain text from every page of a PDF.

    Args:
        uploaded_file: Streamlit UploadedFile object.

    Returns:
        Concatenated page text (pages separated by blank lines).

    Raises:
        RuntimeError: if PyPDF2 cannot read the file.
    """
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(p.strip() for p in pages if p.strip())


# ── Prompt construction ───────────────────────────────────────────────────────

def build_prompt(doc_text: str, question: str, history: list[dict]) -> str:
    """
    Build the Gemini prompt with document context and conversation history.

    Args:
        doc_text : Full extracted PDF text.
        question : The user's current question.
        history  : Previous chat turns [{role, content}, ...].

    Returns:
        Formatted prompt string ready to send to the model.
    """
    history_block = ""
    if history:
        recent = history[-MAX_HISTORY_TURNS:]
        lines = []
        for msg in recent:
            role = "User" if msg["role"] == "user" else "Assistant"
            lines.append(f"{role}: {msg['content']}")
        history_block = "\n".join(lines) + "\n"

    return (
        "You are a precise document assistant for engineering and technical documents.\n"
        "Answer the question using ONLY the information in the document below.\n"
        "If the answer is not in the document, say so clearly — do not guess.\n\n"
        f"--- DOCUMENT START ---\n{doc_text}\n--- DOCUMENT END ---\n\n"
        f"{history_block}"
        f"User: {question}\n"
        "Assistant:"
    )


# ── Gemini inference ──────────────────────────────────────────────────────────

def ask_gemini(doc_text: str, question: str, history: list[dict]) -> str:
    """
    Send a prompt to Gemini and return the model's text response.

    Args:
        doc_text : Extracted PDF content.
        question : Current user question.
        history  : Prior conversation turns.

    Returns:
        Model answer as a plain string.
    """
    client = get_gemini_client()
    prompt = build_prompt(doc_text, question, history)

    response = client.models.generate_content(
        model=MODEL_ID,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=TEMPERATURE,
            max_output_tokens=MAX_OUTPUT_TOKENS,
        ),
    )
    return response.text.strip()


# ── Session state initialisation ──────────────────────────────────────────────

def init_session() -> None:
    """Initialise session-state keys on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages: list[dict] = []
    if "doc_text" not in st.session_state:
        st.session_state.doc_text: str = ""
    if "doc_meta" not in st.session_state:
        st.session_state.doc_meta: dict = {}


init_session()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown(
        '<h2 style="margin-bottom:2px;">📄 Document Assistant'
        '<span class="badge">Prime Track</span></h2>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p style="color:#94A3B8;font-size:0.8rem;margin-top:0;">'
        'ECE &amp; AI/ML Portfolio</p>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── PDF uploader ──
    st.markdown("#### Upload PDF")
    uploaded = st.file_uploader(
        label="Choose a PDF",
        type=["pdf"],
        label_visibility="collapsed",
        help="Upload any PDF — lecture notes, datasheets, research papers…",
    )

    if uploaded is not None:
        new_file = (
            st.session_state.doc_meta.get("name") != uploaded.name
            or st.session_state.doc_meta.get("size") != uploaded.size
        )
        if new_file:
            with st.spinner("Extracting text…"):
                try:
                    text = extract_pdf_text(uploaded)
                    if not text:
                        st.error(
                            "No readable text found. "
                            "The PDF may be scanned or encrypted."
                        )
                    else:
                        st.session_state.doc_text = text
                        st.session_state.doc_meta = {
                            "name": uploaded.name,
                            "size": uploaded.size,
                        }
                        st.session_state.messages = []
                        st.rerun()
                except Exception as exc:
                    st.error(f"Failed to read PDF: {exc}")

    # ── Document stats ──
    if st.session_state.doc_text:
        st.markdown("---")
        st.markdown("#### Document Info")
        words = len(st.session_state.doc_text.split())
        chars = len(st.session_state.doc_text)
        col1, col2 = st.columns(2)
        col1.metric("Words", f"{words:,}")
        col2.metric("Chars", f"{chars:,}")
        st.caption(f"**File:** {st.session_state.doc_meta.get('name', '—')}")
        st.markdown("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        if st.button("📂 Remove Document", use_container_width=True):
            st.session_state.doc_text = ""
            st.session_state.doc_meta = {}
            st.session_state.messages = []
            st.rerun()

    st.markdown("---")
    st.caption(f"Model · `{MODEL_ID}`")
    st.caption("Powered by **Google Gemini**")


# ── Main chat area ────────────────────────────────────────────────────────────

st.markdown(
    '<h1 style="font-size:1.9rem;margin-bottom:4px;">'
    'Document Assistant'
    '<span class="badge">Prime Track</span>'
    '</h1>'
    '<p style="color:#94A3B8;margin-top:0;font-size:0.9rem;">'
    'Upload a PDF in the sidebar, then ask any question about it.'
    '</p><hr>',
    unsafe_allow_html=True,
)

if not st.session_state.doc_text:
    # ── Landing state ──
    st.info("👈 Upload a PDF in the sidebar to get started.")
    c1, c2, c3 = st.columns(3)
    c1.markdown("**📋 Summarise**\nGet a quick summary of any document.")
    c2.markdown("**🔍 Extract facts**\nPull out names, values, or key data.")
    c3.markdown("**💬 Deep-dive Q&A**\nAsk follow-up questions naturally.")

else:
    # ── Render existing conversation ──
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ── Accept new question ──
    if question := st.chat_input("Ask a question about your document…"):
        with st.chat_message("user"):
            st.markdown(question)
        st.session_state.messages.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    answer = ask_gemini(
                        doc_text=st.session_state.doc_text,
                        question=question,
                        history=st.session_state.messages[:-1],
                    )
                    st.markdown(answer)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": answer}
                    )
                except ValueError as exc:
                    st.error(f"⚠️ Configuration error: {exc}")
                except Exception as exc:
                    st.error(f"⚠️ Gemini error: {exc}")
