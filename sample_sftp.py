
"""
Quick Streamlit UI to test repo‑agent powered by Microsoft Phi‑3‑Mini (MLX).

Launch with:
    streamlit run repo_agent_ui.py

Prerequisites (once):
    pip install streamlit smolagents mlx-lm ripgrep
"""

from pathlib import Path
import streamlit as st
from smolagents import CodeAgent, MLXModel
from repo_search_tool import RepoSearchTool

# -----------------------------------------------------------------------------
# One‑time lazy loader so the heavy model is initialised just once per process
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_agent(repo_root: Path = Path(".")) -> CodeAgent:
    """Download/load the Phi‑3 model and wire it with the repo‑search tool."""
    model = MLXModel(
        model_id="microsoft/phi-3-mini-4k-instruct-gguf",
        max_tokens=1536,
        temperature=0.0,
        top_p=0.9,
        streaming=True,
    )

    return CodeAgent(
        tools=[RepoSearchTool()],
        model=model,
        add_base_tools=False,
        max_steps=5,
    )

# -----------------------------------------------------------------------------
# Streamlit UI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Repo Q&A – Phi‑3 Mini", page_icon="🤖", layout="wide")
st.title("🤖 Repo Q&A with Phi‑3‑Mini (MLX)")

query = st.text_input(
    "Ask a question about the codebase",
    placeholder="e.g. Where is get_finmodelprep_secret defined?",
)

col1, col2 = st.columns([1, 3])
with col1:
    run_btn = st.button("Run", type="primary")
with col2:
    repo_root = st.text_input("Repo root (optional)", value=".")

if run_btn and query:
    agent = load_agent(Path(repo_root))
    with st.spinner("Thinking …"):
        answer: str = agent.run(query)

    st.markdown("### Answer")
    st.code(answer, language="text")
