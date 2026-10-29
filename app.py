"""Optional Streamlit interface: python -m streamlit run app.py."""

import streamlit as st

from sourcegrounded.corpus import load_sources
from sourcegrounded.llm import ChatClient, ProviderError, load_local_env
from sourcegrounded.pipeline import ConversationState, MODES, TextGenerationPipeline


load_local_env()
st.set_page_config(page_title="SourceGrounded AI", page_icon="📚", layout="wide")
st.title("📚 SourceGrounded: GenAI Q&A")
st.caption("AAI 6665 | Bhagyasri Uddandam | Retrieval-augmented text generation")
st.warning(
    "An educational prototype, not an authoritative fact checker. "
    "Citation IDs are structurally verified; passage support requires review.",
    icon="⚠️",
)

with st.sidebar:
    st.header("Configuration")
    source_path = st.text_input("Source collection", "data/demo_sources.json")
    mode = st.selectbox("Generation condition", MODES, index=2)
    top_k = st.slider("Top K passages", 1, 8, 3)
    min_score = st.slider("Minimum TF-IDF cosine score", 0.0, 0.5, 0.08, 0.01)
    st.caption("For live generations set GROQ_API_KEY in a local .env file.")
    if st.button("Reset conversation"):
        st.session_state["clarification_state"] = ConversationState()
        st.session_state["history"] = []
        st.rerun()

if "clarification_state" not in st.session_state:
    st.session_state["clarification_state"] = ConversationState()
if "history" not in st.session_state:
    st.session_state["history"] = []

for entry in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(entry["question"])
    with st.chat_message("assistant"):
        st.write(entry["answer"])
        st.caption(f"Evidence: {entry['evidence_status']} | Citations: {', '.join(entry['cited_source_ids']) or 'none'}")

question = st.chat_input("Ask a question about the approved source collection")
if question:
    try:
        pipeline = TextGenerationPipeline(
            load_sources(source_path), ChatClient(), top_k=top_k, min_score=min_score
        )
        response = pipeline.answer(question, mode=mode, state=st.session_state["clarification_state"])
    except (OSError, ValueError, RuntimeError, ProviderError) as exc:
        st.error(f"Cannot answer: {exc}")
    else:
        st.session_state["history"].append(response.to_mapping())
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            st.write(response.answer)
            st.caption(f"Evidence status: {response.evidence_status}")
            if response.warnings:
                st.info(" ".join(response.warnings))
            if response.retrieved:
                with st.expander("Retrieved evidence and source IDs"):
                    for hit in response.retrieved:
                        st.markdown(f"**[{hit['passage_id']}] {hit['title']}** (score: {hit['score']:.3f})")
                        st.write(hit["text"])
                        if hit["url"]:
                            st.caption(hit["url"])
