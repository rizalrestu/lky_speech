"""Streamlit chat UI."""
import traceback

import streamlit as st

from core import answer_question, load_resources

st.set_page_config(page_title="What Would Lee Kuan Yew Do?", page_icon="🇸🇬")
st.title("🇸🇬 What Would Lee Kuan Yew Do?")
st.caption("RAG chatbot over speeches, interviews and writings of Lee Kuan Yew. "
           "Answers are drafted by a small local model from retrieved passages — "
           "check the sources before quoting anything.")


@st.cache_resource
def get_resources():
    return load_resources()


embedder, client = get_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(hits):
    if not hits:
        return
    with st.expander("Sources"):
        for h in hits:
            p = h.payload
            st.markdown(f"- **{p.get('title')}** ({p.get('date')}) — score {h.score:.3f}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        render_sources(msg.get("hits"))

question = st.chat_input("Ask LKY something...", max_chars=600)
if question:
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching the archive and drafting an answer..."):
            try:
                answer, hits = answer_question(question, embedder, client)
            except Exception:
                # traceback to the server log, generic text to the browser
                traceback.print_exc()
                st.error("Something went wrong reaching the model. Please try again.")
                st.stop()
        st.markdown(answer)
        render_sources(hits)

    # only on success, so a failed turn can't desync the transcript
    st.session_state.messages.append({"role": "user", "content": question})
    st.session_state.messages.append({"role": "assistant", "content": answer, "hits": hits})
