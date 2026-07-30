"""Fase 5: simple Streamlit chat UI for the "What Would Lee Kuan Yew Do?"
RAG chatbot.

pip install streamlit

Run:
    streamlit run app.py
"""
import streamlit as st

from core import load_resources, retrieve, build_context, generate

st.set_page_config(page_title="What Would Lee Kuan Yew Do?", page_icon="🇸🇬")
st.title("🇸🇬 What Would Lee Kuan Yew Do?")
st.caption("RAG chatbot based on speeches, interviews, and writings of Lee Kuan Yew. "
           "Answers are generated from retrieved context, not model hallucination.")


@st.cache_resource
def get_resources():
    return load_resources()


embedder, client = get_resources()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("Ask LKY something...")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.status("Analyzing request...", expanded=True) as status:
            step1 = st.empty()
            step1.markdown("⏳ **Step 1:** Searching for relevant speeches in the vector database...")
            
            hits = retrieve(question, embedder, client)
            
            if not hits:
                step1.markdown("❌ **Step 1:** No relevant records found.")
                status.update(label="No sources found", state="error")
                answer = "I'm sorry, but I couldn't find any relevant public records regarding this matter."
            else:
                step1.markdown(f"✅ **Step 1:** Found {len(hits)} relevant documents.")
                
                step2 = st.empty()
                step2.markdown("⏳ **Step 2:** Building and sorting context from documents...")
                
                context = build_context(hits)
                
                step2.markdown("✅ **Step 2:** Context successfully built.")
                
                step3 = st.empty()
                step3.markdown("⏳ **Step 3:** Prompting LLM to formulate a response (this may take a few moments)...")
                
                answer = generate(question, context)
                
                step3.markdown("✅ **Step 3:** Response successfully generated.")
                status.update(label="Analysis complete!", state="complete", expanded=False)
        
        st.markdown(answer)
        
        if hits:
            with st.expander("Sources"):
                for h in hits:
                    p = h.payload
                    st.markdown(f"- **{p.get('title')}** ({p.get('date')}) — score {h.score:.3f}")

    st.session_state.messages.append({"role": "assistant", "content": answer})
