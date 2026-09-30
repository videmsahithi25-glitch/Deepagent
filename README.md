# DeepAgent

A multi-agent assistant built with LangGraph and Streamlit.

- Intent engine: a stack of gates decides whether a message is one focused question or a broad topic to explore.
- Document search: reads uploaded PDF and Word files, extracts tables and images (with OCR), and searches them using embeddings plus keyword scoring.
- Web search: falls back to Tavily when the answer isn't in your documents.
- Preference memory: a ChromaDB store learns which topics you return to.
- Subagents for research and for generating timeline graphs.

## Run
1. Install the packages the code imports
2. Add your API keys to a `.env` file
3. `streamlit run imatb.py`
