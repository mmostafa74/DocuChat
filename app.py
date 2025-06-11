import streamlit as st
from config import config
from openrouter_client import get_openrouter_client, chat_with_openrouter
from utils import (
    get_embeddings,
    load_documents_from_files,
    create_vector_store,
    get_relevant_context,
    create_rag_prompt,
)
from datetime import datetime
import time

# App configuration
st.set_page_config(
    page_title=config.app["title"],
    page_icon=config.app["page_icon"],
    layout=config.ui["layout"],
    initial_sidebar_state=config.ui["initial_sidebar_state"],
)

# Apply CSS from config
st.markdown(f"<style>{config.styles['css']}</style>", unsafe_allow_html=True)


# Initialize resources
@st.cache_resource
def init_client():
    return get_openrouter_client()


@st.cache_resource
def init_embeddings():
    return get_embeddings()


# Initialize
client = init_client()
embeddings = init_embeddings()

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "response_times" not in st.session_state:
    st.session_state.response_times = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "documents" not in st.session_state:
    st.session_state.documents = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# Enhanced Sidebar
with st.sidebar:
    # Header with logo/icon
    st.markdown(
        """
        <div class="sidebar-header">
            <h1>🤖 AI Assistant</h1>
            <p>Powered by OpenRouter</p>
        </div>
    """,
        unsafe_allow_html=True,
    )

    # Document Upload Section
    st.markdown(
        '<div class="section-header">📁 Document Management</div>',
        unsafe_allow_html=True,
    )

    # File upload
    uploaded_files = st.file_uploader(
        "",
        type=["pdf", "txt", "csv", "doc", "docx"],
        accept_multiple_files=True,
        help="📎 Drag and drop your documents here or click to browse",
        label_visibility="collapsed",
    )

    # File upload info
    if uploaded_files:
        st.markdown(
            f"""
            <div class="status-info">
                📄 {len(uploaded_files)} file(s) selected
            </div>
        """,
            unsafe_allow_html=True,
        )

        # Show file details
        with st.expander("📋 File Details", expanded=False):
            for file in uploaded_files:
                file_size = len(file.getvalue()) / 1024  # KB
                st.markdown(
                    f"""
                    <div class="doc-item">
                        📄 {file.name}<br>
                        <small>Size: {file_size:.1f} KB | Type: {file.type}</small>
                    </div>
                """,
                    unsafe_allow_html=True,
                )

    # Process button with enhanced styling
    col1, col2 = st.columns([3, 1])
    with col1:
        if uploaded_files:
            process_clicked = st.button(
                "🚀 Process Documents",
                key="process_btn",
                help="Click to process and index your documents",
                use_container_width=True,
            )
            # Apply process button styling
            st.markdown(
                '<style>.stButton:has([key="process_btn"]) { .process-btn; }</style>',
                unsafe_allow_html=True,
            )
        else:
            st.button("📁 Select Files First", disabled=True, use_container_width=True)

    with col2:
        if st.session_state.documents:
            if st.button("🗑️", help="Clear all documents", key="clear_docs"):
                st.session_state.documents = []
                st.session_state.vectorstore = None
                st.rerun()
            # Apply clear button styling
            st.markdown(
                '<style>.stButton:has([key="clear_docs"]) { .clear-btn; }</style>',
                unsafe_allow_html=True,
            )

    # Processing indicator
    if uploaded_files and "process_clicked" in locals() and process_clicked:
        st.session_state.processing = True

    if st.session_state.processing:
        st.markdown(
            '<div class="processing-indicator">⚡ Processing documents...</div>',
            unsafe_allow_html=True,
        )

        documents = load_documents_from_files(uploaded_files)
        if documents:
            st.session_state.documents = documents
            st.session_state.vectorstore = create_vector_store(documents, embeddings)
            st.session_state.processing = False
            st.markdown(
                '<div class="status-success">✅ Documents processed successfully!</div>',
                unsafe_allow_html=True,
            )
            time.sleep(1)  # Brief pause to show success
            st.rerun()
        else:
            st.session_state.processing = False
            st.markdown(
                '<div class="status-warning">❌ Failed to process documents</div>',
                unsafe_allow_html=True,
            )

    # Display loaded documents with enhanced styling
    if st.session_state.documents:
        st.markdown(
            '<div class="section-header">📚 Loaded Documents</div>',
            unsafe_allow_html=True,
        )
        sources = list(
            set(
                [
                    doc.metadata.get("source", "Unknown")
                    for doc in st.session_state.documents
                ]
            )
        )

        for i, source in enumerate(sources, 1):
            st.markdown(
                f"""
                <div class="doc-item">
                    <strong>{i}.</strong> {source}
                </div>
            """,
                unsafe_allow_html=True,
            )

        # Document stats
        total_chunks = len(st.session_state.documents)
        st.markdown(
            f"""
            <div class="status-success">
                📊 {len(sources)} documents • {total_chunks} chunks
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Model Configuration Section
    st.markdown(
        '<div class="section-header">⚙️ Model Settings</div>', unsafe_allow_html=True
    )

    # Model selection with descriptions
    model_options = {
        "mistralai/mistral-7b-instruct:free": "🔥 Mistral 7B - Fast & Efficient",
        "deepseek/deepseek-chat-v3-0324:free": "🧠 DeepSeek - Advanced Reasoning",
        "google/gemini-2.0-flash-exp:free": "⚡ Gemini Flash - Lightning Fast",
    }

    selected_model = st.selectbox(
        "🤖 AI Model:",
        options=list(model_options.keys()),
        format_func=lambda x: model_options[x],
        index=0,
        help="Choose the AI model for your conversations",
    )

    # Advanced settings in expander
    with st.expander("🔧 Advanced Settings", expanded=False):
        temperature = st.slider(
            "🌡️ Temperature:",
            min_value=0.0,
            max_value=2.0,
            value=config.app["temperature"],
            step=0.1,
            help="Controls randomness: 0 = focused, 2 = creative",
        )

        max_tokens = st.slider(
            "📝 Max Tokens:",
            min_value=100,
            max_value=4000,
            value=config.app["max_tokens"],
            step=100,
            help="Maximum length of AI responses",
        )

    # Retrieval settings (only show if documents are loaded)
    if st.session_state.vectorstore:
        st.markdown(
            '<div class="section-header">🔍 Search Settings</div>',
            unsafe_allow_html=True,
        )
        num_docs = st.slider(
            "📄 Document Chunks:",
            min_value=1,
            max_value=10,
            value=3,
            help="Number of relevant document chunks to use for context",
        )

        # Search quality indicator
        quality_labels = {
            1: "⚡ Fast",
            3: "⚖️ Balanced",
            5: "🎯 Precise",
            10: "🔬 Comprehensive",
        }
        current_quality = min(quality_labels.keys(), key=lambda x: abs(x - num_docs))
        st.markdown(
            f"""
            <div class="quality-indicator">
                Search Quality: {quality_labels.get(current_quality, "⚖️ Balanced")}
            </div>
        """,
            unsafe_allow_html=True,
        )
    else:
        num_docs = 3

    st.markdown("---")

    # Chat Statistics Section
    st.markdown(
        '<div class="section-header">📊 Chat Statistics</div>', unsafe_allow_html=True
    )

    # Enhanced metrics display
    user_msgs = len([m for m in st.session_state.messages if m["role"] == "user"])
    ai_msgs = len([m for m in st.session_state.messages if m["role"] == "assistant"])

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value" style="color: #4ECDC4;">{user_msgs}</div>
                <div class="metric-label">Your Messages</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value" style="color: #FF6B6B;">{ai_msgs}</div>
                <div class="metric-label">AI Responses</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    # Average response time
    if st.session_state.response_times:
        avg_time = sum(st.session_state.response_times) / len(
            st.session_state.response_times
        )
        st.markdown(
            f"""
            <div class="metric-container">
                <div class="metric-value" style="color: #667eea;">⏱️ {avg_time:.1f}s</div>
                <div class="metric-label">Avg Response Time</div>
            </div>
        """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Action Buttons Section
    st.markdown('<div class="section-header">🎛️ Actions</div>', unsafe_allow_html=True)

    # Clear chat button
    if st.button(config.sidebar["clear_button_text"], use_container_width=True):
        st.session_state.messages = []
        st.session_state.response_times = []
        st.rerun()

    # Export chat button
    if (
        st.button(config.sidebar["export_button_text"], use_container_width=True)
        and st.session_state.messages
    ):
        chat_export = "\n".join(
            [f"{m['role'].title()}: {m['content']}" for m in st.session_state.messages]
        )
        st.download_button(
            "📥 Download Chat",
            chat_export,
            file_name=f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

# Main chat interface
st.markdown(
    """
    <div class="main-header">
        <h1>🤖 AI Chat Assistant</h1>
        <p>Chat with AI models and your documents using OpenRouter API</p>
    </div>
""",
    unsafe_allow_html=True,
)

# Display document status
if st.session_state.vectorstore:
    st.markdown(
        """
        <div class="info-card">
            📚 Documents loaded! You can now ask questions about your uploaded files.
        </div>
    """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <div class="info-card">
            💡 Upload documents in the sidebar to enable document-based chat.
        </div>
    """,
        unsafe_allow_html=True,
    )

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.text(f"• {source}")
        if "timestamp" in message:
            st.caption(f"*{message['timestamp']}*")

# Chat input
if prompt := st.chat_input(config.sidebar["input_placeholder"]):
    # Add user message
    timestamp = datetime.now().strftime(config.chat["timestamp_format"])
    st.session_state.messages.append(
        {"role": "user", "content": prompt, "timestamp": timestamp}
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
        st.caption(f"*{timestamp}*")

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner(config.messages["thinking"]):
            start_time = time.time()

            # Check if we have documents loaded for RAG
            if st.session_state.vectorstore:
                # Use RAG approach
                context, sources = get_relevant_context(
                    st.session_state.vectorstore, prompt, num_docs
                )
                enhanced_prompt = create_rag_prompt(context, prompt)

                messages = [
                    {"role": "system", "content": config.app["system_prompt"]},
                    {"role": "user", "content": enhanced_prompt},
                ]
            else:
                # Use regular chat approach
                messages = [{"role": "system", "content": config.app["system_prompt"]}]
                messages.extend(
                    [
                        {"role": m["role"], "content": m["content"]}
                        for m in st.session_state.messages
                    ]
                )
                sources = []

            # Get response from OpenRouter
            response = chat_with_openrouter(
                client=client,
                model=selected_model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            end_time = time.time()
            response_time = end_time - start_time
            st.session_state.response_times.append(response_time)

            # Display response
            st.markdown(response)

            # Display sources if available
            if sources:
                with st.expander("📄 Sources"):
                    for source in sources:
                        st.text(f"• {source}")

            response_timestamp = datetime.now().strftime(
                config.chat["timestamp_format"]
            )
            st.caption(f"*{response_timestamp}*")

            # Add assistant message to session state
            assistant_message = {
                "role": "assistant",
                "content": response,
                "timestamp": response_timestamp,
            }
            if sources:
                assistant_message["sources"] = sources

            st.session_state.messages.append(assistant_message)

# Footer
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit and OpenRouter
    </div>
""",
    unsafe_allow_html=True,
)
