"""
Streamlit App untuk Integrasi Ollama
Aplikasi chat dengan LLM lokal menggunakan Ollama.
"""

import streamlit as st
import time
from ollama_client import OllamaClient, get_response

# Page configuration
st.set_page_config(
    page_title="Ollama Chat Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stTextInput > div > div > input {
        font-size: 16px;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #2b313e;
        border-left: 4px solid #4d8af0;
    }
    .chat-message.assistant {
        background-color: #262730;
        border-left: 4px solid #00cc88;
    }
    .message-header {
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .message-content {
        font-size: 16px;
        line-height: 1.5;
    }
    .stButton button {
        width: 100%;
    }
    .status-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #1e1e1e;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "ollama_client" not in st.session_state:
    st.session_state.ollama_client = OllamaClient()
    
if "model_checked" not in st.session_state:
    st.session_state.model_checked = False

# Title and description
st.title("🤖 Ollama Chat Assistant")
st.markdown("Chat dengan LLM lokal menggunakan Ollama. Model yang digunakan: **gemma3:1b**")

# Sidebar for configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Connection status
    st.subheader("Connection Status")
    
    if st.button("Check Ollama Connection", use_container_width=True):
        with st.spinner("Checking connection..."):
            client = st.session_state.ollama_client
            if client.check_connection():
                st.success("✅ Ollama is running!")
                models = client.get_available_models()
                if models:
                    st.info(f"Available models: {len(models)}")
                    for model in models:
                        st.text(f"• {model['name']}")
                else:
                    st.warning("No models found in Ollama")
            else:
                st.error("❌ Cannot connect to Ollama")
    
    st.divider()
    
    # Model selection
    st.subheader("Model Settings")
    
    # Simple model input (could be enhanced with dropdown from API)
    model_name = st.text_input("Model Name", value="gemma3:1b", 
                              help="Enter the model name installed in Ollama")
    
    if st.button("Update Model", use_container_width=True):
        st.session_state.ollama_client.model = model_name
        st.success(f"Model updated to: {model_name}")
    
    st.divider()
    
    # Chat controls
    st.subheader("Chat Controls")
    
    if st.button("Clear Chat History", use_container_width=True, type="secondary"):
        st.session_state.messages = []
        st.rerun()
    
    # Display chat stats
    st.divider()
    st.subheader("Chat Statistics")
    st.metric("Total Messages", len(st.session_state.messages))
    
    # System prompt (optional)
    st.divider()
    st.subheader("System Prompt")
    system_prompt = st.text_area(
        "System Prompt (optional)",
        value="You are a helpful AI assistant. Answer questions clearly and concisely.",
        height=100
    )
    
    st.session_state.system_prompt = system_prompt

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Chat container
    chat_container = st.container(height=500, border=True)
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show metadata for assistant messages
                if message["role"] == "assistant" and "metadata" in message:
                    meta = message["metadata"]
                    with st.expander("Details"):
                        st.caption(f"Model: {meta.get('model', 'N/A')}")
                        st.caption(f"Duration: {meta.get('total_duration', 0) / 1e9:.2f}s")
                        st.caption(f"Tokens: {meta.get('eval_count', 'N/A')}")

with col2:
    # Quick actions panel
    st.subheader("Quick Actions")
    
    # Example prompts
    example_prompts = [
        "Explain quantum computing in simple terms",
        "Write a Python function to calculate factorial",
        "What are the benefits of renewable energy?",
        "Tell me a short story about a robot",
        "How does machine learning work?"
    ]
    
    for i, prompt in enumerate(example_prompts):
        if st.button(prompt[:40] + "..." if len(prompt) > 40 else prompt, 
                    key=f"example_{i}", use_container_width=True):
            # Directly process the example prompt
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Generate AI response
            with chat_container:
                with st.chat_message("assistant"):
                    message_placeholder = st.empty()
                    message_placeholder.markdown("Thinking...")
                    
                    # Get response from Ollama
                    client = st.session_state.ollama_client
                    result = client.generate_response(
                        prompt=prompt,
                        system_prompt=st.session_state.get("system_prompt", None)
                    )
                    
                    if result["success"]:
                        response_text = result["response"]
                        # Display streaming effect
                        full_response = ""
                        for chunk in response_text.split():
                            full_response += chunk + " "
                            time.sleep(0.05)  # Small delay for streaming effect
                            message_placeholder.markdown(full_response + "▌")
                        
                        message_placeholder.markdown(full_response)
                        
                        # Add assistant message to history with metadata
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": full_response.strip(),
                            "metadata": {
                                "model": result.get("model"),
                                "total_duration": result.get("total_duration"),
                                "eval_count": result.get("eval_count")
                            }
                        })
                    else:
                        error_msg = f"Error: {result.get('error', 'Unknown error')}"
                        message_placeholder.markdown(f"❌ {error_msg}")
                        
                        # Add error message to history
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": error_msg,
                            "metadata": {"error": True}
                        })
            
            st.rerun()
    
    st.divider()
    
    # Info panel
    st.subheader("Info")
    st.info("""
    **Instructions:**
    1. Type your message below
    2. Press Enter or click Send
    3. Wait for AI response
    4. Use sidebar for controls
    """)

# Chat input at bottom
st.divider()

# Create a form for chat input
with st.form(key="chat_form", clear_on_submit=True):
    # Create two columns for input and button
    input_col, button_col = st.columns([5, 1])
    
    with input_col:
        user_input = st.text_input(
            "Type your message here...",
            key="user_input_widget",
            label_visibility="collapsed",
            placeholder="Ask me anything... (Press Enter or click Send)"
        )
    
    with button_col:
        submit_button = st.form_submit_button(
            "Send",
            use_container_width=True,
            type="primary"
        )

# Handle user input (form submission)
if submit_button and user_input:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Display user message
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_input)
    
    # Generate AI response
    with chat_container:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            # Get response from Ollama
            client = st.session_state.ollama_client
            result = client.generate_response(
                prompt=user_input,
                system_prompt=st.session_state.get("system_prompt", None)
            )
            
            if result["success"]:
                response_text = result["response"]
                # Display streaming effect
                full_response = ""
                for chunk in response_text.split():
                    full_response += chunk + " "
                    time.sleep(0.05)  # Small delay for streaming effect
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                # Add assistant message to history with metadata
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": full_response.strip(),
                    "metadata": {
                        "model": result.get("model"),
                        "total_duration": result.get("total_duration"),
                        "eval_count": result.get("eval_count")
                    }
                })
            else:
                error_msg = f"Error: {result.get('error', 'Unknown error')}"
                message_placeholder.markdown(f"❌ {error_msg}")
                
                # Add error message to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": {"error": True}
                })
    
    # Rerun to update the UI
    st.rerun()

# Footer
st.divider()
st.caption("""
**Ollama Chat Assistant** | Built with Streamlit & Ollama | 
[Ollama Documentation](https://github.com/ollama/ollama)
""")

# Auto-check Ollama connection on first load
if not st.session_state.model_checked:
    with st.spinner("Checking Ollama connection..."):
        client = st.session_state.ollama_client
        if client.check_connection():
            st.session_state.model_checked = True
            # Don't show success message to avoid clutter
        else:
            st.error("⚠️ Cannot connect to Ollama. Make sure Ollama is running on http://localhost:11434")
            st.session_state.model_checked = True