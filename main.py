import streamlit as st
import google.generativeai as genai
from config import Config
from utils import (
    initialize_gemini, 
    get_gemini_response, 
    process_uploaded_image, 
    validate_prompt,
    sanitize_output,
    GeminiError,
    logger
)
import time
from typing import Optional

def initialize_session_state():
    """Initializes the session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = Config()
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0

def create_sidebar(config: Config):
    """Creates and manages the configuration sidebar"""
    with st.sidebar:
        st.title("⚙️ Configuration")
        
        # Text model configuration
        st.subheader("Text Model")
        text_config = config.get_model_config("text")
        
        col1, col2 = st.columns(2)
        with col1:
            new_temp = st.slider(
                "Temperature",
                0.0, 1.0,
                text_config.temperature,
                help="Controls the creativity of responses"
            )
        with col2:
            new_tokens = st.slider(
                "Max Tokens",
                100, 500,
                text_config.max_tokens,
                help="Response length limit"
            )

        # Vision model configuration
        st.subheader("Vision Model")
        vision_config = config.get_model_config("vision")
        top_p = st.slider(
            "Top P",
            0.0, 1.0,
            vision_config.top_p,
            help="Controls the diversity of responses"
        )
        
        # Update configurations
        config.update_model_params("text", temperature=new_temp, max_tokens=new_tokens)
        config.update_model_params("vision", top_p=top_p)
        
        # Utilities
        st.subheader("Utilities")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()
            
        if st.button("🔄 Reset Configuration", use_container_width=True):
            st.session_state.config = Config()
            st.experimental_rerun()
        
        # System information
        st.divider()
        st.caption("System Information")
        st.info(f"Text Model: {text_config.name}\nVision Model: {vision_config.name}")
        
        if st.session_state.error_count > 0:
            st.warning(f"Session errors: {st.session_state.error_count}")

def handle_chat_input(model: genai.GenerativeModel, prompt: str) -> Optional[str]:
    """Handles chat input and generates the response"""
    is_valid, error_message = validate_prompt(prompt)
    if not is_valid:
        st.error(error_message)
        return None

    try:
        with st.spinner("Generating response..."):
            response = get_gemini_response(model, prompt)
            if isinstance(response, dict) and "error" in response:
                st.error(response["error"])
                st.session_state.error_count += 1
                return None
            return sanitize_output(response)
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        st.error(f"Error generating response: {str(e)}")
        st.session_state.error_count += 1
        return None

def handle_image_analysis(
    model: genai.GenerativeModel,
    image_prompt: str,
    image,
    metadata
) -> Optional[str]:
    """Handles image analysis"""
    try:
        with st.spinner("Analyzing image..."):
            response = get_gemini_response(model, image_prompt, image)
            if isinstance(response, dict) and "error" in response:
                st.error(response["error"])
                st.session_state.error_count += 1
                return None
            return sanitize_output(response)
    except Exception as e:
        logger.error(f"Error in image analysis: {str(e)}")
        st.error(f"Error analyzing image: {str(e)}")
        st.session_state.error_count += 1
        return None

def render_chat_messages():
    """Renders chat messages"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    try:
        # Initialization
        initialize_session_state()
        config = st.session_state.config
        
        # Page configuration
        st.set_page_config(
            page_title=config.app.title,
            layout=config.app.layout,
            initial_sidebar_state=config.app.initial_sidebar_state,
            menu_items={
                'Get Help': 'https://github.com/tuuser/tuchatbot',
                'Report a bug': 'https://github.com/tuuser/tuchatbot/issues',
                'About': config.app.description
            }
        )

        # Gemini initialization
        if not initialize_gemini(config.api_key):
            st.error("Error initializing Gemini. Please verify your API key.")
            return

        # Model configuration
        model = genai.GenerativeModel(config.get_model_config("text").name)
        model_vision = genai.GenerativeModel(config.get_model_config("vision").name)

        # User interface
        st.title(config.app.title)
        st.caption(config.app.description)
        
        # Sidebar
        create_sidebar(config)

        # Main tabs
        tab1, tab2 = st.tabs(["💭 Text Chat", "🖼️ Image Chat"])

        with tab1:
            render_chat_messages()

            if prompt := st.chat_input("Type your message here...", key="text_input"):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    response = handle_chat_input(model, prompt)
                    if response:
                        st.markdown(response)
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response
                        })

        with tab2:
            st.write("📤 Upload an image and ask me questions about it")
            
            uploaded_file = st.file_uploader(
                "Choose an image...",
                type=config.app.supported_image_types,
                help=f"Supported formats: {', '.join(config.app.supported_image_types)}"
            )
            
            if uploaded_file:
                result = process_uploaded_image(uploaded_file)
                
                if isinstance(result, dict) and "error" in result:
                    st.error(result["error"])
                else:
                    image, metadata = result
                    st.image(
                        image,
                        caption=f"Image: {metadata.filename} ({metadata.size[0]}x{metadata.size[1]})",
                        use_container_width=True
                    )
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        image_prompt = st.text_input(
                            "Ask me a question about the image:",
                            key="image_prompt"
                        )
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        analyze_button = st.button("🔍 Analyze", use_container_width=True)
                    
                    if image_prompt and analyze_button:
                        response = handle_image_analysis(
                            model_vision,
                            image_prompt,
                            image,
                            metadata
                        )
                        if response:
                            st.markdown("**Analysis:**")
                            st.markdown(response)
                            
                    # Display metadata
                    with st.expander("📋 Image Details"):
                        st.json({
                            "filename": metadata.filename,
                            "format": metadata.format,
                            "size": f"{metadata.size[0]}x{metadata.size[1]}",
                            "mode": metadata.mode,
                            "file_size": f"{metadata.file_size/1024:.1f} KB",
                            "timestamp": metadata.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        })

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error(f"An unexpected error has occurred: {str(e)}")
        if st.button("🔄 Restart Application"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()