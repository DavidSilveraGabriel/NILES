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
    """Inicializa las variables de estado de la sesión"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "config" not in st.session_state:
        st.session_state.config = Config()
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0

def create_sidebar(config: Config):
    """Crea y maneja la barra lateral de configuración"""
    with st.sidebar:
        st.title("⚙️ Configuración")
        
        # Configuración del modelo de texto
        st.subheader("Modelo de Texto")
        text_config = config.get_model_config("text")
        
        col1, col2 = st.columns(2)
        with col1:
            new_temp = st.slider(
                "Temperatura",
                0.0, 1.0,
                text_config.temperature,
                help="Controla la creatividad de las respuestas"
            )
        with col2:
            new_tokens = st.slider(
                "Máx. Tokens",
                100, 4096,
                text_config.max_tokens,
                help="Límite de longitud de respuesta"
            )

        # Configuración del modelo de visión
        st.subheader("Modelo de Visión")
        vision_config = config.get_model_config("vision")
        top_p = st.slider(
            "Top P",
            0.0, 1.0,
            vision_config.top_p,
            help="Controla la diversidad de las respuestas"
        )
        
        # Actualizar configuraciones
        config.update_model_params("text", temperature=new_temp, max_tokens=new_tokens)
        config.update_model_params("vision", top_p=top_p)
        
        # Utilidades
        st.subheader("Utilidades")
        if st.button("🗑️ Limpiar Chat", use_container_width=True):
            st.session_state.messages = []
            st.experimental_rerun()
            
        if st.button("🔄 Reiniciar Configuración", use_container_width=True):
            st.session_state.config = Config()
            st.experimental_rerun()
        
        # Información del sistema
        st.divider()
        st.caption("Información del Sistema")
        st.info(f"Modelo Texto: {text_config.name}\nModelo Visión: {vision_config.name}")
        
        if st.session_state.error_count > 0:
            st.warning(f"Errores en la sesión: {st.session_state.error_count}")

def handle_chat_input(model: genai.GenerativeModel, prompt: str) -> Optional[str]:
    """Maneja la entrada del chat y genera la respuesta"""
    is_valid, error_message = validate_prompt(prompt)
    if not is_valid:
        st.error(error_message)
        return None

    try:
        with st.spinner("Generando respuesta..."):
            response = get_gemini_response(model, prompt)
            if isinstance(response, dict) and "error" in response:
                st.error(response["error"])
                st.session_state.error_count += 1
                return None
            return sanitize_output(response)
    except Exception as e:
        logger.error(f"Error al generar respuesta: {str(e)}")
        st.error(f"Error al generar respuesta: {str(e)}")
        st.session_state.error_count += 1
        return None

def handle_image_analysis(
    model: genai.GenerativeModel,
    image_prompt: str,
    image,
    metadata
) -> Optional[str]:
    """Maneja el análisis de imágenes"""
    try:
        with st.spinner("Analizando imagen..."):
            response = get_gemini_response(model, image_prompt, image)
            if isinstance(response, dict) and "error" in response:
                st.error(response["error"])
                st.session_state.error_count += 1
                return None
            return sanitize_output(response)
    except Exception as e:
        logger.error(f"Error en análisis de imagen: {str(e)}")
        st.error(f"Error al analizar la imagen: {str(e)}")
        st.session_state.error_count += 1
        return None

def render_chat_messages():
    """Renderiza los mensajes del chat"""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def main():
    try:
        # Inicialización
        initialize_session_state()
        config = st.session_state.config
        
        # Configuración de la página
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

        # Inicialización de Gemini
        if not initialize_gemini(config.api_key):
            st.error("Error al inicializar Gemini. Por favor, verifica tu API key.")
            return

        # Configuración de modelos
        model = genai.GenerativeModel(config.get_model_config("text").name)
        model_vision = genai.GenerativeModel(config.get_model_config("vision").name)

        # Interfaz de usuario
        st.title(config.app.title)
        st.caption(config.app.description)
        
        # Sidebar
        create_sidebar(config)

        # Pestañas principales
        tab1, tab2 = st.tabs(["💭 Chat de Texto", "🖼️ Chat con Imágenes"])

        with tab1:
            render_chat_messages()

            if prompt := st.chat_input("Escribe tu mensaje aquí...", key="text_input"):
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
            st.write("📤 Sube una imagen y hazme preguntas sobre ella")
            
            uploaded_file = st.file_uploader(
                "Elige una imagen...",
                type=config.app.supported_image_types,
                help=f"Formatos soportados: {', '.join(config.app.supported_image_types)}"
            )
            
            if uploaded_file:
                result = process_uploaded_image(uploaded_file)
                
                if isinstance(result, dict) and "error" in result:
                    st.error(result["error"])
                else:
                    image, metadata = result
                    st.image(
                        image,
                        caption=f"Imagen: {metadata.filename} ({metadata.size[0]}x{metadata.size[1]})",
                        use_container_width=True
                    )
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        image_prompt = st.text_input(
                            "Hazme una pregunta sobre la imagen:",
                            key="image_prompt"
                        )
                    with col2:
                        st.markdown("<br>", unsafe_allow_html=True)
                        analyze_button = st.button("🔍 Analizar", use_container_width=True)
                    
                    if image_prompt and analyze_button:
                        response = handle_image_analysis(
                            model_vision,
                            image_prompt,
                            image,
                            metadata
                        )
                        if response:
                            st.markdown("**Análisis:**")
                            st.markdown(response)
                            
                    # Mostrar metadata
                    with st.expander("📋 Detalles de la imagen"):
                        st.json({
                            "filename": metadata.filename,
                            "format": metadata.format,
                            "size": f"{metadata.size[0]}x{metadata.size[1]}",
                            "mode": metadata.mode,
                            "file_size": f"{metadata.file_size/1024:.1f} KB",
                            "timestamp": metadata.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        })

    except Exception as e:
        logger.error(f"Error en la aplicación: {str(e)}")
        st.error(f"Ha ocurrido un error inesperado: {str(e)}")
        if st.button("🔄 Reiniciar Aplicación"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()