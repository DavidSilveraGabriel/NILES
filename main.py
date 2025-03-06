import base64
import streamlit as st
import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic
import time
import uuid
from datetime import datetime
from PIL import Image
import io
import json
import os
from typing import Optional, Dict, List, Tuple, Any

# Importamos configuraciones y utilidades
from config import Config, ModelConfig, ModelProvider, ModelType
from utils import (
    initialize_models, 
    get_model_response, 
    process_uploaded_image, 
    validate_prompt,
    sanitize_output,
    ModelError,
    logger
)

# Configuración de tema y estilos
def set_custom_theme():
    """Configura el tema personalizado en tonos marrones"""
    st.markdown("""
    <style>
    :root {
        --primary-color: #8B4513;
        --background-color: #F5F5DC;
        --secondary-background-color: #DEB887;
        --text-color: #3E2723;
        --font: 'Helvetica Neue', sans-serif;
    }
    
    .stApp {
        background-color: var(--background-color);
        color: var(--text-color);
    }
    
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        background-color: #A0522D;
        transform: translateY(-2px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 15px;
        margin-bottom: 10px;
        display: flex;
        flex-direction: column;
    }
    
    .chat-message.user {
        background-color: #D2B48C;
        border-bottom-right-radius: 0;
        align-self: flex-end;
    }
    
    .chat-message.assistant {
        background-color: #E6CCB2;
        border-bottom-left-radius: 0;
        align-self: flex-start;
    }
    
    .chat-header {
        padding: 1rem;
        background-color: var(--primary-color);
        color: white;
        border-radius: 10px 10px 0 0;
        margin-bottom: 1rem;
    }
    
    .chat-input {
        background-color: white;
        border-radius: 20px;
        padding: 0.5rem;
        display: flex;
        align-items: center;
        margin-top: 1rem;
    }
    
    .stTextInput>div>div>input {
        background-color: white;
        color: var(--text-color);
        border-radius: 20px;
        border: 1px solid #D2B48C;
        padding: 0.75rem 1rem;
    }
    
    .conversation-list {
        background-color: #E6CCB2;
        border-radius: 10px;
        padding: 0.5rem;
        margin-bottom: 1rem;
    }
    
    .conversation-item {
        padding: 0.5rem;
        border-radius: 5px;
        margin-bottom: 0.5rem;
        background-color: white;
        cursor: pointer;
        transition: all 0.2s;
    }
    
    .conversation-item:hover {
        background-color: #D2B48C;
    }
    
    .model-selector {
        background-color: white;
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 1rem;
    }
    
    .stSelectbox>div>div {
        background-color: white;
        border-radius: 20px;
        border: 1px solid #D2B48C;
    }
    
    .settings-panel {
        background-color: #F5F5DC;
        border-radius: 10px;
        padding: 1rem;
        border: 1px solid #D2B48C;
    }
    
    /* Personalización para imágenes en el chat */
    .chat-image {
        max-width: 100%;
        border-radius: 10px;
        margin-top: 0.5rem;
    }
    
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        padding: 1rem;
    }
    
    .logo {
        max-width: 150px;
        height: auto;
    }
    
    /* Spinner personalizado */
    .stSpinner > div > div > div {
        border-top-color: var(--primary-color) !important;
    }
    
    /* Ajustes para dispositivos móviles */
    @media (max-width: 768px) {
        .chat-message {
            padding: 0.75rem;
        }
        
        .stButton>button {
            padding: 0.4rem 0.8rem;
        }
    }
    </style>
    """, unsafe_allow_html=True)

def initialize_session_state():
    """Inicializa el estado de la sesión con valores predeterminados"""
    if "conversations" not in st.session_state:
        st.session_state.conversations = {}
        
    if "current_conversation_id" not in st.session_state:
        new_id = str(uuid.uuid4())
        st.session_state.current_conversation_id = new_id
        st.session_state.conversations[new_id] = {
            "name": f"Conversación {datetime.now().strftime('%d/%m/%Y %H:%M')}",
            "messages": [],
            "created_at": datetime.now().isoformat(),
            "model_provider": "gemini",
            "model_name": "gemini-1.5-flash-latest"
        }
    
    if "config" not in st.session_state:
        st.session_state.config = Config()
    
    if "error_count" not in st.session_state:
        st.session_state.error_count = 0
        
    if "api_keys" not in st.session_state:
        st.session_state.api_keys = {
            "gemini": st.session_state.config.api_keys.get("gemini", ""),
            "openai": st.session_state.config.api_keys.get("openai", ""),
            "anthropic": st.session_state.config.api_keys.get("anthropic", ""),
            "deepseek": st.session_state.config.api_keys.get("deepseek", "")
        }

def create_new_conversation(model_provider="gemini", model_name="gemini-1.5-flash-latest"):
    """Crea una nueva conversación"""
    new_id = str(uuid.uuid4())
    st.session_state.current_conversation_id = new_id
    st.session_state.conversations[new_id] = {
        "name": f"Conversación {datetime.now().strftime('%d/%m/%Y %H:%M')}",
        "messages": [],
        "created_at": datetime.now().isoformat(),
        "model_provider": model_provider,
        "model_name": model_name
    }
    return new_id

def rename_conversation(conversation_id, new_name):
    """Renombra una conversación existente"""
    if conversation_id in st.session_state.conversations:
        st.session_state.conversations[conversation_id]["name"] = new_name

def delete_conversation(conversation_id):
    """Elimina una conversación"""
    if conversation_id in st.session_state.conversations:
        del st.session_state.conversations[conversation_id]
        # Si eliminamos la conversación actual, cambiamos a otra o creamos una nueva
        if conversation_id == st.session_state.current_conversation_id:
            if st.session_state.conversations:
                st.session_state.current_conversation_id = next(iter(st.session_state.conversations))
            else:
                create_new_conversation()

def export_conversation(conversation_id):
    """Exporta una conversación como JSON"""
    if conversation_id in st.session_state.conversations:
        return json.dumps(st.session_state.conversations[conversation_id], indent=2)
    return None

def format_message(message, is_user=True):
    """Formatea un mensaje para su visualización"""
    role = "user" if is_user else "assistant"
    with st.container():
        st.markdown(f"""
        <div class="chat-message {role}">
            <strong>{'Usuario' if is_user else 'NILES'}</strong>
            <div>{message['content']}</div>
            {f'<img src="data:image/png;base64,{message["image_base64"]}" class="chat-image" />' if 'image_base64' in message and message['image_base64'] else ''}
        </div>
        """, unsafe_allow_html=True)

def handle_sidebar(config: Config):
    """Gestiona el panel lateral con opciones de configuración y conversaciones"""
    with st.sidebar:
        # Logo
        st.image("src/imgs/cover.png", width=250)
        
        # Conversaciones
        st.subheader("📂 Conversaciones")
        
        # Nueva conversación
        if st.button("📝 Nueva conversación", use_container_width=True):
            current_provider = st.session_state.conversations[st.session_state.current_conversation_id]["model_provider"]
            current_model = st.session_state.conversations[st.session_state.current_conversation_id]["model_name"]
            create_new_conversation(current_provider, current_model)
            st.experimental_rerun()
        
        # Lista de conversaciones
        for conv_id, conv_data in st.session_state.conversations.items():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                if st.button(f"{conv_data['name'][:20]}...", key=f"conv_{conv_id}", use_container_width=True):
                    st.session_state.current_conversation_id = conv_id
                    st.experimental_rerun()
            
            with col2:
                if st.button("✏️", key=f"edit_{conv_id}"):
                    st.session_state.edit_conv_id = conv_id
                    st.session_state.edit_conv_name = conv_data["name"]
            
            with col3:
                if st.button("🗑️", key=f"delete_{conv_id}"):
                    delete_conversation(conv_id)
                    st.experimental_rerun()
        
        # Editar nombre de conversación
        if "edit_conv_id" in st.session_state:
            with st.form(key="rename_form"):
                new_name = st.text_input("Nuevo nombre:", value=st.session_state.edit_conv_name)
                if st.form_submit_button("Guardar"):
                    rename_conversation(st.session_state.edit_conv_id, new_name)
                    del st.session_state.edit_conv_id
                    del st.session_state.edit_conv_name
                    st.experimental_rerun()
        
        # Exportar conversación actual
        if st.button("📥 Exportar conversación actual", use_container_width=True):
            conv_data = export_conversation(st.session_state.current_conversation_id)
            if conv_data:
                st.download_button(
                    label="Descargar JSON",
                    data=conv_data,
                    file_name=f"niles_conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        st.divider()
        
        # Configuración de modelos
        st.subheader("⚙️ Configuración")
        
        # Selección de proveedor y modelo
        current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
        
        provider_options = {
            "gemini": "Google Gemini",
            "openai": "OpenAI",
            "anthropic": "Anthropic Claude",
            "deepseek": "DeepSeek AI"
        }
        
        selected_provider = st.selectbox(
            "Proveedor de modelo:",
            options=list(provider_options.keys()),
            format_func=lambda x: provider_options[x],
            index=list(provider_options.keys()).index(current_conv["model_provider"])
        )
        
        # Obtener modelos disponibles para el proveedor seleccionado
        model_options = config.get_models_by_provider(selected_provider)
        model_names = [model.name for model in model_options]
        
        try:
            current_model_index = model_names.index(current_conv["model_name"])
        except ValueError:
            current_model_index = 0
            
        selected_model = st.selectbox(
            "Modelo:",
            options=model_names,
            index=current_model_index
        )
        
        # Aplicar cambios de modelo
        if st.button("Aplicar cambios", use_container_width=True):
            current_conv["model_provider"] = selected_provider
            current_conv["model_name"] = selected_model
            st.success("Configuración actualizada")
            time.sleep(1)
            st.experimental_rerun()
        
        # Panel de configuración avanzada
        with st.expander("⚙️ Configuración avanzada"):
            # Parámetros del modelo
            model_config = config.get_model_config(selected_provider, selected_model)
            
            if model_config:
                st.subheader("Parámetros del modelo")
                
                new_temp = st.slider(
                    "Temperatura",
                    0.0, 1.0,
                    model_config.temperature,
                    help="Controla la creatividad de las respuestas"
                )
                
                new_tokens = st.slider(
                    "Máx. Tokens",
                    100, 4096,
                    model_config.max_tokens,
                    help="Límite de longitud de respuesta"
                )
                
                # Actualizar configuración
                if st.button("Guardar parámetros", use_container_width=True):
                    config.update_model_params(
                        selected_provider, 
                        selected_model, 
                        temperature=new_temp, 
                        max_tokens=new_tokens
                    )
                    st.success("Parámetros actualizados")
                    time.sleep(1)
            
            # Configuración de APIs
            st.subheader("Claves de API")
            
            gemini_api = st.text_input(
                "API Key de Google Gemini:",
                value=st.session_state.api_keys["gemini"],
                type="password"
            )
            
            openai_api = st.text_input(
                "API Key de OpenAI:",
                value=st.session_state.api_keys["openai"],
                type="password"
            )
            
            anthropic_api = st.text_input(
                "API Key de Anthropic:",
                value=st.session_state.api_keys["anthropic"],
                type="password"
            )
            
            deepseek_api = st.text_input(
                "API Key de DeepSeek:",
                value=st.session_state.api_keys["deepseek"],
                type="password"
            )
            
            if st.button("Guardar claves de API", use_container_width=True):
                st.session_state.api_keys = {
                    "gemini": gemini_api,
                    "openai": openai_api,
                    "anthropic": anthropic_api,
                    "deepseek": deepseek_api
                }
                
                # Actualizar en la configuración
                config.update_api_keys(st.session_state.api_keys)
                
                st.success("Claves de API actualizadas")
                time.sleep(1)
                st.experimental_rerun()
        
        # Limpiar errores
        if st.session_state.error_count > 0:
            st.warning(f"Errores en la sesión: {st.session_state.error_count}")
            if st.button("Limpiar errores", use_container_width=True):
                st.session_state.error_count = 0
                st.experimental_rerun()
        
        # Información del sistema
        st.divider()
        st.caption("Información del sistema")
        current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
        st.info(f"Modelo actual: {current_conv['model_name']} ({provider_options[current_conv['model_provider']]})")

def handle_chat_input(model, prompt: str, image=None) -> Optional[str]:
    """Procesa la entrada del chat y genera una respuesta"""
    is_valid, error_message = validate_prompt(prompt)
    if not is_valid:
        st.error(error_message)
        return None

    try:
        current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
        provider = current_conv["model_provider"]
        model_name = current_conv["model_name"]
        
        with st.spinner("Generando respuesta..."):
            response = get_model_response(
                provider=provider,
                model_name=model_name,
                prompt=prompt,
                image=image,
                api_key=st.session_state.api_keys[provider],
                config=st.session_state.config
            )
            
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

def main():
    try:
        # Inicialización
        initialize_session_state()
        config = st.session_state.config
        
        # Configuración de página
        st.set_page_config(
            page_title="NILES - Chatbot Multimodal",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/DavidSilveraGabriel/NILES',
                'Report a bug': 'https://github.com/DavidSilveraGabriel/NILES/issues',
                'About': "NILES - Un chatbot multimodal con múltiples proveedores"
            }
        )
        
        # Aplicar tema personalizado
        set_custom_theme()
        
        # Inicializar modelos
        initialize_models(st.session_state.api_keys)
        
        # Interfaz principal
        handle_sidebar(config)
        
        # Área principal de chat
        col1, col2 = st.columns([5, 1])
        with col1:
            current_conv = st.session_state.conversations[st.session_state.current_conversation_id]
            st.subheader(f"💬 {current_conv['name']}")
        with col2:
            provider_icon = {
                "gemini": "🌐",
                "openai": "🧠",
                "anthropic": "🔮",
                "deepseek": "🔍"
            }.get(current_conv["model_provider"], "🤖")
            st.caption(f"{provider_icon} {current_conv['model_name']}")
        
        # Contenedor de mensajes
        chat_container = st.container()
        
        # Área de entrada
        input_container = st.container()
        
        # Mostrar mensajes
        with chat_container:
            for message in current_conv["messages"]:
                format_message(message, message["role"] == "user")
        
        # Área de entrada
        with input_container:
            col1, col2 = st.columns([6, 1])
            
            # Upload de imagen
            uploaded_file = st.file_uploader(
                "Subir imagen (opcional)",
                type=config.app.supported_image_types,
                key="chat_file_uploader",
                label_visibility="collapsed"
            )
            
            # Vista previa de imagen
            image_data = None
            image_base64 = None
            
            if uploaded_file:
                result = process_uploaded_image(uploaded_file)
                
                if isinstance(result, dict) and "error" in result:
                    st.error(result["error"])
                else:
                    image, metadata = result
                    st.image(
                        image,
                        caption=f"Imagen: {metadata.filename}",
                        use_container_width=True
                    )
                    image_data = image
                    
                    # Convertir imagen a base64 para guardarla en el historial
                    buffered = io.BytesIO()
                    image.save(buffered, format=metadata.format)
                    image_base64 = base64.b64encode(
                        buffered.getvalue()
                    ).decode()
            
            # Campo de entrada de texto
            with st.form(key="chat_form", clear_on_submit=True):
                prompt = st.text_input(
                    "Escribe tu mensaje:",
                    key="prompt_input",
                    placeholder="Escribe aquí tu mensaje...",
                    label_visibility="collapsed"
                )
                
                col1, col2 = st.columns([1, 5])
                with col1:
                    submit = st.form_submit_button("Enviar", use_container_width=True)
                with col2:
                    # Espacio en blanco para equilibrar el diseño
                    st.write("")
            
            if submit and (prompt or uploaded_file):
                # Añadir mensaje del usuario al historial
                user_message = {
                    "role": "user",
                    "content": prompt,
                    "timestamp": datetime.now().isoformat()
                }
                
                if image_base64:
                    user_message["image_base64"] = image_base64
                
                current_conv["messages"].append(user_message)
                
                # Mostrar mensaje del usuario
                format_message(user_message, is_user=True)
                
                # Generar respuesta
                response = handle_chat_input(None, prompt, image_data)
                
                if response:
                    # Añadir respuesta al historial
                    assistant_message = {
                        "role": "assistant",
                        "content": response,
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    current_conv["messages"].append(assistant_message)
                    
                    # Mostrar respuesta
                    format_message(assistant_message, is_user=False)

    except Exception as e:
        logger.error(f"Error en la aplicación: {str(e)}")
        st.error(f"Ha ocurrido un error inesperado: {str(e)}")
        if st.button("🔄 Reiniciar aplicación"):
            st.experimental_rerun()

if __name__ == "__main__":
    main()