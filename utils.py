import google.generativeai as genai
from openai import OpenAI
from anthropic import Anthropic

from PIL import Image
import io
import logging
from typing import Optional, Tuple, Union, Dict, Any, List
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import os
import base64
import requests
from enum import Enum
import json

# Importaciones para las definiciones de tipos
from config import Config, ModelConfig, ModelProvider, ModelType

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ImageMetadata:
    filename: str
    format: str
    size: Tuple[int, int]
    mode: str
    file_size: int
    timestamp: datetime

class ModelError(Exception):
    """Clase personalizada para errores relacionados con los modelos de IA"""
    pass

def initialize_models(api_keys: Dict[str, str]) -> Dict[str, bool]:
    """
    Inicializa los clientes de los modelos de IA disponibles.
    
    Args:
        api_keys (Dict[str, str]): Diccionario con las claves de API
        
    Returns:
        Dict[str, bool]: Estado de inicialización de cada proveedor
    """
    status = {
        "gemini": False,
        "openai": False,
        "anthropic": False,
        "deepseek": False
    }
    
    # Inicializar Gemini
    if api_keys.get("gemini"):
        try:
            genai.configure(api_key=api_keys["gemini"])
            # Verificar la configuración intentando acceder a un modelo
            genai.GenerativeModel('gemini-pro')
            status["gemini"] = True
            logger.info("Gemini inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar Gemini: {str(e)}")
    else:
        logger.warning("No se proporcionó clave de API para Gemini")
    
    # Inicializar OpenAI
    if api_keys.get("openai"):
        try:
            client = OpenAI(api_key=api_keys["openai"])
            # Verificar el cliente con una solicitud sencilla
            client.models.list()
            status["openai"] = True
            logger.info("OpenAI inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar OpenAI: {str(e)}")
    else:
        logger.warning("No se proporcionó clave de API para OpenAI")
    
    # Inicializar Anthropic
    if api_keys.get("anthropic"):
        try:
            client = Anthropic(api_key=api_keys["anthropic"])
            # No hay forma directa de verificar la clave sin hacer una solicitud
            # Aquí solo comprobamos que se pudo crear el cliente
            status["anthropic"] = True
            logger.info("Anthropic inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar Anthropic: {str(e)}")
    else:
        logger.warning("No se proporcionó clave de API para Anthropic")
    
    # Inicializar DeepSeek (utilizando la API REST)
    if api_keys.get("deepseek"):
        try:
            # Realizar una pequeña comprobación de autenticación
            headers = {
                "Authorization": f"Bearer {api_keys['deepseek']}",
                "Content-Type": "application/json"
            }
            # No hacemos una solicitud real, solo asumimos que si tenemos la API key, está bien
            status["deepseek"] = True
            logger.info("DeepSeek inicializado correctamente")
        except Exception as e:
            logger.error(f"Error al inicializar DeepSeek: {str(e)}")
    else:
        logger.warning("No se proporcionó clave de API para DeepSeek")
    
    return status

def get_model_response(
    provider: str,
    model_name: str,
    prompt: str,
    image: Optional[Image.Image] = None,
    api_key: str = "",
    config: Optional[Config] = None,
    retry_count: int = 2
) -> Union[str, Dict[str, str]]:
    """
    Obtiene respuesta del modelo seleccionado unificando la interfaz para todos los proveedores.
    
    Args:
        provider: Nombre del proveedor del modelo
        model_name: Nombre del modelo específico
        prompt: Texto del prompt
        image: Imagen opcional para análisis
        api_key: Clave de API para el proveedor
        config: Configuración global
        retry_count: Número de reintentos en caso de error
        
    Returns:
        Union[str, Dict[str, str]]: Respuesta del modelo o diccionario con error
    """
    if not api_key:
        return {"error": f"No se proporcionó clave de API para {provider}"}
    
    if not config:
        return {"error": "Configuración no proporcionada"}
    
    # Obtener la configuración del modelo
    model_config = None
    for model in config.models.get(provider, []):
        if model.name == model_name:
            model_config = model
            break
    
    if not model_config:
        return {"error": f"Modelo {model_name} no encontrado para el proveedor {provider}"}
    
    # Comprobar soporte para imágenes
    if image and not model_config.supports_vision:
        return {"error": f"El modelo {model_name} no soporta procesamiento de imágenes"}
    
    # Distribuir la petición según el proveedor
    for attempt in range(retry_count + 1):
        try:
            if provider == "gemini":
                return _get_gemini_response(model_name, prompt, image, model_config)
            elif provider == "openai":
                return _get_openai_response(api_key, model_name, prompt, image, model_config)
            elif provider == "anthropic":
                return _get_anthropic_response(api_key, model_name, prompt, image, model_config)
            elif provider == "deepseek":
                return _get_deepseek_response(api_key, model_name, prompt, model_config)
            else:
                return {"error": f"Proveedor {provider} no soportado"}
                
        except Exception as e:
            logger.error(f"Intento {attempt + 1} fallido para {provider}/{model_name}: {str(e)}")
            if attempt == retry_count:
                return {"error": f"Error después de {retry_count} intentos: {str(e)}"}
            continue

def _get_gemini_response(
    model_name: str,
    prompt: str,
    image: Optional[Image.Image],
    model_config: ModelConfig
) -> str:
    """Obtiene respuesta específica de Gemini"""
    model = genai.GenerativeModel(
        model_name=model_name,
        generation_config={
            "temperature": model_config.temperature,
            "top_p": model_config.top_p,
            "top_k": model_config.top_k,
            "max_output_tokens": model_config.max_tokens,
            "stop_sequences": model_config.stop_sequences
        }
    )
    
    # Crear la petición según si hay imagen o no
    if image:
        response = model.generate_content([prompt, image])
    else:
        response = model.generate_content(prompt)
    
    return response.text

def _get_openai_response(
    api_key: str,
    model_name: str,
    prompt: str,
    image: Optional[Image.Image],
    model_config: ModelConfig
) -> str:
    """Obtiene respuesta específica de OpenAI"""
    client = OpenAI(api_key=api_key)
    
    if image:
        # Convertir imagen a base64
        buffer = io.BytesIO()
        image.save(buffer, format=image.format if image.format else "PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Crear mensaje con contenido multimedia
        messages = [
            {"role": "system", "content": "Eres un asistente útil que puede analizar texto e imágenes."},
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/{image.format.lower() if image.format else 'png'};base64,{image_base64}"}}
            ]}
        ]
    else:
        # Mensaje solo con texto
        messages = [
            {"role": "system", "content": "Eres un asistente útil y amigable."},
            {"role": "user", "content": prompt}
        ]
    
    # Realizar la petición
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=model_config.temperature,
        max_tokens=model_config.max_tokens,
        top_p=model_config.top_p
    )
    
    return response.choices[0].message.content

def _get_anthropic_response(
    api_key: str,
    model_name: str,
    prompt: str,
    image: Optional[Image.Image],
    model_config: ModelConfig
) -> str:
    """Obtiene respuesta específica de Anthropic Claude"""
    client = Anthropic(api_key=api_key)
    
    if image:
        # Convertir imagen a base64
        buffer = io.BytesIO()
        image.save(buffer, format=image.format if image.format else "PNG")
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Crear mensaje con imagen
        message = client.messages.create(
            model=model_name,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            system="Eres un asistente útil que puede analizar texto e imágenes.",
            messages=[
                {
                    "role": "user", 
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": f"image/{image.format.lower() if image.format else 'png'}",
                            "data": image_base64
                        }}
                    ]
                }
            ]
        )
    else:
        # Mensaje solo con texto
        message = client.messages.create(
            model=model_name,
            max_tokens=model_config.max_tokens,
            temperature=model_config.temperature,
            system="Eres un asistente útil y amigable.",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
    
    return message.content[0].text

def _get_deepseek_response(
    api_key: str,
    model_name: str,
    prompt: str,
    model_config: ModelConfig
) -> str:
    """Obtiene respuesta específica de DeepSeek (solo texto)"""
    # URLs de API de DeepSeek (ajustar según documentación oficial)
    api_url = "https://api.deepseek.com/v1/chat/completions"
    
    # Cabeceras para la petición
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # Datos de la petición
    data = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": "Eres un asistente útil y amigable."},
            {"role": "user", "content": prompt}
        ],
        "temperature": model_config.temperature,
        "max_tokens": model_config.max_tokens,
    }
    
    # Realizar la petición
    response = requests.post(api_url, headers=headers, json=data)
    
    # Verificar respuesta
    if response.status_code == 200:
        response_data = response.json()
        if "choices" in response_data and len(response_data["choices"]) > 0:
            return response_data["choices"][0]["message"]["content"]
        else:
            return {"error": "Formato de respuesta inesperado"}
    else:
        return {"error": f"Error en la API de DeepSeek: {response.status_code} - {response.text}"}

def initialize_gemini(api_key: str) -> bool:
    """
    Inicializa la configuración de Gemini con manejo de errores.
    
    Args:
        api_key (str): API key de Google
        
    Returns:
        bool: True si la inicialización fue exitosa
        
    Raises:
        GeminiError: Si hay un error en la inicialización
    """
    try:
        genai.configure(api_key=api_key)
        # Verificar la configuración intentando acceder a un modelo
        genai.GenerativeModel('gemini-pro')
        logger.info("Gemini inicializado correctamente")
        return True
    except Exception as e:
        logger.error(f"Error al inicializar Gemini: {str(e)}")
        raise GeminiError(f"Error de inicialización: {str(e)}")

class GeminiError(Exception):
    """Clase personalizada para errores relacionados con Gemini"""
    pass

def get_gemini_response(
    model: genai.GenerativeModel,
    prompt: str,
    image: Optional[Image.Image] = None,
    retry_count: int = 2,
    safety_check: bool = True
) -> Union[str, dict]:
    """
    Obtiene respuesta de Gemini con reintentos y manejo de errores mejorado.
    
    Args:
        model: Modelo de Gemini
        prompt: Texto del prompt
        image: Imagen opcional para análisis
        retry_count: Número de reintentos en caso de error
        safety_check: Si se debe realizar verificación de seguridad
        
    Returns:
        Union[str, dict]: Respuesta del modelo o diccionario con error
    """
    def safety_check_content(text: str) -> bool:
        # Implementar verificaciones básicas de seguridad
        unsafe_patterns = ['<script', 'javascript:', 'data:']
        return not any(pattern in text.lower() for pattern in unsafe_patterns)

    for attempt in range(retry_count + 1):
        try:
            if safety_check and not safety_check_content(prompt):
                return {"error": "Contenido potencialmente inseguro detectado"}
            
            if image:
                if not isinstance(image, Image.Image):
                    return {"error": "Formato de imagen inválido"}
                response = model.generate_content([prompt, image])
            else:
                response = model.generate_content(prompt)
            
            if response.text and safety_check and not safety_check_content(response.text):
                return {"error": "Respuesta potencialmente insegura detectada"}
                
            return response.text
            
        except Exception as e:
            logger.error(f"Intento {attempt + 1} fallido: {str(e)}")
            if attempt == retry_count:
                return {"error": f"Error después de {retry_count} intentos: {str(e)}"}
            continue

def process_uploaded_image(
    uploaded_file,
    max_size: int = 5 * 1024 * 1024,  # 5MB
    allowed_formats: set = {'JPEG', 'PNG', 'WEBP'},
    max_dimensions: Tuple[int, int] = (4096, 4096)
) -> Union[Tuple[Image.Image, ImageMetadata], dict]:
    """
    Procesa y valida una imagen subida con comprobaciones de seguridad.
    
    Args:
        uploaded_file: Archivo subido
        max_size: Tamaño máximo permitido en bytes
        allowed_formats: Formatos de imagen permitidos
        max_dimensions: Dimensiones máximas permitidas
        
    Returns:
        Union[Tuple[Image.Image, ImageMetadata], dict]: Imagen procesada y metadata o error
    """
    try:
        if uploaded_file is None:
            return {"error": "No se proporcionó ningún archivo"}

        # Verificar tamaño del archivo
        if uploaded_file.size > max_size:
            return {"error": f"Archivo demasiado grande. Máximo permitido: {max_size/1024/1024}MB"}

        # Leer y verificar el formato
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.format not in allowed_formats:
            return {"error": f"Formato no permitido. Formatos aceptados: {allowed_formats}"}

        # Verificar dimensiones
        if image.size[0] > max_dimensions[0] or image.size[1] > max_dimensions[1]:
            return {"error": f"Dimensiones de imagen excesivas. Máximo permitido: {max_dimensions}"}

        # Limpiar metadatos potencialmente peligrosos
        if hasattr(image, 'info'):
            image.info = {}

        # Convertir a RGB si es necesario
        if image.mode not in ['RGB', 'RGBA']:
            image = image.convert('RGB')

        # Crear metadata
        metadata = ImageMetadata(
            filename=uploaded_file.name,
            format=image.format,
            size=image.size,
            mode=image.mode,
            file_size=uploaded_file.size,
            timestamp=datetime.now()
        )

        logger.info(f"Imagen procesada exitosamente: {metadata.filename}")
        return image, metadata

    except Exception as e:
        logger.error(f"Error al procesar imagen: {str(e)}")
        return {"error": f"Error al procesar la imagen: {str(e)}"}

def sanitize_output(text: str) -> str:
    """
    Sanitiza el texto de salida para prevenir XSS y otros problemas de seguridad.
    
    Args:
        text (str): Texto a sanitizar
        
    Returns:
        str: Texto sanitizado
    """
    # Implementar sanitización básica
    dangerous_patterns = {
        '<script': '&lt;script',
        'javascript:': 'javascript&#58;',
        'data:': 'data&#58;',
        'onclick': 'onclick&#58;',
        'onerror': 'onerror&#58;'
    }
    
    sanitized_text = text
    for pattern, replacement in dangerous_patterns.items():
        sanitized_text = sanitized_text.replace(pattern, replacement)
    
    return sanitized_text

def validate_prompt(prompt: str, max_length: int = 4096) -> Tuple[bool, str]:
    """
    Valida un prompt antes de enviarlo al modelo.
    
    Args:
        prompt (str): Prompt a validar
        max_length (int): Longitud máxima permitida
        
    Returns:
        Tuple[bool, str]: (es_válido, mensaje_error)
    """
    if not prompt or not prompt.strip():
        return False, "El prompt está vacío"
    
    if len(prompt) > max_length:
        return False, f"El prompt excede la longitud máxima de {max_length} caracteres"
    
    return True, ""

def update_model_config(config: Config, provider: str, model_name: str, **kwargs) -> bool:
    """
    Actualiza la configuración de un modelo específico.
    
    Args:
        config: Objeto de configuración
        provider: Nombre del proveedor
        model_name: Nombre del modelo
        **kwargs: Parámetros a actualizar (temperature, max_tokens, etc.)
        
    Returns:
        bool: True si se actualizó correctamente
    """
    try:
        model_config = None
        for model in config.models.get(provider, []):
            if model.name == model_name:
                model_config = model
                break
        
        if not model_config:
            return False
        
        # Actualizar los parámetros proporcionados
        for key, value in kwargs.items():
            if hasattr(model_config, key):
                setattr(model_config, key, value)
        
        logger.info(f"Configuración de modelo {provider}/{model_name} actualizada")
        return True
    except Exception as e:
        logger.error(f"Error al actualizar configuración: {str(e)}")
        return False

def export_conversation(conversation: Dict[str, Any], format_type: str = "json") -> Union[str, Dict[str, Any]]:
    """
    Exporta una conversación en diferentes formatos.
    
    Args:
        conversation: Datos de la conversación
        format_type: Formato deseado ("json", "text", "html")
        
    Returns:
        Union[str, Dict[str, Any]]: Conversación exportada
    """
    if format_type == "json":
        return json.dumps(conversation, indent=2)
    
    elif format_type == "text":
        text_export = f"Conversación: {conversation.get('name', 'Sin nombre')}\n"
        text_export += f"Fecha: {conversation.get('created_at', 'Desconocida')}\n"
        text_export += f"Modelo: {conversation.get('model_name', 'Desconocido')}\n\n"
        
        for msg in conversation.get("messages", []):
            role = "Usuario" if msg.get("role") == "user" else "NILES"
            text_export += f"{role}: {msg.get('content', '')}\n\n"
        
        return text_export
    
    elif format_type == "html":
        html_export = f"<h2>Conversación: {conversation.get('name', 'Sin nombre')}</h2>\n"
        html_export += f"<p>Fecha: {conversation.get('created_at', 'Desconocida')}</p>\n"
        html_export += f"<p>Modelo: {conversation.get('model_name', 'Desconocido')}</p>\n\n"
        
        for msg in conversation.get("messages", []):
            role = "Usuario" if msg.get("role") == "user" else "NILES"
            content = msg.get('content', '').replace('\n', '<br>')
            
            html_export += f"<div style='margin-bottom: 15px;'>\n"
            html_export += f"  <strong>{role}:</strong>\n"
            html_export += f"  <p>{content}</p>\n"
            
            if 'image_base64' in msg and msg['image_base64']:
                html_export += f"  <img src='data:image/png;base64,{msg['image_base64']}' style='max-width: 100%; border-radius: 5px;'>\n"
            
            html_export += f"</div>\n"
        
        return html_export
    
    else:
        return {"error": f"Formato de exportación no soportado: {format_type}"}