# utils.py
import google.generativeai as genai
from PIL import Image
import io
import logging
from typing import Optional, Tuple, Union
import numpy as np
from dataclasses import dataclass
from datetime import datetime
import os

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

class GeminiError(Exception):
    """Clase personalizada para errores relacionados con Gemini"""
    pass

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