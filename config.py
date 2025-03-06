import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum
from dotenv import load_dotenv

class ModelType(Enum):
    TEXT = "text"
    VISION = "vision"
    CHAT = "chat"
    MULTI = "multimodal"

class ModelProvider(Enum):
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    DEEPSEEK = "deepseek"

@dataclass
class ModelConfig:
    """
    Configuración para un modelo específico de lenguaje.
    """
    name: str
    provider: ModelProvider
    type: ModelType
    max_tokens: int = 500
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 40
    stop_sequences: List[str] = field(default_factory=list)
    supports_vision: bool = False
    friendly_name: str = ""

    def __post_init__(self):
        """
        Inicialización posterior para establecer el nombre amigable si no se proporciona.
        """
        if not self.friendly_name:
            self.friendly_name = self.name.replace("-", " ").title()

@dataclass
class AppConfig:
    """
    Configuración general de la aplicación.
    """
    title: str = "NILES - Chatbot Multimodal"
    description: str = "Un chatbot inteligente capaz de procesar texto e imágenes con múltiples proveedores"
    theme_color: str = "#8B4513"
    layout: str = "wide"
    initial_sidebar_state: str = "expanded"
    supported_image_types: List[str] = field(default_factory=lambda: ["jpg", "jpeg", "png", "webp"])
    max_image_size_mb: int = 5
    version: str = "2.0.0"

class Config:
    """
    Clase principal de configuración que carga variables de entorno y define modelos disponibles.
    """
    def __init__(self):
        """
        Inicializa la configuración cargando variables de entorno y definiendo modelos.
        """
        load_dotenv()
        # Claves de API
        self.api_keys = {
            "gemini": os.getenv("GOOGLE_API_KEY", ""),
            "openai": os.getenv("OPENAI_API_KEY", ""),
            "anthropic": os.getenv("ANTHROPIC_API_KEY", ""),
            "deepseek": os.getenv("DEEPSEEK_API_KEY", "")
        }
        # Configuraciones de modelos
        self.models = {
            # Modelos de Google Gemini
            "gemini": [
                ModelConfig(
                    name="gemini-1.5-flash-latest",
                    provider=ModelProvider.GEMINI,
                    type=ModelType.MULTI,
                    max_tokens=500,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    stop_sequences=["Usuario:", "Asistente:"],
                    supports_vision=True,
                    friendly_name="Gemini 1.5 Flash"
                ),
                ModelConfig(
                    name="gemini-1.5-pro-latest",
                    provider=ModelProvider.GEMINI,
                    type=ModelType.MULTI,
                    max_tokens=1000,
                    temperature=0.8,
                    top_p=0.97,
                    top_k=40,
                    stop_sequences=["Usuario:", "Asistente:"],
                    supports_vision=True,
                    friendly_name="Gemini 1.5 Pro"
                ),
                ModelConfig(
                    name="gemini-pro",
                    provider=ModelProvider.GEMINI,
                    type=ModelType.TEXT,
                    max_tokens=500,
                    temperature=0.7,
                    top_p=0.95,
                    top_k=40,
                    stop_sequences=["Usuario:", "Asistente:"],
                    supports_vision=False,
                    friendly_name="Gemini Pro (Solo texto)"
                )
            ],
            # Modelos de OpenAI
            "openai": [
                ModelConfig(
                    name="gpt-4o",
                    provider=ModelProvider.OPENAI,
                    type=ModelType.MULTI,
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.95,
                    supports_vision=True,
                    friendly_name="GPT-4o"
                ),
                ModelConfig(
                    name="gpt-4-turbo",
                    provider=ModelProvider.OPENAI,
                    type=ModelType.MULTI,
                    max_tokens=1000,
                    temperature=0.7,
                    top_p=0.95,
                    supports_vision=True,
                    friendly_name="GPT-4 Turbo"
                ),
                ModelConfig(
                    name="gpt-3.5-turbo",
                    provider=ModelProvider.OPENAI,
                    type=ModelType.TEXT,
                    max_tokens=500,
                    temperature=0.7,
                    top_p=0.95,
                    supports_vision=False,
                    friendly_name="GPT-3.5 Turbo (Solo texto)"
                )
            ],
            # Modelos de Anthropic
            "anthropic": [
                ModelConfig(
                    name="claude-3-5-sonnet",
                    provider=ModelProvider.ANTHROPIC,
                    type=ModelType.MULTI,
                    max_tokens=800,
                    temperature=0.7,
                    top_p=0.95,
                    supports_vision=True,
                    friendly_name="Claude 3.5 Sonnet"
                ),
                ModelConfig(
                    name="claude-3-opus",
                    provider=ModelProvider.ANTHROPIC,
                    type=ModelType.MULTI,
                    max_tokens=1000,
                    temperature=0.8,
                    top_p=0.97,
                    supports_vision=True,
                    friendly_name="Claude 3 Opus" # Corrección del nombre amigable
                )
            ],
             # Modelos de DeepSeek
            "deepseek": [
                ModelConfig(
                    name="deepseek-chat", # Nombre más general para el chat
                    provider=ModelProvider.DEEPSEEK,
                    type=ModelType.TEXT, # Asumiendo que es principalmente texto
                    max_tokens=500,
                    temperature=0.7,
                    top_p=0.95,
                    friendly_name="DeepSeek Chat" # Nombre amigable
                )
            ]
        }
        self.app_config = AppConfig() # Inicializa la configuración de la aplicación

    def get_api_key(self, provider: ModelProvider) -> str:
        """
        Retorna la clave de API para el proveedor especificado.

        Args:
            provider (ModelProvider): El proveedor de la API.

        Returns:
            str: La clave de API.

        Raises:
            ValueError: Si no se encuentra la clave de API para el proveedor o está vacía.
        """
        key = self.api_keys.get(provider.value)
        if not key:
            raise ValueError(f"Clave de API no encontrada o vacía para el proveedor: {provider.value}. Asegúrate de que la variable de entorno esté configurada.")
        return key

    def get_models_for_provider(self, provider: ModelProvider) -> List[ModelConfig]:
        """
        Retorna la lista de configuraciones de modelos para un proveedor específico.

        Args:
            provider (ModelProvider): El proveedor de modelos.

        Returns:
            List[ModelConfig]: Lista de objetos ModelConfig para el proveedor.
        """
        return self.models.get(provider.value, []) # Retorna una lista vacía si el proveedor no se encuentra

    def get_model_config(self, provider: ModelProvider, model_name: str) -> Optional[ModelConfig]:
        """
        Retorna la configuración de un modelo específico por proveedor y nombre del modelo.

        Args:
            provider (ModelProvider): El proveedor del modelo.
            model_name (str): El nombre del modelo.

        Returns:
            Optional[ModelConfig]: El objeto ModelConfig si se encuentra, None si no.
        """
        models_for_provider = self.get_models_for_provider(provider)
        for model_config in models_for_provider:
            if model_config.name == model_name:
                return model_config
        return None

    def get_all_models(self) -> Dict[str, List[ModelConfig]]:
        """
        Retorna un diccionario con todas las configuraciones de modelos, agrupadas por proveedor.

        Returns:
            Dict[str, List[ModelConfig]]: Diccionario de modelos.
        """
        return self.models

    def get_app_config(self) -> AppConfig:
        """
        Retorna la configuración general de la aplicación.

        Returns:
            AppConfig: Objeto AppConfig.
        """
        return self.app_config


# Ejemplo de uso:
if __name__ == "__main__":
    config = Config()

    # Obtener configuración de la aplicación
    app_config = config.get_app_config()
    print("Configuración de la Aplicación:")
    print(f"  Título: {app_config.title}")
    print(f"  Versión: {app_config.version}")

    # Obtener clave de API para Gemini
    try:
        gemini_api_key = config.get_api_key(ModelProvider.GEMINI)
        print(f"\nClave de API de Gemini: {gemini_api_key[:10]}... (mostrando solo los primeros 10 caracteres)") # Mostrar solo los primeros caracteres por seguridad
    except ValueError as e:
        print(f"\nError al obtener la clave de API de Gemini: {e}")

    # Obtener modelos de OpenAI
    openai_models = config.get_models_for_provider(ModelProvider.OPENAI)
    print("\nModelos de OpenAI disponibles:")
    for model in openai_models:
        print(f"  - {model.friendly_name} ({model.name}), Tipo: {model.type.value}, Soporta Vision: {model.supports_vision}")

    # Obtener configuración específica de un modelo
    gpt_4o_config = config.get_model_config(ModelProvider.OPENAI, "gpt-4o")
    if gpt_4o_config:
        print(f"\nConfiguración de GPT-4o:")
        print(f"  Nombre Amigable: {gpt_4o_config.friendly_name}")
        print(f"  Max Tokens: {gpt_4o_config.max_tokens}")
        print(f"  Temperatura: {gpt_4o_config.temperature}")
    else:
        print("\nConfiguración de GPT-4o no encontrada.")

    # Obtener todos los modelos
    all_models = config.get_all_models()
    print("\nTodos los modelos configurados:")
    for provider_name, models in all_models.items():
        print(f"\n  Proveedor: {provider_name.upper()}")
        for model in models:
            print(f"    - {model.friendly_name} ({model.name})")