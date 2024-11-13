import os
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
from dotenv import load_dotenv

class ModelType(Enum):
    TEXT = "text"
    VISION = "vision"
    CHAT = "chat"

@dataclass
class ModelConfig:
    name: str
    type: ModelType
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop_sequences: List[str] = None

@dataclass
class AppConfig:
    title: str = "💬 Chatbot Multimodal"
    description: str = "Un chatbot inteligente que puede procesar texto e imágenes"
    theme_color: str = "blue"
    layout: str = "centered"
    initial_sidebar_state: str = "expanded"
    supported_image_types: List[str] = None
    max_image_size_mb: int = 5

    def __post_init__(self):
        if self.supported_image_types is None:
            self.supported_image_types = ["jpg", "jpeg", "png", "webp"]

class Config:
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("API key no encontrada. Asegúrate de tener un archivo .env con GOOGLE_API_KEY")

        self.models = {
            "text": ModelConfig(
                name="gemini-pro",
                type=ModelType.TEXT,
                max_tokens=500,
                temperature=0.7,
                top_p=0.95,
                top_k=40,
                stop_sequences=["Usuario:", "Asistente:"]
            ),
            "vision": ModelConfig(
                name="gemini-pro-vision",
                type=ModelType.VISION,
                max_tokens=500,
                temperature=0.4,
                top_p=0.95,
                top_k=40
            )
        }

        self.app = AppConfig()
        
    def get_model_config(self, model_type: str) -> ModelConfig:
        return self.models.get(model_type)

    def update_model_params(self, model_type: str, **kwargs):
        model_config = self.models.get(model_type)
        if model_config:
            for key, value in kwargs.items():
                if hasattr(model_config, key):
                    setattr(model_config, key, value)