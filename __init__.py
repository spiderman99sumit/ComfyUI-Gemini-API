from .nodes import (
    GeminiClient,
    GeminiGenerate,
    GeminiChat,
)

NODE_CLASS_MAPPINGS = {
    "GeminiClient": GeminiClient,
    "GeminiGenerate": GeminiGenerate,
    "GeminiChat": GeminiChat,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "GeminiClient": "Gemini API Client",
    "GeminiGenerate": "Gemini Generate",
    "GeminiChat": "Gemini Chat",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
