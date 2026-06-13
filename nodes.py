import json
import os
import base64
import struct
import tempfile
import folder_paths
from io import BytesIO

try:
    import google.generativeai as genai
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


GEMINI_MODELS = [
    # Gemini 3.x
    "gemini-3.5-flash",
    "gemini-3.1-pro",
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    # Gemini 2.5
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    # Gemini 2.0
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    # Gemma 4
    "gemma-4-31b-it",
    "gemma-4-26b-it",
    # Legacy
    "gemini-1.5-flash",
    "gemini-1.5-flash-8b",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
]

DEFAULT_SAFETY_SETTINGS = {
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}


class GeminiClient:
    """Configures the Gemini API client with an API key."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Your Google AI Studio / Gemini API key"
                }),
            },
        }

    RETURN_TYPES = ("GEMINI_CLIENT",)
    RETURN_NAMES = ("client",)
    FUNCTION = "configure"
    CATEGORY = "Gemini/API"
    DESCRIPTION = "Configure Gemini API client with your API key."

    def configure(self, api_key):
        if not api_key or api_key.strip() == "":
            raise ValueError("API key is required. Get one from https://aistudio.google.com/apikey")

        if not HAS_GOOGLE_GENAI:
            raise ImportError(
                "google-generativeai not installed. "
                "Run: pip install google-generativeai"
            )

        genai.configure(api_key=api_key.strip())

        try:
            genai.list_models()
        except Exception as e:
            raise ValueError(f"API key validation failed: {e}")

        return ({"api_key": api_key.strip()},)


class GeminiGenerate:
    """Generate text responses from Gemini models with optional image/video input."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("GEMINI_CLIENT", {"forceInput": True}),
                "model": (GEMINI_MODELS,),
                "prompt": ("STRING", {
                    "default": "Describe this image in detail.",
                    "multiline": True,
                    "tooltip": "Your prompt / system instruction"
                }),
                "max_tokens": ("INT", {
                    "default": 8192,
                    "min": 1,
                    "max": 65536,
                    "step": 1,
                    "tooltip": "Maximum number of tokens in the response"
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                    "tooltip": "Sampling temperature (0 = deterministic, 2 = most random)"
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                    "tooltip": "Nucleus sampling threshold"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "video": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Path to a video file (MP4, MOV, AVI, WEBM)"
                }),
                "system_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                    "tooltip": "Optional system instruction / agent persona"
                }),
                "stop_sequences": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Comma-separated stop sequences"
                }),
                "safety_settings": (["BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH"],),
                "seed": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 0xFFFFFFFF,
                    "tooltip": "Random seed for reproducibility (0 = random)"
                }),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text", "model_used", "token_count")
    FUNCTION = "generate"
    CATEGORY = "Gemini/API"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def _compress_image(self, image_tensor):
        """Convert ComfyUI IMAGE tensor to compressed JPEG bytes."""
        if not HAS_PIL:
            raise ImportError("Pillow not installed. Run: pip install Pillow")

        img_array = image_tensor.squeeze(0).cpu().numpy()
        img_array = (img_array * 255).astype("uint8")

        if img_array.ndim == 3 and img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        img = Image.fromarray(img_array)
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        return buffer.getvalue()

    def _load_video(self, video_path):
        """Load video file and return bytes."""
        video_path = video_path.strip()
        if not video_path:
            return None

        if not os.path.isabs(video_path):
            for search_dir in [
                folder_paths.get_output_directory(),
                folder_paths.get_input_directory(),
                os.path.join(folder_paths.base_path, "custom_nodes"),
                os.path.dirname(os.path.abspath(__file__)),
            ]:
                full_path = os.path.join(search_dir, video_path)
                if os.path.isfile(full_path):
                    video_path = full_path
                    break

        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        with open(video_path, "rb") as f:
            return f.read()

    def _get_safety_settings(self, level):
        """Map safety setting string to Google safety thresholds."""
        mapping = {
            "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
            "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        threshold = mapping.get(level, HarmBlockThreshold.BLOCK_NONE)
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
        }

    def generate(self, client, model, prompt, max_tokens, temperature, top_p,
                 image=None, video="", system_instruction="",
                 stop_sequences="", safety_settings="BLOCK_NONE", seed=0):

        if not HAS_GOOGLE_GENAI:
            raise ImportError("google-generativeai not installed.")

        genai.configure(api_key=client["api_key"])

        model_instance = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction if system_instruction.strip() else None,
        )

        contents = []

        if image is not None:
            image_bytes = self._compress_image(image)
            contents.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(image_bytes).decode("utf-8"),
                }
            })

        if video and video.strip():
            video_bytes = self._load_video(video)
            if video_bytes is not None:
                ext = os.path.splitext(video.strip())[1].lower()
                mime_map = {
                    ".mp4": "video/mp4",
                    ".mov": "video/quicktime",
                    ".avi": "video/x-msvideo",
                    ".webm": "video/webm",
                    ".mkv": "video/x-matroska",
                }
                mime_type = mime_map.get(ext, "video/mp4")
                contents.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": base64.b64encode(video_bytes).decode("utf-8"),
                    }
                })

        contents.append(prompt)

        gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        if stop_sequences and stop_sequences.strip():
            gen_config.stop_sequences = [s.strip() for s in stop_sequences.split(",") if s.strip()]

        if seed > 0:
            gen_config.seed = seed

        safety = self._get_safety_settings(safety_settings)

        response = model_instance.generate_content(
            contents=contents,
            generation_config=gen_config,
            safety_settings=safety,
        )

        text = response.text if response.text else ""

        token_count = 0
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            token_count = getattr(response.usage_metadata, "total_token_count", 0) or 0

        return {
            "ui": {"text": [text]},
            "result": (text, model, token_count),
        }


class GeminiChat:
    """Multi-turn chat with Gemini models."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "client": ("GEMINI_CLIENT", {"forceInput": True}),
                "model": (GEMINI_MODELS,),
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "max_tokens": ("INT", {
                    "default": 8192,
                    "min": 1,
                    "max": 65536,
                }),
                "temperature": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "top_p": ("FLOAT", {
                    "default": 0.95,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.05,
                }),
                "chat_history": ("STRING", {
                    "default": "[]",
                    "multiline": True,
                    "tooltip": "JSON array of previous messages: [{\"role\":\"user\",\"parts\":[\"text\"]},{\"role\":\"model\",\"parts\":[\"text\"]}]"
                }),
            },
            "optional": {
                "image": ("IMAGE",),
                "system_instruction": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "safety_settings": (["BLOCK_NONE", "BLOCK_LOW_AND_ABOVE", "BLOCK_MEDIUM_AND_ABOVE", "BLOCK_ONLY_HIGH"],),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("response", "updated_history", "model_used")
    FUNCTION = "chat"
    CATEGORY = "Gemini/API"
    OUTPUT_NODE = True

    @classmethod
    def IS_CHANGED(cls, *args, **kwargs):
        return float("NaN")

    def chat(self, client, model, prompt, max_tokens, temperature, top_p,
             chat_history="[]", image=None, system_instruction="",
             safety_settings="BLOCK_NONE"):

        if not HAS_GOOGLE_GENAI:
            raise ImportError("google-generativeai not installed.")

        genai.configure(api_key=client["api_key"])

        model_instance = genai.GenerativeModel(
            model_name=model,
            system_instruction=system_instruction if system_instruction.strip() else None,
        )

        try:
            history = json.loads(chat_history) if chat_history.strip() else []
        except json.JSONDecodeError:
            history = []

        chat = model_instance.start_chat(history=history)

        user_parts = []
        if image is not None:
            from io import BytesIO
            img_array = image.squeeze(0).cpu().numpy()
            img_array = (img_array * 255).astype("uint8")
            if img_array.ndim == 3 and img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
            pil_img = Image.fromarray(img_array)
            buf = BytesIO()
            pil_img.save(buf, format="JPEG", quality=85)
            user_parts.append({
                "inline_data": {
                    "mime_type": "image/jpeg",
                    "data": base64.b64encode(buf.getvalue()).decode("utf-8"),
                }
            })

        user_parts.append(prompt)

        gen_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        safety = self._get_safety_settings(safety_settings)

        response = chat.send_message(
            user_parts,
            generation_config=gen_config,
            safety_settings=safety,
        )

        text = response.text if response.text else ""
        updated_history = json.dumps(chat.history, default=str)

        return {
            "ui": {"text": [text]},
            "result": (text, updated_history, model),
        }

    def _get_safety_settings(self, level):
        mapping = {
            "BLOCK_NONE": HarmBlockThreshold.BLOCK_NONE,
            "BLOCK_LOW_AND_ABOVE": HarmBlockThreshold.BLOCK_LOW_AND_ABOVE,
            "BLOCK_MEDIUM_AND_ABOVE": HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
            "BLOCK_ONLY_HIGH": HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }
        threshold = mapping.get(level, HarmBlockThreshold.BLOCK_NONE)
        return {
            HarmCategory.HARM_CATEGORY_HARASSMENT: threshold,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: threshold,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: threshold,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: threshold,
        }


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
