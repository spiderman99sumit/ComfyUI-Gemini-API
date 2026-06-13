# ComfyUI-Gemini-API

Custom ComfyUI nodes for Google Gemini AI models via API.

## Features

- **Gemini API Client** - Configure API key once, reuse across nodes
- **Gemini Generate** - Single-turn text generation with image/video input
- **Gemini Chat** - Multi-turn conversation with history

## Supported Models (Google AI Studio API)

### Gemini 3.x
| Model | Description |
|---|---|
| `gemini-3.5-flash` | Latest fast model |
| `gemini-3.1-pro` | Most capable reasoning |
| `gemini-3.1-flash-lite` | Fastest, cheapest |
| `gemini-3-flash` | Balanced performance |

### Gemini 2.5
| Model | Description |
|---|---|
| `gemini-2.5-flash` | Fast multimodal |
| `gemini-2.5-flash-lite` | Ultra lightweight |
| `gemini-2.5-pro` | Advanced reasoning |

### Gemini 2.0
| Model | Description |
|---|---|
| `gemini-2.0-flash` | Multimodal, fast |
| `gemini-2.0-flash-lite` | Lightweight |

### Gemma 4 (Open Source)
| Model | Description |
|---|---|
| `gemma-4-31b-it` | 31B parameter instruction-tuned |
| `gemma-4-26b-it` | 26B parameter instruction-tuned |

### Legacy
| Model | Description |
|---|---|
| `gemini-1.5-flash` | Legacy flash |
| `gemini-1.5-flash-8b` | Small efficient |
| `gemini-1.5-pro` | Legacy pro |
| `gemini-1.0-pro` | Baseline |

## Installation

1. Copy this folder to `ComfyUI/custom_nodes/ComfyUI-Gemini-API`
2. Install dependencies:
   ```bash
   cd ComfyUI/custom_nodes/ComfyUI-Gemini-API
   pip install -r requirements.txt
   ```
3. Restart ComfyUI

## Get an API Key

1. Go to [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a free API key
3. Paste it into the **Gemini API Client** node

## Node Usage

### Gemini API Client
Enter your API key. Connect the `client` output to other Gemini nodes.

### Gemini Generate
- **model** - Select from dropdown
- **prompt** - Your instruction / question
- **max_tokens** - Max response length (1-65536)
- **temperature** - Creativity (0=deterministic, 2=maximum random)
- **top_p** - Nucleus sampling
- **image** - Optional IMAGE input from Load Image node
- **video** - Optional path to video file (MP4, MOV, AVI, WEBM)
- **system_instruction** - Agent persona / system prompt
- **stop_sequences** - Comma-separated stop strings
- **safety_settings** - Content filtering level
- **seed** - For reproducibility (0=random)

### Gemini Chat
Multi-turn conversation. Pass `chat_history` JSON between calls.

```json
[
  {"role": "user", "parts": ["Hello"]},
  {"role": "model", "parts": ["Hi there!"]}
]
```

## Example Workflow

```
Gemini API Client → Gemini Generate (with Load Image)
```

```
Gemini API Client → Gemini Chat (loop for conversation)
```

## License

MIT
