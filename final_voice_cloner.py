#!/usr/bin/env python3
"""
Final Qwen3-TTS Voice Cloning Interface
Record audio, add transcription, convert text to voice
"""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
os.environ["HF_HUB_DISABLE_XET"] = "1"

import warnings

warnings.filterwarnings("ignore")

import gradio as gr
import torch
import soundfile as sf
import numpy as np
import tempfile
import time
import gc
from pathlib import Path
from qwen_tts import Qwen3TTSModel

# Global variables
model = None
voice_clone_prompt_cache = None


def load_model():
    """Load the Qwen3-TTS model"""
    global model
    if model is not None:
        return model

    print("Loading Qwen3-TTS 1.7B Base model...")
    start_time = time.time()

    try:
        model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base", dtype=torch.float16, device_map="cuda:0"
        )

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        # Try with older parameter name
        try:
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                torch_dtype=torch.float16,
                device_map="cuda:0",
            )
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.1f}s (legacy params)")
            return model
        except Exception as e2:
            print(f"❌ Failed to load model: {e2}")
            return None


def voice_clone(text, reference_audio, ref_transcript, use_fast_mode):
    """Generate speech using voice cloning"""
    global voice_clone_prompt_cache

    # Input validation
    if not text or not text.strip():
        return None, "Please enter text to synthesize"

    if reference_audio is None:
        return None, "Please record or upload reference audio"

    try:
        # Load model
        tts_model = load_model()
        if tts_model is None:
            return None, "Failed to load model"

        # Create voice clone prompt
        if use_fast_mode or not ref_transcript or voice_clone_prompt_cache is None:
            prompt_items = tts_model.create_voice_clone_prompt(
                ref_audio=reference_audio, x_vector_only_mode=True
            )
        else:
            prompt_items = voice_clone_prompt_cache

        # Generate audio
        with torch.inference_mode():
            wavs, sr = tts_model.generate_voice_clone(
                text=text, voice_clone_prompt=prompt_items
            )

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)

        # Cache prompt if needed
        if use_fast_mode or not ref_transcript or voice_clone_prompt_cache is None:
            voice_clone_prompt_cache = prompt_items

        # Calculate metrics
        audio_duration = len(wavs[0]) / sr

        # Cleanup
        torch.cuda.empty_cache()
        gc.collect()

        return temp_file.name, f"Success! Generated {audio_duration:.1f}s of audio"

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return None, f"Error: {str(e)}"


def clear_cache():
    """Clear the voice clone prompt cache"""
    global voice_clone_prompt_cache
    voice_clone_prompt_cache = None
    return "Cache cleared"


# Simple interface focused on the core request
with gr.Blocks(title="Qwen3-TTS Voice Cloner") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Voice Cloner")
    gr.Markdown("### Record audio, add transcription, convert text to voice")

    with gr.Row():
        with gr.Column():
            # Text to synthesize
            text_input = gr.Textbox(
                label="📝 Text to Synthesize",
                placeholder="Enter the text you want to convert to speech...",
                lines=3,
                value="Hello, this is a test of my cloned voice using Qwen3-TTS.",
            )

            # Reference audio recording/upload
            audio_input = gr.Audio(
                label="🎤 Record Reference Audio (3+ seconds)",
                type="filepath",
                sources=["microphone", "upload"],
            )

            # Transcription input
            transcript_input = gr.Textbox(
                label="📄 Transcription (Optional - improves quality)",
                placeholder="What is said in the audio above? Leave empty for faster processing...",
                lines=2,
            )

            # Fast mode option
            fast_mode = gr.Checkbox(
                label="⚡ Fast Mode (skip transcription for quicker processing)",
                value=True,
            )

            # Buttons
            with gr.Row():
                generate_btn = gr.Button(
                    "🚀 Generate Speech", variant="primary", size="lg"
                )
                clear_btn = gr.Button("🗑️ Clear Cache", size="sm")

        with gr.Column():
            # Output audio
            audio_output = gr.Audio(label="🔊 Generated Speech")

            # Status
            status_output = gr.Textbox(label="Status", interactive=False, lines=2)

    # Event handlers
    generate_btn.click(
        fn=voice_clone,
        inputs=[text_input, audio_input, transcript_input, fast_mode],
        outputs=[audio_output, status_output],
    )

    clear_btn.click(fn=clear_cache, outputs=status_output)

    # Instructions
    gr.Markdown("""
    ### How to Use:
    1. **Enter text** in the text box (what you want the voice to say)
    2. **Record or upload** 3+ seconds of your voice (clear audio works best)
    3. **(Optional)** Provide transcription of what's in the audio for better quality
    4. **Click Generate Speech** and wait for processing
    5. **Listen** to the cloned voice output
    
    ### Notes:
    - First generation takes 2-3 minutes (model loading)
    - Subsequent generations are much faster (10-30 seconds)
    - Ignore SoX warnings if audio generation works correctly
    - For best results: use clear audio with minimal background noise
    """)

print("=" * 50)
print("🎙️ Qwen3-TTS Voice Cloner Ready")
print("💡 Run this script to start the interface")
print("=" * 50)

if __name__ == "__main__":
    print("🚀 Starting Qwen3-TTS Voice Cloner...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False)
