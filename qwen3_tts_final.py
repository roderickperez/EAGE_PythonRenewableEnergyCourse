#!/usr/bin/env python3
"""
Qwen3-TTS Gradio UI - Final Working Version
"""

import gradio as gr
import torch
import soundfile as sf
import numpy as np
import tempfile
import os
import time
import gc
import warnings
from pathlib import Path
from qwen_tts import Qwen3TTSModel

# Suppress warnings
warnings.filterwarnings("ignore")

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
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )

        load_time = time.time() - start_time
        print(f"✅ Model loaded in {load_time:.1f}s")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None


def voice_clone(text, reference_audio, ref_transcript, use_fast_mode):
    """Generate speech using voice cloning"""
    global voice_clone_prompt_cache

    if not text or reference_audio is None:
        return None, "Please provide text and reference audio"

    try:
        # Load model
        tts_model = load_model()
        if tts_model is None:
            return None, "Failed to load model"

        # Create prompt
        if use_fast_mode or not ref_transcript or voice_clone_prompt_cache is None:
            print("Creating voice clone prompt...")
            prompt_items = tts_model.create_voice_clone_prompt(
                ref_audio=reference_audio, x_vector_only_mode=True
            )
        else:
            print("Using cached prompt")
            prompt_items = voice_clone_prompt_cache

        # Generate audio
        print("Generating audio...")
        with torch.inference_mode():
            wavs, sr = tts_model.generate_voice_clone(
                text=text, voice_clone_prompt=prompt_items
            )

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)

        # Cache prompt if we created a new one
        if use_fast_mode or not ref_transcript or voice_clone_prompt_cache is None:
            voice_clone_prompt_cache = prompt_items

        # Calculate metrics
        audio_len = len(wavs[0]) / sr
        print(f"✅ Generated {audio_len:.1f}s of audio")

        return temp_file.name, f"Success! Generated {audio_len:.1f}s of audio"

    except Exception as e:
        print(f"❌ Error in voice_clone: {e}")
        import traceback

        traceback.print_exc()
        return None, f"Error: {str(e)}"


def clear_cache():
    """Clear the voice clone prompt cache"""
    global voice_clone_prompt_cache
    voice_clone_prompt_cache = None
    print("🗑️ Cache cleared")
    return "Cache cleared"


# Create interface
with gr.Blocks(title="Qwen3-TTS Voice Cloning") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Voice Cloning")
    gr.Markdown("Clone voices with just 3 seconds of audio")

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text to convert to speech",
                lines=3,
            )

            audio_input = gr.Audio(
                label="Reference Audio (3+ seconds)",
                type="filepath",
                sources=["microphone", "upload"],
            )

            transcript_input = gr.Textbox(
                label="Transcript (Optional)",
                placeholder="What's said in the audio",
                lines=2,
            )

            fast_mode = gr.Checkbox(label="Fast Mode", value=True)

            with gr.Row():
                generate_btn = gr.Button("🎵 Generate Speech", variant="primary")
                clear_btn = gr.Button("🗑️ Clear Cache")

        with gr.Column():
            audio_output = gr.Audio(label="Output")
            status_output = gr.Textbox(label="Status", interactive=False)

    # Event handlers
    generate_btn.click(
        fn=voice_clone,
        inputs=[text_input, audio_input, transcript_input, fast_mode],
        outputs=[audio_output, status_output],
    )

    clear_btn.click(fn=clear_cache, outputs=status_output)

    gr.Markdown("""
    ### Instructions:
    1. Enter text to synthesize
    2. Record or upload 3+ seconds of reference audio
    3. (Optional) Provide transcript for better quality
    4. Click Generate Speech
    5. Listen to the cloned voice output
    
    ### Notes:
    - First generation takes 2-3 minutes (model loading)
    - Subsequent generations are much faster
    - Ignore SoX warnings if audio generation works
    """)

if __name__ == "__main__":
    print("🚀 Starting Qwen3-TTS Gradio Interface...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False)
