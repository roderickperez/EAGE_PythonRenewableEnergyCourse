#!/usr/bin/env python3
"""
Working Qwen3-TTS Gradio UI for Local Voice Cloning
"""

import os
import gradio as gr
import torch
import soundfile as sf
import numpy as np
import tempfile
import time
import gc
import warnings
from pathlib import Path
from qwen_tts import Qwen3TTSModel

# Suppress warnings
warnings.filterwarnings("ignore", message="SoX could not be found")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

# Global variables
model = None
voice_clone_prompt_cache = None


def load_model():
    """Load the Qwen3-TTS model for voice cloning"""
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
        # Try without attn_implementation if that's causing issues
        try:
            print("Trying without attn_implementation...")
            model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                dtype=torch.float16,
                device_map="cuda:0",
            )
            load_time = time.time() - start_time
            print(f"✅ Model loaded in {load_time:.1f}s (no attn_implementation)")
            return model
        except Exception as e2:
            print(f"❌ Second attempt failed: {e2}")
            return None


def voice_clone(text, reference_audio, ref_transcript, use_fast_mode):
    """Generate speech using voice cloning"""
    global voice_clone_prompt_cache

    # Validate inputs
    if not text or not text.strip():
        return None, "Please enter text to synthesize"

    if reference_audio is None:
        return None, "Please record or upload reference audio"

    try:
        # Load model
        tts_model = load_model()
        if tts_model is None:
            return None, "Failed to load model. Check console for details."

        # Create prompt
        print("Creating voice clone prompt...")
        prompt_start = time.time()

        if use_fast_mode or not ref_transcript or voice_clone_prompt_cache is None:
            prompt_items = tts_model.create_voice_clone_prompt(
                ref_audio=reference_audio, x_vector_only_mode=True
            )
            print("✅ Created new prompt")
        else:
            prompt_items = voice_clone_prompt_cache
            print("✅ Using cached prompt")

        prompt_time = time.time() - prompt_start
        print(f"   Prompt creation time: {prompt_time:.1f}s")

        # Generate audio
        print("Generating audio...")
        gen_start = time.time()

        with torch.inference_mode():
            wavs, sr = tts_model.generate_voice_clone(
                text=text, voice_clone_prompt=prompt_items
            )

        gen_time = time.time() - gen_start
        print(f"   Generation time: {gen_time:.1f}s")

        # Save to temp file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)

        # Cache prompt if we created a new one
        if use_fast_mode or not ref_transcript or voice_clone_prompt_cache is None:
            voice_clone_prompt_cache = prompt_items

        # Calculate metrics
        audio_len = len(wavs[0]) / sr
        total_time = time.time() - start_time
        rtf = gen_time / audio_len if audio_len > 0 else 0

        print(f"✅ Generated {audio_len:.1f}s of audio in {total_time:.1f}s total")
        print(f"   Generation RTF: {rtf:.2f}x")

        status_msg = (
            f"✅ Success! Generated {audio_len:.1f}s of audio (RTF: {rtf:.2f}x)"
        )
        return temp_file.name, status_msg

    except Exception as e:
        print(f"❌ Error in voice_clone: {e}")
        import traceback

        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


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
                value="Hello, this is a test of my cloned voice using Qwen3-TTS.",
            )

            audio_input = gr.Audio(
                label="Reference Audio (3+ seconds)",
                type="filepath",
                sources=["microphone", "upload"],
            )

            transcript_input = gr.Textbox(
                label="Transcript (Optional)",
                placeholder="What's said in the audio (leave empty for faster processing)",
                lines=2,
            )

            fast_mode = gr.Checkbox(
                label="Fast Mode (skip transcript for quicker processing)", value=True
            )

            with gr.Row():
                generate_btn = gr.Button(
                    "🎵 Generate Speech", variant="primary", size="lg"
                )
                clear_btn = gr.Button("🗑️ Clear Cache", size="sm")

        with gr.Column():
            audio_output = gr.Audio(label="Output")
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
    ### Instructions:
    1. Enter text to synthesize in the top box
    2. Record or upload 3+ seconds of reference audio (your voice)
    3. (Optional) Provide transcript for better quality - leave empty for faster processing
    4. Click "Generate Speech"
    5. Listen to the cloned voice output below
    
    ### Notes:
    - First generation takes 2-3 minutes (model loading)
    - Subsequent generations are much faster (uses cached model)
    - Ignore SoX warnings if audio generation works correctly
    - For best results: use clear audio with minimal background noise
    """)

    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    **Powered by Qwen3-TTS 1.7B Base Model** | 
    **Running on NVIDIA RTX A5000** | 
    **Using SDPA Optimization**
    """)

print("=" * 50)
print("🎙️ Qwen3-TTS Voice Cloning UI Ready")
print("💡 Run: python qwen3_tts_working.py")
print("=" * 50)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
    )
