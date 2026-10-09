#!/usr/bin/env python3
"""
Working Qwen3-TTS Gradio UI for Local Voice Cloning
Based on the original notebook but adapted for Gradio interface
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

# Suppress non-critical warnings
warnings.filterwarnings("ignore", message="SoX could not be found")
warnings.filterwarnings("ignore", message="`torch_dtype` is deprecated")

# Global variables
current_model = None
voice_clone_prompt_cache = None


def load_model():
    """Load the base model for voice cloning"""
    global current_model

    if current_model is not None:
        print("✅ Using cached model")
        return current_model

    print("Loading Qwen3-TTS 1.7B Base model...")
    start = time.time()

    try:
        current_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            dtype=torch.float16,  # Updated parameter name
            device_map="cuda:0",
            attn_implementation="sdpa",
        )

        load_time = time.time() - start
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✅ Loaded in {load_time:.1f}s | GPU: {allocated:.2f}GB")

        return current_model

    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def voice_clone(text, reference_audio, ref_transcript, use_fast_mode):
    """Generate speech by cloning a reference voice"""
    global voice_clone_prompt_cache

    if not text or reference_audio is None:
        return None, "Please provide both text and reference audio"

    try:
        total_start = time.time()
        model = load_model()
        if model is None:
            return None, "Failed to load model"

        print(f"⏱️ Creating prompt...")
        prompt_start = time.time()

        # Use cached prompt if available and not forcing regeneration
        if voice_clone_prompt_cache is not None and not use_fast_mode:
            print("✅ Using cached voice clone prompt")
            prompt_items = voice_clone_prompt_cache
        else:
            if use_fast_mode or not ref_transcript:
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=reference_audio, x_vector_only_mode=True
                )
            else:
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=reference_audio,
                    ref_text=ref_transcript,
                    x_vector_only_mode=False,
                )
                # Cache the prompt for future use
                voice_clone_prompt_cache = prompt_items

        prompt_time = time.time() - prompt_start
        print(f"   Prompt: {prompt_time:.1f}s")

        print(f"⏱️ Generating audio...")
        gen_start = time.time()

        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(
                text=text, voice_clone_prompt=prompt_items
            )

        gen_time = time.time() - gen_start

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)

        total_time = time.time() - total_start
        audio_duration = len(wavs[0]) / sr
        rtf = gen_time / audio_duration if audio_duration > 0 else 0

        print(
            f"✅ Done! Total: {total_time:.1f}s | Gen: {gen_time:.1f}s | Audio: {audio_duration:.1f}s | RTF: {rtf:.2f}x"
        )

        torch.cuda.empty_cache()
        gc.collect()

        status_msg = f"✅ Success! RTF: {rtf:.2f}x | Audio: {audio_duration:.1f}s"
        return temp_file.name, status_msg

    except Exception as e:
        print(f"❌ Error in voice_clone: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


def clear_cache():
    """Clear the voice clone prompt cache"""
    global voice_clone_prompt_cache
    voice_clone_prompt_cache = None
    print("🗑️ Voice clone prompt cache cleared")
    return "Cache cleared"


# Custom CSS for clean interface
custom_css = """
.gradio-container {
    max-width: 800px !important;
    margin: 0 auto !important;
}
.info-text {
    background-color: #f8f9fa;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    border-left: 4px solid #667eea;
}
"""

# Gradio Interface
with gr.Blocks(title="Qwen3-TTS Voice Cloning", css=custom_css) as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS: Local Voice Cloning")
    gr.Markdown("### Clone your voice with just 3 seconds of audio")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("## 🎤 Voice Cloning Interface")

            clone_text = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text to convert to speech using your cloned voice...",
                lines=3,
                value="Hello, this is a test of my cloned voice using Qwen3-TTS.",
            )

            with gr.Row():
                clone_audio = gr.Audio(
                    label="Record Reference Audio (3+ seconds)",
                    type="filepath",
                    sources=["microphone", "upload"],
                    scale=2,
                )
                clone_transcript = gr.Textbox(
                    label="Transcript (Optional)",
                    placeholder="What's said in the audio (leave empty for fast mode)...",
                    lines=2,
                    scale=1,
                )

            with gr.Row():
                clone_fast_mode = gr.Checkbox(
                    label="⚡ Fast Mode (skip transcript for quicker processing)",
                    value=True,
                    scale=1,
                )
                clear_btn = gr.Button("🗑️ Clear Cache", scale=1)

            generate_btn = gr.Button(
                "🎵 Generate Cloned Speech", variant="primary", size="lg", scale=2
            )

            status_output = gr.Textbox(label="Status", interactive=False, lines=2)

        with gr.Column(scale=1):
            gr.Markdown("## 🔊 Output")
            clone_output = gr.Audio(label="Generated Speech", type="filepath")

            gr.Markdown(
                """
            <div class="info-text">
            <strong>How to use:</strong><br>
            1. Record or upload 3+ seconds of clear audio<br>
            2. (Optional) Provide transcript for better quality<br>
            3. Enter text to synthesize<br>
            4. Click Generate<br>
            5. Listen to your cloned voice!
            </div>
            """,
            )

            gr.Markdown("""
            <div class="info-text">
            <strong>Tips for best results:</strong><br>
            • Use clear audio with minimal background noise<br>
            • 5-10 seconds of reference audio works best<br>
            • First generation includes model loading time (~2-3 min)<br>
            • Subsequent generations are much faster<br>
            • Ignore SoX warning if audio generation works
            </div>
            """)

    # Examples
    gr.Markdown("## 📝 Example Texts")
    gr.Examples(
        examples=[
            [
                "Hello, this is a test of my cloned voice using Qwen3-TTS.",
                None,
                "",
                True,
            ],
            ["The quick brown fox jumps over the lazy dog.", None, "", True],
            ["Welcome to the future of voice cloning technology!", None, "", True],
        ],
        inputs=[clone_text, clone_audio, clone_transcript, clone_fast_mode],
        label="Click to try an example",
    )

    # Event handlers
    generate_btn.click(
        fn=voice_clone,
        inputs=[clone_text, clone_audio, clone_transcript, clone_fast_mode],
        outputs=[clone_output, status_output],
    )

    clear_btn.click(fn=clear_cache, outputs=status_output)

    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    **Powered by Qwen3-TTS 1.7B Base Model** | 
    **Running on NVIDIA RTX A5000** | 
    **Using SDPA Optimization**
    """)

print("=" * 50)
print("🎙️ Qwen3-TTS Voice Cloning UI Ready")
print("💡 Save this file and run: python working_qwen3_tts_ui.py")
print("=" * 50)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        css=custom_css,
        show_error=True,
    )
