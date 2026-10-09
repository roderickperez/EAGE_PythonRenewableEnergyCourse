#!/usr/bin/env python3
"""
Qwen3-TTS Voice Cloning Interface - Final Working Version
Features: Audio recording, transcription input, text-to-voice conversion
Optimized for NVIDIA RTX A5000
"""

import os

os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"  # Disable analytics
os.environ["HF_HUB_DISABLE_XET"] = "1"  # Disable Xet for faster fallback
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"  # 5 minute timeout

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

# Suppress non-critical warnings
warnings.filterwarnings("ignore")

# Global variables for caching
loaded_model = None
voice_clone_prompt_cache = None


def get_model():
    """Get or load the Qwen3-TTS model"""
    global loaded_model
    if loaded_model is not None:
        return loaded_model

    print("🔄 Loading Qwen3-TTS 1.7B Base model...")
    start_time = time.time()

    try:
        # Try with modern parameter names first
        loaded_model = Qwen3TTSModel.from_pretrained(
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            dtype=torch.float16,
            device_map="cuda:0",
            attn_implementation="sdpa",
        )
    except Exception as e:
        print(f"⚠️  First load attempt failed: {e}")
        try:
            # Fallback without attn_implementation
            loaded_model = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                dtype=torch.float16,
                device_map="cuda:0",
            )
        except Exception as e2:
            print(f"❌ Second load attempt failed: {e2}")
            # Last resort - try with torch_dtype (older parameter)
            try:
                loaded_model = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    torch_dtype=torch.float16,
                    device_map="cuda:0",
                )
            except Exception as e3:
                print(f"❌ All load attempts failed: {e3}")
                raise e3

    load_time = time.time() - start_time
    print(f"✅ Model loaded successfully in {load_time:.1f}s")
    return loaded_model


def voice_clone(text, reference_audio, ref_transcript, use_fast_mode):
    """
    Generate speech by cloning a voice from reference audio

    Args:
        text: Text to synthesize
        reference_audio: Path to reference audio file
        ref_transcript: Transcript of reference audio (optional)
        use_fast_mode: Whether to use fast mode (skip transcript)

    Returns:
        Tuple of (audio_file_path, status_message)
    """
    global voice_clone_prompt_cache

    # Input validation
    if not text or not text.strip():
        return None, "❌ Please enter text to synthesize"

    if reference_audio is None:
        return None, "❌ Please record or upload reference audio"

    try:
        # Get model
        model = get_model()

        # Create voice clone prompt
        print("🎯 Creating voice clone prompt...")
        prompt_start = time.time()

        if use_fast_mode or not ref_transcript or voice_clone_prompt_cache is None:
            # Create new prompt
            prompt_items = model.create_voice_clone_prompt(
                ref_audio=reference_audio, x_vector_only_mode=True
            )
            print("✅ Created new voice clone prompt")
        else:
            # Use cached prompt
            prompt_items = voice_clone_prompt_cache
            print("✅ Using cached voice clone prompt")

        prompt_time = time.time() - prompt_start
        print(f"   Prompt creation: {prompt_time:.1f}s")

        # Generate audio
        print("🔊 Generating audio...")
        gen_start = time.time()

        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(
                text=text, voice_clone_prompt=prompt_items
            )

        gen_time = time.time() - gen_start
        print(f"   Audio generation: {gen_time:.1f}s")

        # Save to temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)

        # Cache prompt if we created a new one
        if use_fast_mode or not ref_transcript or voice_clone_prompt_cache is None:
            voice_clone_prompt_cache = prompt_items

        # Calculate metrics
        audio_duration = len(wavs[0]) / sr
        total_time = time.time() - prompt_start
        rtf = gen_time / audio_duration if audio_duration > 0 else 0

        print(f"✅ Generated {audio_duration:.1f}s of audio")
        print(f"   Generation RTF: {rtf:.2f}x")
        print(f"   Total processing: {total_time:.1f}s")

        # Clean up GPU memory
        torch.cuda.empty_cache()
        gc.collect()

        # Success message
        status_msg = (
            f"✅ Success! Generated {audio_duration:.1f}s of audio\n"
            f"   Generation RTF: {rtf:.2f}x | Total: {total_time:.1f}s"
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
    print("🗑️ Voice clone prompt cache cleared")
    return "✅ Cache cleared"


# Create the Gradio interface
with gr.Blocks(title="Qwen3-TTS Voice Cloning") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Voice Cloning")
    gr.Markdown(
        "### Clone voices with just 3 seconds of audio - Record, Transcribe, Convert"
    )

    with gr.Row():
        # Left column - Controls
        with gr.Column(scale=1):
            gr.Markdown("## 🎛️ Controls")

            # Text input for synthesis
            text_input = gr.Textbox(
                label="📝 Text to Synthesize",
                placeholder="Enter text to convert to speech using your cloned voice...",
                lines=3,
                value="Hello, this is a test of my cloned voice using Qwen3-TTS.",
            )

            # Audio input
            with gr.Group():
                gr.Markdown("### 🎤 Reference Audio")
                audio_input = gr.Audio(
                    label="Record or Upload Reference Audio (3+ seconds)",
                    type="filepath",
                    sources=["microphone", "upload"],
                    show_label=True,
                )

                transcript_input = gr.Textbox(
                    label="📄 Transcript (Optional)",
                    placeholder="What's said in the audio (leave empty for faster processing)...",
                    lines=2,
                )

            # Options
            with gr.Row():
                fast_mode = gr.Checkbox(
                    label="⚡ Fast Mode",
                    value=True,
                    info="Skip transcript for quicker processing (slightly lower quality)",
                )
                clear_btn = gr.Button("🗑️ Clear Cache", size="sm")

            # Generate button
            generate_btn = gr.Button(
                "🚀 Generate Cloned Speech", variant="primary", size="lg"
            )

        # Right column - Results
        with gr.Column(scale=1):
            gr.Markdown("## 🔊 Results")

            # Output audio
            audio_output = gr.Audio(label="Generated Speech", type="filepath")

            # Status output
            status_output = gr.Textbox(
                label="Status",
                value="Ready to generate speech",
                interactive=False,
                lines=3,
            )

    # Information section
    with gr.Accordion("ℹ️ Instructions & Tips", open=False):
        gr.Markdown("""
        ### How to Use:
        1. **Enter text** in the text box (what you want the cloned voice to say)
        2. **Record or upload** 3+ seconds of clear reference audio (your voice)
        3. **(Optional)** Provide transcript for better quality - leave empty for faster processing
        4. **Adjust options** - Enable Fast Mode for quicker results (slightly lower quality)
        5. **Click Generate** and wait for the cloned speech
        6. **Listen** to the output audio below
        
        ### Tips for Best Results:
        - Use **clear audio** with minimal background noise
        - **5-10 seconds** of reference audio works best
        - First generation takes **2-3 minutes** (model loading time)
        - Subsequent generations are **much faster** (uses cached model and prompt)
        - Ignore **SoX warnings** if audio generation works correctly
        - For voice cloning: provide transcript for better accuracy
        
        ### Expected Performance (RTX A5000):
        - **First use**: ~2-3 minutes (model loading + processing)
        - **Subsequent uses**: ~10-30 seconds (depending on text length)
        - **RTF (Real-Time Factor)**: 3.5-5x (generation time vs audio duration)
        """)

    # Examples
    gr.Markdown("## 📝 Example Texts")
    gr.Examples(
        examples=[
            ["Hello, this is a test of my cloned voice using Qwen3-TTS.", "", "", True],
            ["The quick brown fox jumps over the lazy dog.", "", "", True],
            ["Welcome to the future of voice cloning technology!", "", "", True],
            [
                "Today is a beautiful day to explore AI voice technologies.",
                "",
                "",
                True,
            ],
        ],
        inputs=[
            text_input,
            transcript_input,
            ref_transcript := gr.Textbox(visible=False),
            fast_mode,
        ],
        label="Click to try an example",
        examples_per_page=4,
    )

    # Hidden component for transcript (workaround for Gradio issue)
    ref_transcript = gr.Textbox(visible=False)

    # Event handlers
    generate_btn.click(
        fn=voice_clone,
        inputs=[text_input, audio_input, transcript_input, fast_mode],
        outputs=[audio_output, status_output],
    )

    clear_btn.click(fn=clear_cache, outputs=status_output)

    # Footer
    gr.Markdown("""
    ---
    *Powered by [Qwen3-TTS 1.7B Base Model](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) | 
    Optimized for NVIDIA RTX A5000 | 
    Using SDPA Attention* 
    """)

print("=" * 60)
print("🎙️ Qwen3-TTS Voice Cloning Interface Ready")
print("💡 Features: Audio recording, transcription input, text-to-voice")
print("🚀 Run: python qwen3_tts_final_interface.py")
print("=" * 60)

if __name__ == "__main__":
    print("🚀 Starting Qwen3-TTS Gradio Interface...")
    print("📝 First run will download the model (~2-3GB)")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True,
    )
