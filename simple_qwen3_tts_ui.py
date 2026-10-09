#!/usr/bin/env python3
"""
Simple Qwen3-TTS Gradio UI with audio recording
Fixed LSP errors
"""

import gradio as gr
import torch
import soundfile as sf
import numpy as np
import tempfile
import os
import time
import gc
from pathlib import Path
from qwen_tts import Qwen3TTSModel

# Global variables
current_model = None
current_model_type = None
voice_clone_prompt_cache = None

# Enable PyTorch optimizations (fixed for compatibility)
if hasattr(torch.backends, "cudnn"):
    if hasattr(torch.backends.cudnn, "benchmark"):
        torch.backends.cudnn.benchmark = True

if hasattr(torch.backends, "cuda"):
    if hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = True

if hasattr(torch.backends, "cudnn"):
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def load_model(model_type):
    """Load model with SDPA optimization"""
    global current_model, current_model_type

    if current_model_type == model_type:
        print(f"✅ Using cached {model_type} model")
        return current_model

    if current_model is not None:
        print(f"Unloading {current_model_type} model...")
        del current_model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"Loading {model_type} model (1.7B)...")
    start = time.time()

    try:
        if model_type == "base":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        elif model_type == "custom":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
        elif model_type == "design":
            model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        current_model = Qwen3TTSModel.from_pretrained(
            model_name,
            dtype=torch.float16,  # Fixed parameter name
            device_map="cuda:0",
            attn_implementation="sdpa",
        )

        current_model_type = model_type
        load_time = time.time() - start

        allocated = torch.cuda.memory_allocated(0) / 1024**3
        print(f"✅ Loaded in {load_time:.1f}s | GPU: {allocated:.2f}GB")

        return current_model

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def voice_clone(text, reference_audio, ref_transcript, use_fast_mode):
    """Generate speech by cloning a reference voice"""
    global voice_clone_prompt_cache

    if not text or reference_audio is None:
        return None

    try:
        total_start = time.time()
        model = load_model("base")
        if model is None:
            return None

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

        return temp_file.name

    except Exception as e:
        print(f"❌ Error in voice_clone: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def custom_voice(text, voice_name, instruction):
    """Generate speech using preset voices"""
    if not text:
        return None

    try:
        total_start = time.time()
        model = load_model("custom")
        if model is None:
            return None

        print(f"⏱️ Generating with voice: {voice_name}...")
        if instruction and instruction.strip():
            print(f"   Style instruction: '{instruction}'")

        gen_start = time.time()

        with torch.inference_mode():
            if instruction and instruction.strip():
                wavs, sr = model.generate_custom_voice(
                    text=text, speaker=voice_name, instruct=instruction
                )
            else:
                wavs, sr = model.generate_custom_voice(text=text, speaker=voice_name)

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

        return temp_file.name

    except Exception as e:
        print(f"❌ Error in custom_voice: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def voice_design(text, voice_description):
    """Generate speech from text description"""
    if not text or not voice_description:
        return None

    try:
        total_start = time.time()
        model = load_model("design")
        if model is None:
            return None

        print(f"⏱️ Generating...")
        gen_start = time.time()

        with torch.inference_mode():
            wavs, sr = model.generate_voice_design(
                text=text, instruct=voice_description
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

        return temp_file.name

    except Exception as e:
        print(f"❌ Error in voice_design: {str(e)}")
        import traceback

        traceback.print_exc()
        return None


def clear_cache():
    """Clear the voice clone prompt cache"""
    global voice_clone_prompt_cache
    voice_clone_prompt_cache = None
    print("🗑️ Voice clone prompt cache cleared")
    return "Cache cleared"


# Custom CSS for clean interface
custom_css = """
.gradio-container {
    max-width: 900px !important;
    margin: 0 auto !important;
}
"""

# Gradio Interface
with gr.Blocks(title="Qwen3-TTS Local Voice Cloning") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS: Local Voice Cloning & Generation")
    gr.Markdown("### Advanced Text-to-Speech AI for NVIDIA RTX A5000")

    with gr.Tab("🎤 Voice Cloning"):
        gr.Markdown("### Clone any voice with recorded audio")

        with gr.Row():
            with gr.Column():
                clone_text = gr.Textbox(
                    label="Text to Synthesize",
                    placeholder="Enter text to convert to speech...",
                    lines=3,
                )
                clone_audio = gr.Audio(
                    label="Record Reference Audio (3+ seconds)",
                    type="filepath",
                    sources=["microphone", "upload"],
                )
                clone_transcript = gr.Textbox(
                    label="Transcript (Optional - improves quality)",
                    placeholder="What's said in the audio...",
                    lines=2,
                )
                clone_fast_mode = gr.Checkbox(
                    label="Fast Mode (skip transcript for quicker processing)",
                    value=True,
                )
                with gr.Row():
                    clone_btn = gr.Button(
                        "🎵 Generate Speech", variant="primary", size="lg"
                    )
                    clear_cache_btn = gr.Button("🗑️ Clear Prompt Cache", size="sm")

                cache_status = gr.Textbox(
                    label="Cache Status", value="Ready", interactive=False
                )

            with gr.Column():
                clone_output = gr.Audio(label="Generated Speech")
                gr.Markdown("""
                **Instructions:**
                1. Record or upload 3+ seconds of clear audio
                2. Provide transcript for better quality (optional)
                3. Enter text to synthesize
                4. Click Generate Speech
                
                **Tips:**
                - Use clear audio with minimal background noise
                - 5-10 seconds of reference audio works best
                - First generation includes model loading time (~2-3 min)
                """)

        clone_btn.click(
            voice_clone,
            inputs=[clone_text, clone_audio, clone_transcript, clone_fast_mode],
            outputs=clone_output,
        )

        clear_cache_btn.click(clear_cache, outputs=cache_status)

    with gr.Tab("🎭 Custom Voice"):
        gr.Markdown("### Use preset character voices with style control")

        with gr.Row():
            with gr.Column():
                custom_text = gr.Textbox(
                    label="Text to Synthesize", placeholder="Enter text...", lines=3
                )
                custom_voice_name = gr.Dropdown(
                    choices=[
                        "serena",  # Female voice
                        "vivian",  # Female voice
                        "ono_anna",  # Female voice (Japanese-style)
                        "sohee",  # Female voice (Korean-style)
                        "aiden",  # Male voice
                        "dylan",  # Male voice
                        "eric",  # Male voice
                        "ryan",  # Male voice
                        "uncle_fu",  # Male voice (Chinese-style)
                    ],
                    label="Voice Character",
                    value="serena",
                )
                custom_instruction = gr.Textbox(
                    label="Style Instruction (Optional)",
                    placeholder="e.g., 'speak slowly and cheerfully'",
                    lines=2,
                )
                custom_btn = gr.Button(
                    "🎵 Generate Speech", variant="primary", size="lg"
                )

                gr.Markdown("""
                **Voice Guide:**
                - **Female**: serena, vivian, ono_anna, sohee
                - **Male**: aiden, dylan, eric, ryan, uncle_fu
                
                **Style Instructions Examples:**
                - "speak slowly and clearly"
                - "cheerful and energetic"
                - "whisper softly"
                - "authoritative tone"
                """)

            with gr.Column():
                custom_output = gr.Audio(label="Generated Speech")
                gr.Markdown("**Expected Speed**: RTF 3.5-5x on RTX A5000")

        custom_btn.click(
            custom_voice,
            inputs=[custom_text, custom_voice_name, custom_instruction],
            outputs=custom_output,
        )

    with gr.Tab("🎨 Voice Design"):
        gr.Markdown("### Design a unique voice from text description")

        with gr.Row():
            with gr.Column():
                design_text = gr.Textbox(
                    label="Text to Synthesize", placeholder="Enter text...", lines=3
                )
                design_description = gr.Textbox(
                    label="Voice Description",
                    placeholder="A young female, cheerful, speaking clearly",
                    lines=3,
                )
                design_btn = gr.Button(
                    "🎵 Generate Speech", variant="primary", size="lg"
                )

                gr.Markdown("""
                **Description Tips:**
                - Age: young / middle-aged / elderly
                - Gender: male / female
                - Emotion: cheerful / serious / calm / excited
                - Style: clear / soft / authoritative / energetic
                
                **Examples:**
                - "A middle-aged male, deep and authoritative, speaking slowly"
                - "A young female, cheerful and bubbly, speaking energetically"
                - "An elderly man, warm and gentle, speaking softly"
                """)

            with gr.Column():
                design_output = gr.Audio(label="Generated Speech")
                gr.Markdown("**Expected Speed**: RTF 3.5-5x on RTX A5000")

        design_btn.click(
            voice_design,
            inputs=[design_text, design_description],
            outputs=design_output,
        )

    # Footer
    gr.Markdown("---")
    gr.Markdown("""
    **Note**: This interface uses the Qwen3-TTS 1.7B models with SDPA optimization.
    - First use requires model download (~2-3GB)
    - Subsequent uses load from local cache
    - GPU memory usage: ~4-6GB depending on model type
    """)

print("=" * 60)
print("🎙️ Qwen3-TTS Local Gradio UI")
print("💡 Ready for voice cloning and generation")
print("=" * 60)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False)
