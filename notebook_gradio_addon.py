# COPY AND PASTE THIS ENTIRE BLOCK INTO A NEW CELL IN YOUR NOTEBOOK
# AFTER THE CELL THAT LOADS base_model AND CREATES voice_clone_prompt

import gradio as gr
import soundfile as sf
import numpy as np
import tempfile
import time
import gc
from pathlib import Path
import torch

# Global cache for voice clone prompt (optional but improves performance)
# Initialize as None - will be set when user first provides audio/transcript
cached_voice_clone_prompt = None
cached_audio_path = None
cached_transcript = None


def create_voice_clone_interface():
    """Create and launch Gradio interface for voice cloning using notebook's base_model"""

    def process_audio_and_generate(
        text_input, audio_input, transcript_input, use_fast_mode
    ):
        """
        Process the audio input and generate cloned speech
        Uses the pre-loaded base_model from the notebook
        """
        global cached_voice_clone_prompt, cached_audio_path, cached_transcript

        # Validate inputs
        if not text_input or not text_input.strip():
            return None, "❌ Please enter text to synthesize"

        if audio_input is None:
            return None, "❌ Please record or upload reference audio"

        try:
            start_time = time.time()
            print(f"🔄 Starting voice cloning process...")

            # Use the pre-loaded base_model from notebook
            model = base_model  # This comes from the notebook's execution context

            # Determine if we need to create a new prompt or can use cached one
            audio_changed = cached_audio_path != audio_input
            transcript_changed = cached_transcript != transcript_input

            if (
                not use_fast_mode
                and not audio_changed
                and not transcript_changed
                and cached_voice_clone_prompt is not None
            ):
                # Use cached prompt
                print("✅ Using cached voice clone prompt")
                voice_clone_prompt = cached_voice_clone_prompt
            else:
                # Create new prompt
                print("🔄 Creating voice clone prompt...")
                prompt_start = time.time()

                if use_fast_mode or not transcript_input:
                    # Fast mode - no transcript
                    voice_clone_prompt = model.create_voice_clone_prompt(
                        ref_audio=audio_input, x_vector_only_mode=True
                    )
                else:
                    # Full mode - with transcript
                    voice_clone_prompt = model.create_voice_clone_prompt(
                        ref_audio=audio_input,
                        ref_text=transcript_input,
                        x_vector_only_mode=False,
                    )

                prompt_time = time.time() - prompt_start
                print(f"   Prompt creation time: {prompt_time:.1f}s")

                # Cache the prompt and associated data
                cached_voice_clone_prompt = voice_clone_prompt
                cached_audio_path = audio_input
                cached_transcript = transcript_input

            print("🔊 Generating audio...")
            gen_start = time.time()

            # Generate audio using the same approach as in the notebook
            with torch.inference_mode():
                wavs, sr = model.generate_voice_clone(
                    text=text_input, voice_clone_prompt=voice_clone_prompt
                )

            gen_time = time.time() - gen_start
            print(f"   Generation time: {gen_time:.1f}s")

            # Save to temporary file for Gradio output
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
            sf.write(temp_file.name, wavs[0], sr)

            # Calculate metrics
            audio_duration = len(wavs[0]) / sr
            total_time = time.time() - start_time
            rtf = gen_time / audio_duration if audio_duration > 0 else 0

            print(
                f"✅ Generated {audio_duration:.1f}s of audio in {total_time:.1f}s total"
            )
            print(f"   Generation RTF: {rtf:.2f}x")

            # Clean up GPU memory
            torch.cuda.empty_cache()
            gc.collect()

            status_msg = (
                f"✅ Success! Generated {audio_duration:.1f}s of audio\n"
                f"   Generation RTF: {rtf:.2f}x | Total: {total_time:.1f}s"
            )

            return temp_file.name, status_msg

        except Exception as e:
            print(f"❌ Error in voice cloning: {e}")
            import traceback

            traceback.print_exc()
            return None, f"❌ Error: {str(e)}"

    def clear_cache():
        """Clear the voice clone prompt cache"""
        global cached_voice_clone_prompt, cached_audio_path, cached_transcript
        cached_voice_clone_prompt = None
        cached_audio_path = None
        cached_transcript = None
        print("🗑️ Voice clone prompt cache cleared")
        return "✅ Cache cleared"

    # Create the Gradio interface
    with gr.Blocks(title="Qwen3-TTS Voice Cloning (from Notebook)") as demo:
        gr.Markdown("# 🎙️ Qwen3-TTS Voice Cloning")
        gr.Markdown(
            "### Clone voices with recorded audio - Using pre-loaded model from notebook"
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
                    label="🚀 Generate Cloned Speech", variant="primary", size="lg"
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

        # Event handlers
        generate_btn.click(
            fn=process_audio_and_generate,
            inputs=[text_input, audio_input, transcript_input, fast_mode],
            outputs=[audio_output, status_output],
        )

        clear_btn.click(fn=clear_cache, outputs=status_output)

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
            - Since the model is already loaded from your notebook, startup is fast
            - Voice clone prompt is cached for subsequent generations
            - Ignore **SoX warnings** if audio generation works correctly
            - For voice cloning: provide transcript for better accuracy
            
            ### Performance (with pre-loaded model):
            - **First generation**: ~10-30 seconds (prompt creation + generation)
            - **Subsequent generations**: ~5-15 seconds (with cached prompt)
            - **RTF (Real-Time Factor)**: 3.5-5x (generation time vs audio duration)
            """)

    return demo


# LAUNCH THE INTERFACE
print("🚀 Creating Gradio interface for Qwen3-TTS voice cloning...")
print("📝 This interface uses the pre-loaded base_model from your notebook")
demo = create_voice_clone_interface()

print("🚀 Launching Gradio interface...")
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    debug=False,
    prevent_thread_lock=False,  # Set to True if you want to keep notebook running after interface starts
)
