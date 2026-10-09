"""
Code to add to Qwen3_TTS_Local_Voice_Clone_A5000.ipynb to launch Gradio interface
after the model has been loaded in the notebook.
"""

import gradio as gr
import soundfile as sf
import numpy as np
import tempfile
import time
import gc
from pathlib import Path

# Assuming these are already defined in the notebook:
# base_model (the loaded Qwen3TTSModel)
# OUTPUT_DIR (Path object for output directory)


def voice_clone_from_notebook(text, reference_audio, ref_transcript, use_fast_mode):
    """
    Generate speech by cloning a reference voice using the pre-loaded base_model
    from the notebook.
    """
    global base_model  # Assuming base_model is available from notebook

    if not text or not reference_audio:
        return None, "Please provide both text and reference audio"

    try:
        total_start = time.time()
        model = base_model  # Use the pre-loaded model from notebook
        if model is None:
            return None, "Model not available"

        print(f"⏱️ Creating prompt...")
        prompt_start = time.time()

        # Use cached prompt if available and not forcing regeneration
        # For simplicity in notebook context, we'll create fresh each time
        # but you could add caching if desired
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

        status_msg = (
            f"✅ Success! Generated {audio_duration:.1f}s of audio (RTF: {rtf:.2f}x)"
        )
        return temp_file.name, status_msg

    except Exception as e:
        print(f"❌ Error in voice_clone: {str(e)}")
        import traceback

        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"


# Create and launch the Gradio interface
with gr.Blocks(title="Qwen3-TTS Voice Cloning from Notebook") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Voice Cloning")
    gr.Markdown(
        "### Clone voices with recorded audio (using pre-loaded model from notebook)"
    )

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
                label="Fast Mode (skip transcript for quicker processing)", value=True
            )
            with gr.Row():
                clone_btn = gr.Button(
                    "🎵 Generate Speech", variant="primary", size="lg"
                )

            status_output = gr.Textbox(label="Status", value="Ready", interactive=False)

        with gr.Column():
            clone_output = gr.Audio(label="Generated Speech")
            gr.Markdown("""
            **Instructions:**
            1. Record or upload 3+ seconds of clear audio
            2. Provide transcript for better quality (optional)
            3. Enter text to synthesize
            4. Click Generate Speech
            
            **Notes:**
            - Uses pre-loaded model from notebook (faster startup)
            - First generation may still take time for prompt creation
            - Subsequent generations are faster
            """)

        clone_btn.click(
            fn=voice_clone_from_notebook,
            inputs=[clone_text, clone_audio, clone_transcript, clone_fast_mode],
            outputs=[clone_output, status_output],
        )

# Launch the interface
print("🚀 Launching Gradio interface...")
demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False)
