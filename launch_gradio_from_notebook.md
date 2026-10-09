# How to Launch Gradio Interface from Qwen3_TTS_Local_Voice_Clone_A5000.ipynb

After you have successfully loaded the model in your notebook (after running the cell that loads `base_model`), add a new cell with the following code to launch the Gradio interface:

```python
# GRADIO INTERFACE FOR VOICE CLONING
# Add this cell AFTER the model loading cell in your notebook

import gradio as gr
import soundfile as sf
import numpy as np
import tempfile
import time
import gc
from pathlib import Path

# Global variable for caching voice clone prompts (recommended for performance)
voice_clone_prompt_cache = None

def voice_clone_from_notebook(text, reference_audio, ref_transcript, use_fast_mode):
    """
    Generate speech by cloning a reference voice using the pre-loaded base_model
    from the notebook.
    """
    global base_model, voice_clone_prompt_cache
    
    # Input validation
    if not text or not text.strip():
        return None, "Please provide text to synthesize"
    
    if reference_audio is None:
        return None, "Please record or upload reference audio"

    try:
        total_start = time.time()
        model = base_model  # Use the pre-loaded model from notebook
        if model is None:
            return None, "Model not available. Please run the model loading cell first."

        print(f"⏱️ Creating prompt...")
        prompt_start = time.time()

        # Use cached prompt if available and not forcing regeneration
        if voice_clone_prompt_cache is not None and not use_fast_mode:
            print("✅ Using cached voice clone prompt")
            prompt_items = voice_clone_prompt_cache
        else:
            if use_fast_mode or not ref_transcript:
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=reference_audio,
                    x_vector_only_mode=True
                )
            else:
                prompt_items = model.create_voice_clone_prompt(
                    ref_audio=reference_audio,
                    ref_text=ref_transcript,
                    x_vector_only_mode=False
                )
                # Cache the prompt for future use
                voice_clone_prompt_cache = prompt_items

        prompt_time = time.time() - prompt_start
        print(f"   Prompt: {prompt_time:.1f}s")

        print(f"⏱️ Generating audio...")
        gen_start = time.time()

        with torch.inference_mode():
            wavs, sr = model.generate_voice_clone(
                text=text,
                voice_clone_prompt=prompt_items
            )

        gen_time = time.time() - gen_start

        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        sf.write(temp_file.name, wavs[0], sr)

        total_time = time.time() - total_start
        audio_duration = len(wavs[0]) / sr
        rtf = gen_time / audio_duration if audio_duration > 0 else 0

        print(f"✅ Done! Total: {total_time:.1f}s | Gen: {gen_time:.1f}s | Audio: {audio_duration:.1f}s | RTF: {rtf:.2f}x")

        torch.cuda.empty_cache()
        gc.collect()

        status_msg = f"✅ Success! Generated {audio_duration:.1f}s of audio (RTF: {rtf:.2f}x)"
        return temp_file.name, status_msg

    except Exception as e:
        print(f"❌ Error in voice_clone: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, f"❌ Error: {str(e)}"

def clear_prompt_cache():
    """Clear the voice clone prompt cache"""
    global voice_clone_prompt_cache
    voice_clone_prompt_cache = None
    print("🗑️ Voice clone prompt cache cleared")
    return "✅ Cache cleared"

# CREATE AND LAUNCH THE GRADIO INTERFACE
with gr.Blocks(title="Qwen3-TTS Voice Cloning (from Notebook)") as demo:
    gr.Markdown("# 🎙️ Qwen3-TTS Voice Cloning")
    gr.Markdown("### Clone voices with recorded audio (using pre-loaded model from notebook)")
    
    with gr.Row():
        with gr.Column():
            clone_text = gr.Textbox(
                label="Text to Synthesize",
                placeholder="Enter text to convert to speech...",
                lines=3
            )
            clone_audio = gr.Audio(
                label="Record Reference Audio (3+ seconds)",
                type="filepath",
                sources=["microphone", "upload"]
            )
            clone_transcript = gr.Textbox(
                label="Transcript (Optional - improves quality)",
                placeholder="What's said in the audio...",
                lines=2
            )
            clone_fast_mode = gr.Checkbox(
                label="Fast Mode (skip transcript for quicker processing)",
                value=True
            )
            with gr.Row():
                clone_btn = gr.Button("🎵 Generate Speech", variant="primary", size="lg")
                clear_cache_btn = gr.Button("🗑️ Clear Prompt Cache", size="sm")
            
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
            - Voice clone prompt is cached for subsequent generations
            - First generation may still take time for prompt creation
            - Subsequent generations are much faster
            """)
        
        # Event handlers
        clone_btn.click(
            fn=voice_clone_from_notebook,
            inputs=[clone_text, clone_audio, clone_transcript, clone_fast_mode],
            outputs=[clone_output, status_output]
        )
        
        clear_cache_btn.click(
            fn=clear_prompt_cache,
            outputs=status_output
        )

# LAUNCH THE INTERFACE
print("🚀 Launching Gradio interface...")
print("📝 Note: This uses the pre-loaded base_model from your notebook")
demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,  # Set to True if you want a public link
    debug=False,
    prevent_thread_lock=False  # Set to True if you want to keep notebook running after interface starts
)
```

## Instructions:

1. **Run all cells in your notebook up to and including the model loading cell** (the one that creates `base_model`)

2. **Add a new cell** after the model loading cell

3. **Paste the above code** into that new cell

4. **Run the new cell** - this will launch the Gradio interface

5. **Access the interface** at the URL shown in the output (typically http://localhost:7860 or a public link if you set share=True)

## Features of this interface:
- 🎤 **Record audio** using your microphone or upload a file
- 📄 **Add transcription** (optional but recommended for better quality)
- 🔤 **Convert text to voice** using your cloned voice
- ⚡ **Fast mode** option for quicker processing
- 💾 **Prompt caching** for faster subsequent generations
- 🚀 **Uses pre-loaded model** from notebook (no reload delay)

## Expected Performance:
- **First generation**: Model is already loaded, so just prompt creation + generation (~10-30 seconds)
- **Subsequent generations**: Very fast with cached prompt (~5-15 seconds)
- **Audio quality**: Depends on your reference audio quality and transcript accuracy

## Troubleshooting:
- If you see "Model not available", make sure you've run the model loading cell first
- Ignore SoX warnings if audio generation works correctly
- For best results: use clear audio with 5-10 seconds duration and provide an accurate transcript