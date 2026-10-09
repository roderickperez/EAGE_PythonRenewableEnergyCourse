#!/usr/bin/env python3
"""
Test interface to verify Gradio works without model loading
"""

import warnings

warnings.filterwarnings("ignore")

import gradio as gr


def dummy_function(text, audio):
    return "Test successful!"


with gr.Blocks(title="Test Interface") as demo:
    gr.Markdown("# 🎙️ Test Interface")
    gr.Markdown("If you see this, Gradio is working correctly.")

    with gr.Row():
        text_input = gr.Textbox(label="Text Input")
        audio_input = gr.Audio(label="Audio Input", type="filepath")
        output = gr.Textbox(label="Output")

        btn = gr.Button("Test")
        btn.click(dummy_function, inputs=[text_input, audio_input], outputs=output)

print("=" * 50)
print("🧪 Test Interface Ready")
print("💡 Run: python test_interface.py")
print("=" * 50)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, debug=False)
