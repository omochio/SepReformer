"""
Analyze ONNX model outputs to understand which outputs are the main separated sources.
"""

import argparse
import numpy as np
import onnxruntime as ort
import soundfile as sf
import os


def analyze_outputs(model_path, input_audio_path):
    """Analyze all outputs from the ONNX model."""
    
    # Load input audio
    audio, sr = sf.read(input_audio_path, dtype='float32')
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    
    # Run inference
    session = ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    
    if len(audio.shape) == 1:
        audio = audio.reshape(1, -1)
    
    outputs = session.run(None, {input_name: audio})
    
    print(f"=== Output Analysis ===")
    print(f"Number of outputs: {len(outputs)}")
    print(f"Input audio RMS: {np.sqrt(np.mean(audio**2)):.6f}")
    print()
    
    for i, output in enumerate(outputs):
        output_squeezed = output.squeeze()
        rms = np.sqrt(np.mean(output_squeezed**2))
        max_val = np.max(np.abs(output_squeezed))
        energy = np.sum(output_squeezed**2)
        
        print(f"Output {i}:")
        print(f"  Shape: {output.shape}")
        print(f"  RMS: {rms:.6f}")
        print(f"  Max absolute value: {max_val:.6f}")
        print(f"  Total energy: {energy:.6f}")
        print(f"  SNR vs input: {20*np.log10(rms / np.sqrt(np.mean(audio**2))):.2f} dB")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze ONNX model outputs")
    parser.add_argument("--input", "-i", required=True, help="Input audio file path")
    parser.add_argument("--model", "-m", default="onnx/SepReformer_Tiny_Libri2Mix.onnx", 
                       help="ONNX model path")
    args = parser.parse_args()
    
    analyze_outputs(args.model, args.input)


if __name__ == "__main__":
    main()
