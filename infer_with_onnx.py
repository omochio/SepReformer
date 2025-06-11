"""
Simple ONNX inference script for SepReformer audio source separation.
"""

import argparse
import numpy as np
import onnxruntime as ort
import soundfile as sf
import os
import torch
import librosa


def load_audio(file_path, target_sr=8000):
    """Load audio file using librosa, following _inference_sample approach."""
    # Use librosa.load like in _inference_sample
    mixture, sr = librosa.load(file_path, sr=target_sr)
    
    if sr != target_sr:
        print(f"Warning: Audio sample rate is {sr}Hz, expected {target_sr}Hz")
    
    return mixture, sr


def run_inference(onnx_model_path, input_audio, stride=4):
    """Run ONNX inference with dynamic sequence length, following _inference_sample approach."""
    print(f"Audio length: {len(input_audio)} samples")

    # Convert to tensor and add batch dimension
    mixture = torch.tensor(input_audio, dtype=torch.float32)[None]
    
    # Apply padding if needed (following _inference_sample logic)
    remains = mixture.shape[-1] % stride
    if remains != 0:
        padding = stride - remains
        mixture_padded = torch.nn.functional.pad(mixture, (0, padding), "constant", 0)
        print(f"Applied padding: {padding} samples")
    else:
        mixture_padded = mixture
    
    print(f"Input tensor shape: {mixture_padded.shape}")

    # Create ONNX runtime session
    session = ort.InferenceSession(onnx_model_path)
    
    # Run inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: mixture_padded.numpy()})
    
    # Trim outputs to original length (following _inference_sample approach)
    original_length = mixture.shape[-1]
    trimmed_outputs = []
    for output in outputs:
        if output.ndim >= 2:
            # Trim the last dimension to original length
            trimmed_output = output[..., :original_length]
        else:
            trimmed_output = output
        trimmed_outputs.append(trimmed_output)
    
    # Normalize outputs like in _inference_sample: 0.9*src/max(abs(src))
    normalized_outputs = []
    for output in trimmed_outputs:
        if output.ndim >= 2:
            # Normalize each speaker separately
            normalized_output = np.zeros_like(output)
            for i in range(output.shape[0]):  # Iterate over speakers
                audio_data = output[i]
                max_val = np.max(np.abs(audio_data))
                if max_val > 0:
                    normalized_output[i] = 0.9 * audio_data / max_val
                else:
                    normalized_output[i] = audio_data
            normalized_outputs.append(normalized_output)
        else:
            normalized_outputs.append(output)
    
    return normalized_outputs


def save_separated_audio(outputs, output_dir, base_name, sample_rate=8000):
    """Save separated audio outputs, following _inference_sample approach.
    
    Args:
        outputs: List of output tensors from the model (already trimmed and normalized)
        output_dir: Directory to save the outputs
        base_name: Base name for the output files
        sample_rate: Sample rate for the audio files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Number of outputs: {len(outputs)}")
    print(f"Output shapes: {[output.shape for output in outputs]}")
    
    print(f"Saving {len(outputs)} separated speakers")
    
    for i, separated_audio in enumerate(outputs):  # Iterate over each output
        # Extract audio data (remove batch dimension if present)
        if separated_audio.ndim > 1:
            audio_data = separated_audio.squeeze()  # Remove batch dimension
        else:
            audio_data = separated_audio
        
        output_path = os.path.join(output_dir, f"{base_name}_out_{i}.wav")
        sf.write(output_path, audio_data, sample_rate)
        print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="ONNX inference for SepReformer")
    parser.add_argument("--input", "-i", required=True, help="Input audio file path")
    parser.add_argument("--model", "-m", default="onnx/SepReformer_Tiny_Libri2Mix.onnx", 
                       help="ONNX model path")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: ONNX model not found at {args.model}")
        return
    
    # Load and preprocess audio
    print(f"Loading audio: {args.input}")
    audio, sr = load_audio(args.input)
    
    # Run inference
    print(f"Running inference with model: {args.model}")
    outputs = run_inference(args.model, audio)
    
    # Save input mixture (following _inference_sample approach)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    mixture_path = os.path.join(args.output, f"{base_name}_in.wav")
    os.makedirs(args.output, exist_ok=True)
    
    # Normalize mixture like in _inference_sample: 0.9*mixture/max(abs(mixture))
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        normalized_mixture = 0.9 * audio / max_val
    else:
        normalized_mixture = audio
    sf.write(mixture_path, normalized_mixture, sr)
    print(f"Saved input mixture: {mixture_path}")
    
    # Save separated audio results
    save_separated_audio(outputs, args.output, base_name, sr)
    
    print("Inference completed!")


if __name__ == "__main__":
    main()