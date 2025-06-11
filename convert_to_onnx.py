import argparse
import importlib
import os
from utils import util_system
from utils.decorators import *
import torch
import onnx
import onnxruntime as ort

class OnnxWrapperModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.baseModel = model

    def forward(self, x):
        return self.baseModel(x)[0]  # Don't return aux
    
    def load_state_dict(self, state_dict, strict=True):
        # Override to handle state_dict loading
        self.baseModel.load_state_dict(state_dict, strict=strict)
    
# Parse args
parser = argparse.ArgumentParser(
    description="Command to convert trained model to ONNX format")
parser.add_argument(
    "--model",
    type=str,
    default="SepReformer_Tiny_Libri2Mix",
    dest="model",
    help="Insert model name")
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    dest="checkpoint",
    help="Path to specific checkpoint file (optional)")
args = parser.parse_args()

# Load model module
model_module = importlib.import_module(f"models.{args.model}.model")

# Setup config from YAML
yaml_path = os.path.join(os.path.dirname(os.path.abspath(model_module.__file__)), "configs.yaml")
yaml_dict = util_system.parse_yaml(yaml_path)
config = yaml_dict["config"]

# Initialize model with configuration
model = OnnxWrapperModel(model_module.Model(**config["model"]))

# Load trained weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = os.path.dirname(os.path.abspath(model_module.__file__))
loaded_epoch = None

if args.checkpoint:
    # Load specific checkpoint file
    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        loaded_epoch = checkpoint.get('epoch', None)
    else:
        raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
else:
    # Auto-load the latest checkpoint
    pretrain_weights_path = os.path.join(model_dir, "log", "pretrain_weights")
    scratch_weights_path = os.path.join(model_dir, "log", "scratch_weights")
    
    # Check for pretrained weights first, then scratch weights
    checkpoint_path = None
    if os.path.exists(pretrain_weights_path) and any(file.endswith(('.pth', '.pt', '.pkl')) for file in os.listdir(pretrain_weights_path)):
        checkpoint_path = pretrain_weights_path
    elif os.path.exists(scratch_weights_path) and any(file.endswith(('.pth', '.pt', '.pkl')) for file in os.listdir(scratch_weights_path)):
        checkpoint_path = scratch_weights_path
    
    if checkpoint_path:
        # Load the latest checkpoint using the same logic as in engine.py
        try:
            from utils import util_engine
            
            # Create a dummy optimizer for the loading function
            dummy_optimizer = torch.optim.Adam(model.parameters())
            epoch = util_engine.load_last_checkpoint_n_get_epoch(checkpoint_path, model, dummy_optimizer, location=device)
            loaded_epoch = epoch - 1
            print(f"✓ Loaded trained model from {checkpoint_path} (epoch {loaded_epoch})")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            print("⚠️  Using randomly initialized model instead")
    else:
        print("⚠️  No trained weights found. Using randomly initialized model.")
        print(f"   Searched in: {pretrain_weights_path}")
        print(f"   Searched in: {scratch_weights_path}")

# Convert to ONNX
model.eval()
# Use a more representative input size based on the model's max_len configuration
max_len = config["dataset"]["max_len"]
dummy_input = torch.randn(1, max_len)

# Try to export with opset version 11 which has better support for dynamic shapes
os.makedirs("onnx", exist_ok=True)
# Add epoch number to filename if model was loaded from checkpoint
if loaded_epoch is not None:
    output_onnx_file = f"onnx/{args.model}_epoch{loaded_epoch:03d}.onnx"
else:
    output_onnx_file = f"onnx/{args.model}.onnx"
torch.onnx.export(
    model,
    dummy_input,
    output_onnx_file,
    input_names=["input"],
    output_names=["talker_0", "talker_1"],
    dynamic_axes= {
        "input": {1: "sequence_length"},
        "talker_0": {1: "sequence_length"}, 
        "talker_1": {1: "sequence_length"},
    },
    dynamo=True,
)

print(f"ONNX file saved to: {output_onnx_file}")

# Load and analyze the exported ONNX model
print("\n=== ONNX Model Information ===")
try:
    # Load ONNX model
    onnx_model = onnx.load(output_onnx_file)
    
    # Check if model is valid
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validation: PASSED")
    
    # Get model info
    print(f"ONNX IR version: {onnx_model.ir_version}")
    print(f"Producer name: {onnx_model.producer_name}")
    print(f"Producer version: {onnx_model.producer_version}")
    
    # Get input/output info
    print("\n--- Input/Output Information ---")
    for input_info in onnx_model.graph.input:
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})" for dim in input_info.type.tensor_type.shape.dim]
        print(f"Input '{input_info.name}': shape={shape}, dtype={onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[input_info.type.tensor_type.elem_type]}")
    
    for output_info in onnx_model.graph.output:
        shape = [dim.dim_value if dim.dim_value > 0 else f"dynamic({dim.dim_param})" for dim in output_info.type.tensor_type.shape.dim]
        print(f"Output '{output_info.name}': shape={shape}, dtype={onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[output_info.type.tensor_type.elem_type]}")
    
    # Test ONNX Runtime inference
    print("\n--- ONNX Runtime Test ---")
    ort_session = ort.InferenceSession(output_onnx_file)
    
    # Get session info
    print(f"ONNX Runtime providers: {ort_session.get_providers()}")
    
    # Run inference with dummy input
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    
    print(f"ONNX Runtime output shape: {ort_outputs[0].shape}")
    print(f"ONNX Runtime output dtype: {ort_outputs[0].dtype}")
    
except Exception as e:
    print(f"❌ Error analyzing ONNX model: {e}")
    import traceback
    traceback.print_exc()