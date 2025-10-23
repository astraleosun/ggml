#!/usr/bin/env python3
import sys
import os
import torch
import gguf
import numpy as np

# Add yolov5 to the Python path
sys.path.insert(0, '/home/jay/yolov5')

def save_conv_layer(gguf_writer, state_dict, prefix, layer_name, inp_c, out_c, ksize, has_bn=True):
    """Save a convolutional layer with optional batch normalization"""
    # Conv weights: [out_c, inp_c, ksize, ksize]
    conv_weight_key = f"{layer_name}.conv.weight"
    if conv_weight_key in state_dict:
        conv_weights = state_dict[conv_weight_key].numpy()
        # Convert to f16 as ggml doesn't support f32 convolution yet
        conv_weights = conv_weights.astype(np.float16)
        gguf_writer.add_tensor(f"{prefix}_weights", conv_weights, raw_shape=(out_c, inp_c, ksize, ksize))
    
    # Conv bias: [out_c] - Note: Conv layers in YOLOv5 typically don't have bias when using BatchNorm
    conv_bias_key = f"{layer_name}.conv.bias"
    if conv_bias_key in state_dict:
        conv_bias = state_dict[conv_bias_key].numpy()
        gguf_writer.add_tensor(f"{prefix}_biases", conv_bias, raw_shape=(1, out_c, 1, 1))
    
    if has_bn:
        # Batch norm parameters
        bn_weight_key = f"{layer_name}.bn.weight"
        bn_bias_key = f"{layer_name}.bn.bias"
        bn_running_mean_key = f"{layer_name}.bn.running_mean"
        bn_running_var_key = f"{layer_name}.bn.running_var"
        
        if bn_weight_key in state_dict:
            bn_scales = state_dict[bn_weight_key].numpy()
            gguf_writer.add_tensor(f"{prefix}_scales", bn_scales, raw_shape=(1, out_c, 1, 1))
        
        if bn_bias_key in state_dict:
            bn_biases = state_dict[bn_bias_key].numpy()
            gguf_writer.add_tensor(f"{prefix}_bn_biases", bn_biases, raw_shape=(1, out_c, 1, 1))
        
        if bn_running_mean_key in state_dict:
            bn_running_mean = state_dict[bn_running_mean_key].numpy()
            gguf_writer.add_tensor(f"{prefix}_rolling_mean", bn_running_mean, raw_shape=(1, out_c, 1, 1))
        
        if bn_running_var_key in state_dict:
            bn_running_var = state_dict[bn_running_var_key].numpy()
            gguf_writer.add_tensor(f"{prefix}_rolling_variance", bn_running_var, raw_shape=(1, out_c, 1, 1))

def save_detection_layer(gguf_writer, state_dict, detect_prefix, conv_idx, inp_c, out_c, ksize):
    """Save detection layer (no batch norm)"""
    conv_weight_key = f"{detect_prefix}.m.{conv_idx}.weight"
    conv_bias_key = f"{detect_prefix}.m.{conv_idx}.bias"
    
    if conv_weight_key in state_dict:
        conv_weights = state_dict[conv_weight_key].numpy()
        conv_weights = conv_weights.astype(np.float16)
        gguf_writer.add_tensor(f"detect_{conv_idx}_weights", conv_weights, raw_shape=(out_c, inp_c, ksize, ksize))
    
    if conv_bias_key in state_dict:
        conv_bias = state_dict[conv_bias_key].numpy()
        gguf_writer.add_tensor(f"detect_{conv_idx}_biases", conv_bias, raw_shape=(1, out_c, 1, 1))

def main():
    if len(sys.argv) != 2:
        print("Usage: %s <yolov5s.pt>" % sys.argv[0])
        sys.exit(1)
    
    input_file = sys.argv[1]
    outfile = 'yolov5s.gguf'
    gguf_writer = gguf.GGUFWriter(outfile, 'yolov5s')
    
    # Load the YOLOv5s model using yolov5's custom loader
    print(f"Loading {input_file}...")
    
    # Import yolov5 modules after adding to path
    from models.experimental import attempt_load
    
    # Load the checkpoint directly with weights_only=False
    checkpoint = torch.load(input_file, map_location='cpu', weights_only=False)
    
    # Extract model state dict
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model_state = checkpoint['model'].state_dict()
        else:
            model_state = checkpoint
    else:
        # If it's a model object directly
        model_state = checkpoint.state_dict()
    
    # Print all keys for debugging (optional, can be commented out for production)
    print("Model keys:")
    for key in sorted(model_state.keys()):
        print(f"  {key}: {model_state[key].shape}")
    
    # YOLOv5s network structure based on yolov5s.yaml
    # The model layers are stored as model.0, model.1, etc.
    
    # Backbone layers
    # Layer 0: Conv (3->32, 6x6, stride=2)
    save_conv_layer(gguf_writer, model_state, "l0", "model.0", 3, 32, 6, has_bn=True)
    
    # Layer 1: Conv (32->64, 3x3, stride=2)  
    save_conv_layer(gguf_writer, model_state, "l1", "model.1", 32, 64, 3, has_bn=True)
    
    # Layer 2: C3 (64->64, n=3)
    # C3 has two main convs (cv1, cv2) and bottleneck layers (m)
    save_conv_layer(gguf_writer, model_state, "l2_cv1", "model.2.cv1", 64, 32, 1, has_bn=True)
    save_conv_layer(gguf_writer, model_state, "l2_cv2", "model.2.cv2", 64, 32, 1, has_bn=True)
    
    # C3 bottleneck layers (3 layers for the first C3)
    for i in range(3):
        save_conv_layer(gguf_writer, model_state, f"l2_m{i}_cv1", f"model.2.m.{i}.cv1", 32, 32, 1, has_bn=True)
        save_conv_layer(gguf_writer, model_state, f"l2_m{i}_cv2", f"model.2.m.{i}.cv2", 32, 32, 3, has_bn=True)
    
    # Layer 3: Conv (64->128, 3x3, stride=2)
    save_conv_layer(gguf_writer, model_state, "l3", "model.3", 64, 128, 3, has_bn=True)
    
    # Layer 4: C3 (128->128, n=6)
    save_conv_layer(gguf_writer, model_state, "l4_cv1", "model.4.cv1", 128, 64, 1, has_bn=True)
    save_conv_layer(gguf_writer, model_state, "l4_cv2", "model.4.cv2", 128, 64, 1, has_bn=True)
    
    for i in range(6):
        save_conv_layer(gguf_writer, model_state, f"l4_m{i}_cv1", f"model.4.m.{i}.cv1", 64, 64, 1, has_bn=True)
        save_conv_layer(gguf_writer, model_state, f"l4_m{i}_cv2", f"model.4.m.{i}.cv2", 64, 64, 3, has_bn=True)
    
    # Layer 5: Conv (128->256, 3x3, stride=2)
    save_conv_layer(gguf_writer, model_state, "l5", "model.5", 128, 256, 3, has_bn=True)
    
    # Layer 6: C3 (256->256, n=9)
    save_conv_layer(gguf_writer, model_state, "l6_cv1", "model.6.cv1", 256, 128, 1, has_bn=True)
    save_conv_layer(gguf_writer, model_state, "l6_cv2", "model.6.cv2", 256, 128, 1, has_bn=True)
    
    for i in range(9):
        save_conv_layer(gguf_writer, model_state, f"l6_m{i}_cv1", f"model.6.m.{i}.cv1", 128, 128, 1, has_bn=True)
        save_conv_layer(gguf_writer, model_state, f"l6_m{i}_cv2", f"model.6.m.{i}.cv2", 128, 128, 3, has_bn=True)
    
    # Layer 7: Conv (256->512, 3x3, stride=2)
    save_conv_layer(gguf_writer, model_state, "l7", "model.7", 256, 512, 3, has_bn=True)
    
    # Layer 8: C3 (512->512, n=3)
    save_conv_layer(gguf_writer, model_state, "l8_cv1", "model.8.cv1", 512, 256, 1, has_bn=True)
    save_conv_layer(gguf_writer, model_state, "l8_cv2", "model.8.cv2", 512, 256, 1, has_bn=True)
    
    for i in range(3):
        save_conv_layer(gguf_writer, model_state, f"l8_m{i}_cv1", f"model.8.m.{i}.cv1", 256, 256, 1, has_bn=True)
        save_conv_layer(gguf_writer, model_state, f"l8_m{i}_cv2", f"model.8.m.{i}.cv2", 256, 256, 3, has_bn=True)
    
    # Layer 9: SPPF (512->512, k=5)
    save_conv_layer(gguf_writer, model_state, "l9_cv1", "model.9.cv1", 512, 256, 1, has_bn=True)
    save_conv_layer(gguf_writer, model_state, "l9_cv2", "model.9.cv2", 1024, 512, 1, has_bn=True)
    
    # Head layers
    # Layer 10: Conv (512->256, 1x1)
    save_conv_layer(gguf_writer, model_state, "l10", "model.10", 512, 256, 1, has_bn=True)
    
    # Layer 13: C3 (512->256, n=3) - after first upsample and concat
    save_conv_layer(gguf_writer, model_state, "l13_cv1", "model.13.cv1", 512, 256, 1, has_bn=True)
    save_conv_layer(gguf_writer, model_state, "l13_cv2", "model.13.cv2", 512, 256, 1, has_bn=True)
    
    for i in range(3):
        save_conv_layer(gguf_writer, model_state, f"l13_m{i}_cv1", f"model.13.m.{i}.cv1", 256, 256, 1, has_bn=True)
        save_conv_layer(gguf_writer, model_state, f"l13_m{i}_cv2", f"model.13.m.{i}.cv2", 256, 256, 3, has_bn=True)
    
    # Layer 14: Conv (256->128, 1x1)
    save_conv_layer(gguf_writer, model_state, "l14", "model.14", 256, 128, 1, has_bn=True)
    
    # Layer 17: C3 (256->128, n=3) - P3/8 small detection head
    save_conv_layer(gguf_writer, model_state, "l17_cv1", "model.17.cv1", 256, 128, 1, has_bn=True)
    save_conv_layer(gguf_writer, model_state, "l17_cv2", "model.17.cv2", 256, 128, 1, has_bn=True)
    
    for i in range(3):
        save_conv_layer(gguf_writer, model_state, f"l17_m{i}_cv1", f"model.17.m.{i}.cv1", 128, 128, 1, has_bn=True)
        save_conv_layer(gguf_writer, model_state, f"l17_m{i}_cv2", f"model.17.m.{i}.cv2", 128, 128, 3, has_bn=True)
    
    # Layer 18: Conv (128->128, 3x3, stride=2)
    save_conv_layer(gguf_writer, model_state, "l18", "model.18", 128, 128, 3, has_bn=True)
    
    # Layer 20: C3 (512->256, n=3) - P4/16 medium detection head
    save_conv_layer(gguf_writer, model_state, "l20_cv1", "model.20.cv1", 512, 256, 1, has_bn=True)
    save_conv_layer(gguf_writer, model_state, "l20_cv2", "model.20.cv2", 512, 256, 1, has_bn=True)
    
    for i in range(3):
        save_conv_layer(gguf_writer, model_state, f"l20_m{i}_cv1", f"model.20.m.{i}.cv1", 256, 256, 1, has_bn=True)
        save_conv_layer(gguf_writer, model_state, f"l20_m{i}_cv2", f"model.20.m.{i}.cv2", 256, 256, 3, has_bn=True)
    
    # Layer 21: Conv (256->256, 3x3, stride=2)
    save_conv_layer(gguf_writer, model_state, "l21", "model.21", 256, 256, 3, has_bn=True)
    
    # Layer 23: C3 (1024->512, n=3) - P5/32 large detection head
    save_conv_layer(gguf_writer, model_state, "l23_cv1", "model.23.cv1", 1024, 512, 1, has_bn=True)
    save_conv_layer(gguf_writer, model_state, "l23_cv2", "model.23.cv2", 1024, 512, 1, has_bn=True)
    
    for i in range(3):
        save_conv_layer(gguf_writer, model_state, f"l23_m{i}_cv1", f"model.23.m.{i}.cv1", 512, 512, 1, has_bn=True)
        save_conv_layer(gguf_writer, model_state, f"l23_m{i}_cv2", f"model.23.m.{i}.cv2", 512, 512, 3, has_bn=True)
    
    # Detection layers (Detect module - model.24)
    # The Detect module has 3 output convolutions for the 3 scales
    # Each outputs (nc + 5) * na = (80 + 5) * 3 = 255 channels
    save_detection_layer(gguf_writer, model_state, "model.24", 0, 128, 255, 1)  # P3/8 detection
    save_detection_layer(gguf_writer, model_state, "model.24", 1, 256, 255, 1)  # P4/16 detection  
    save_detection_layer(gguf_writer, model_state, "model.24", 2, 512, 255, 1)  # P5/32 detection
    
    # Write the GGUF file
    gguf_writer.write_header_to_file()
    gguf_writer.write_kv_data_to_file()
    gguf_writer.write_tensors_to_file()
    gguf_writer.close()
    
    print(f"{input_file} converted to {outfile}")

if __name__ == '__main__':
    main()
