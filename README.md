# depth-anything-v2-ort

This is a [ort](https://github.com/pykeio/ort) ( (Rust)[https://www.rust-lang.org/] binding for (ONNX Runtime)[https://onnx.ai/] ) implementation of (Depth-Anything-V2)[https://github.com/DepthAnything/Depth-Anything-V2], it depends on (Depth-Anything-Android)[https://github.com/shubham0204/Depth-Anything-Android] model.
 
## how to build and run

1. install rust
https://www.rust-lang.org/tools/install

2 clone the repository
    git clone xxxx
    cd depth-anything-v2-ort

3 download model
`wget https://github.com/shubham0204/Depth-Anything-Android/releases/download/model-v2/fused_model_uint8.onnx`

4 build
`cargo build --release`

5 run
`./target/release/depth-anything-v2-ort fused_model_uint8.onnx input.jpg output.png`



