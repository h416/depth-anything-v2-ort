# depth-anything-v2-ort

This is a [ort](https://github.com/pykeio/ort) ( Rust binding for ONNX Runtime ) implementation of [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2), it depends on [Depth-Anything-ONNX](https://github.com/fabio-sim/Depth-Anything-ONNX) model.
 
## how to build and run

### install rust
https://www.rust-lang.org/tools/install

### clone the repository

    git clone https://github.com/h416/depth-anything-v2-ort.git
    cd depth-anything-v2-ort

### download model

    wget https://github.com/fabio-sim/Depth-Anything-ONNX/releases/download/v2.0.0/depth_anything_v2_vitl.onnx

### build

    cargo build --release

### run

    ./target/release/depth-anything-v2-ort depth_anything_v2_vitl.onnx input.jpg output.png
