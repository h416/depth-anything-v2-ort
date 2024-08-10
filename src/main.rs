use image::{imageops::FilterType, GenericImageView};
use ort::{inputs, CUDAExecutionProvider, Session, SessionOutputs};

fn main() -> ort::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        let command = &args[0];
        println!(
            "usage {} fused_model_uint8.onnx input.jpg output.png",
            &command
        );
        return Ok(());
    }

    let model_path = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];
    if input_path == output_path {
        println!("input and output is same");
        return Ok(());
    }

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let original_img = image::open(input_path).unwrap();

    let (img_width, img_height) = (original_img.width(), original_img.height());
    let img = original_img.resize_exact(512, 512, FilterType::CatmullRom);
    let mut input = ndarray::Array::zeros((1, 512, 512, 3));
    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, y, x, 0]] = r;
        input[[0, y, x, 1]] = g;
        input[[0, y, x, 2]] = b;
    }

    let model = Session::builder()?
        .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;

    let outputs: SessionOutputs = model.run(inputs![input.view()]?)?;
    let output = &outputs[0];

    let output_dim = 504;
    let output_tensor = output.try_extract_tensor::<u8>()?;

    let output_vec: Vec<u8> = output_tensor.map(|x| *x).into_iter().collect();
    let output_img = image::ImageBuffer::from_fn(output_dim, output_dim, |x, y| {
        let i = (x + y * output_dim) as usize;
        let value = output_vec[i];
        image::Luma([value])
    });

    let resized_output =
        image::imageops::resize(&output_img, img_width, img_height, FilterType::Triangle);
    resized_output.save(output_path).unwrap();

    Ok(())
}
