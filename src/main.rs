use image::{imageops::FilterType, DynamicImage, GenericImageView};

use ort::{inputs, CUDAExecutionProvider, Session, SessionOutputs};

fn load_image(path: &str, width: u32, height: u32, filter: FilterType) -> (u32, u32, DynamicImage) {
    let original_img = image::open(path).unwrap();

    let (img_width, img_height) = (original_img.width(), original_img.height());

    let img = original_img.resize_exact(width, height, filter);
    (img_width, img_height, img)
}

fn main() -> ort::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 4 {
        let command = &args[0];
        println!(
            "usage {} depth_anything_v2_vitl.onnx input.jpg output.png",
            &command
        );
        return Ok(());
    }

    let model_path = &args[1];
    let input_path = &args[2];
    let output_path = &args[3];

    // let input_resize_type = FilterType::Nearest;
    // let input_resize_type = FilterType::CatmullRom;
    // let input_resize_type = FilterType::Gaussian;
    let input_resize_type = FilterType::Lanczos3;
    let output_resize_type = FilterType::Lanczos3;

    const MODEL_IMAGE_SIZE: u32 = 518;

    let save_debug_image = false;
    if input_path == output_path {
        println!("input and output is same");
        return Ok(());
    }

    ort::init()
        .with_execution_providers([CUDAExecutionProvider::default().build()])
        .commit()?;

    let (img_width, img_height, img) = load_image(
        input_path,
        MODEL_IMAGE_SIZE,
        MODEL_IMAGE_SIZE,
        input_resize_type,
    );

    if save_debug_image {
        let tmp_path = format!("{}_resize.png", input_path);
        img.save(tmp_path).unwrap();
    }

    let mut input =
        ndarray::Array::zeros((1, 3, MODEL_IMAGE_SIZE as usize, MODEL_IMAGE_SIZE as usize));

    // mean
    let r_m = 0.485_f32;
    let g_m = 0.456_f32;
    let b_m = 0.406_f32;
    // std dev
    let r_s = 0.229_f32;
    let g_s = 0.224_f32;
    let b_s = 0.225_f32;

    for pixel in img.pixels() {
        let x = pixel.0 as _;
        let y = pixel.1 as _;
        let [r, g, b, _] = pixel.2 .0;
        input[[0, 0, y, x]] = (r as f32 / 255.0_f32 - r_m) / r_s;
        input[[0, 1, y, x]] = (g as f32 / 255.0_f32 - g_m) / g_s;
        input[[0, 2, y, x]] = (b as f32 / 255.0_f32 - b_m) / b_s;
    }

    drop(img);

    let output_vec: Vec<f32>;

    {
        let model = Session::builder()?
            .with_optimization_level(ort::GraphOptimizationLevel::Level3)?
            .with_intra_threads(4)?
            .commit_from_file(model_path)?;

        // compute !!
        let outputs: SessionOutputs = model.run(inputs![input.view()]?)?;
        let output = &outputs[0];

        let output_tensor = output.try_extract_tensor::<f32>()?;
        output_vec = output_tensor.map(|x| *x).into_iter().collect();
    }
    drop(input);

    let zero = 0.0_f32;
    let min_value = output_vec
        .iter()
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater))
        .unwrap_or(&zero);
    let max_value = output_vec
        .iter()
        .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Greater))
        .unwrap_or(&zero);

    let output_dim = MODEL_IMAGE_SIZE;
    let output_img = image::ImageBuffer::from_fn(output_dim, output_dim, |x, y| {
        let i = (x + y * output_dim) as usize;
        let value = output_vec[i];
        let mut normalized_value = zero;
        if min_value != max_value {
            normalized_value = (value - min_value) / (max_value - min_value);
        }
        image::Luma([normalized_value])
    });

    drop(output_vec);

    if save_debug_image {
        let tmp_path = format!("{}_resize.png", output_path);

        let output_img_u16 = image::ImageBuffer::from_fn(output_dim, output_dim, |x, y| {
            let notmalized_value: f32 = output_img.get_pixel(x, y).0[0];
            let value = (notmalized_value * 65535.0_f32).round() as u16;
            image::Luma([value])
        });

        output_img_u16.save(tmp_path).unwrap();
    }

    // resize to original size
    let resized_output =
        image::imageops::resize(&output_img, img_width, img_height, output_resize_type);

    let resized_output_u16 = image::ImageBuffer::from_fn(img_width, img_height, |x, y| {
        let notmalized_value: f32 = resized_output.get_pixel(x, y).0[0];
        let value = (notmalized_value * 65535.0_f32).round() as u16;
        image::Luma([value])
    });

    resized_output_u16.save(output_path).unwrap();

    Ok(())
}
