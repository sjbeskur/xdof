use image::io::Reader as ImageReader;
use std::time::SystemTime;

use image::GrayImage;
use image::ImageBuffer;
use image::Luma;

use xdof::common::*;
use xdof::descriptors;
use xdof::essential;
use xdof::fast_detect;
use xdof::image_impl;
use xdof::matcher;
use xdof::rand::Rand;
use xdof::Slam;

pub fn read_image(filename: &str) -> GrayImage {
    let img = ImageReader::open("sjb-aerial.png")
        .unwrap()
        .decode()
        .unwrap();

    let gimg =
        image::GrayImage::from_raw(img.width(), img.height(), img.as_bytes().to_vec()).unwrap();

    gimg
}

pub fn blur_image(image: &GrayImage, blur_radius: f32) -> GrayImage {
    let bytes = image_impl::greyscale_gaussian_blur(
        &image.as_raw(),
        image.width() as usize,
        image.height() as usize,
        blur_radius,
    );

    let blurred = image::GrayImage::from_raw(image.width(), image.height(), bytes).unwrap();
    blurred
}

fn main() {
    let start = SystemTime::now();
    let img_a = read_image("coffee.png");
    let img_b = read_image("coffee.png");
    let img_a = blur_image(&img_a, 3.0);
    let img_b = blur_image(&img_b, 3.0);

    // let mut slam = Slam::new(img_a, img_b);
    // let rslt = slam.calculate_pose();
    // println!("{:?}", rslt);
    let (kpswo_a, descriptors_a) = compute_kepoint_descriptors(img_a);
    let (kpswo_b, descriptors_b) = compute_kepoint_descriptors(img_b);

    let max_hamming_distance = 300;
    // PHASE 4  -  Match features between the two images
    let matched_keypoints = matcher::match_features(
        &kpswo_a,
        &descriptors_a,
        &kpswo_b,
        &descriptors_b,
        max_hamming_distance,
    );

    let seed = 2523523;
    let mut random = Rand::new_with_seed(seed);
    // // PHASE 5  -  RANSAC to find the best rotation and translation using 8 point algorithm
    let essential_matrix =
        essential::estimate_essential_ransac(&matched_keypoints, 1000, 10.0, &mut random).unwrap();

    let (rotation, tranlation) = essential::decompose_essential_matrix(essential_matrix);

    let millis = SystemTime::now().duration_since(start).unwrap().as_millis();

    // let color = Rgba([255u8, 0, 0, 10]);
    // let red: Rgb<u8> = Rgb([255u8, 0u8, 0u8]);

    // for kp in matched_keypoints {
    // 	let x = kp.0.x;
    // 	let y = kp.0.y;

    // 	let xx = kp.1.x;
    // 	let yy = kp.1.y;

    // 	let line_img = draw_line_segment(&mut img, (x, y), (xx, yy), color);
    // }

    println!("matched keypoints: {}", matched_keypoints.len());
    println!("essential mtx ");
    println!("    translation : {}", tranlation);
    println!("       rotation : {}", rotation);
    println!("------------------------------------------");
    println!("Total time (millis)           : {:?}", millis);
}

fn compute_kepoint_descriptors(
    img_a: ImageBuffer<Luma<u8>, Vec<u8>>,
) -> (Vec<KeyPoint>, Vec<Descriptor>) {
    let blurred_img_a = image::imageops::blur(&img_a, 2.5);
    let kps_a = fast_detect::fast_keypoints_img(&blurred_img_a, 6);

    let key_points_with_orientation = fast_detect::compute_orientations_img(&blurred_img_a, &kps_a);

    let seed = 2523523;
    let mut random = Rand::new_with_seed(seed);

    // PHASE 3  -  Compute BRIEF descriptors for each keypoint so we can visually match them
    let sampling_pattern = descriptors::generate_sampling_pattern(&mut random, 100, 200);

    let descriptors_a = descriptors::compute_brief_descriptors_img(
        &blurred_img_a,
        &key_points_with_orientation,
        &sampling_pattern,
    );
    (key_points_with_orientation, descriptors_a)
}
