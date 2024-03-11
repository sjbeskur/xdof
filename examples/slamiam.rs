use image::io::Reader as ImageReader;
use image::Rgb;
use image::RgbImage;
use std::time::SystemTime;
use xdof::image_impl::greyscale_gaussian_blur;

use image::GrayImage;
use image::ImageBuffer;
//use image::Luma;

use xdof::common::*;
use xdof::descriptors;
use xdof::essential;
use xdof::fast_detect;
use xdof::image_impl;
use xdof::matcher;
use xdof::rand::Rand;
//use xdof::Slam;

fn main() {
    let args = std::env::args().collect::<Vec<String>>();

    if args.len() < 3 {
        println!("\nUsage: slamiam <image_file_1> <image_file_2> \n");
        return;
    }

    let start = SystemTime::now();
    let img_a = read_image(&args[1]);
    let img_b = read_image(&args[2]);

    // let mut slam = Slam::new(img_a, img_b);
    // let rslt = slam.calculate_pose();
    // println!("{:?}", rslt);
    let (kpswo_a, descriptors_a) = compute_kepoint_descriptors(&img_a);
    let (kpswo_b, descriptors_b) = compute_kepoint_descriptors(&img_b);

    let max_hamming_distance = 300;
    // PHASE 4  -  Match features between the two images
    let matched_keypoints = matcher::match_features(
        &kpswo_a,
        &descriptors_a,
        &kpswo_b,
        &descriptors_b,
        max_hamming_distance,
    );

    println!("matched keypoints: {}", matched_keypoints.len());

    let seed = 2523523;
    let mut random = Rand::new_with_seed(seed);
    // // PHASE 5  -  RANSAC to find the best rotation and translation using 8 point algorithm
    let essential_matrix =
        essential::estimate_essential_ransac(&matched_keypoints, 1000, 10.0, &mut random);

    // let (rotation, tranlation) = essential::decompose_essential_matrix(essential_matrix);

    let decomposed_essential = if let Some(essential) = essential_matrix {
        Some(essential::decompose_essential_matrix(essential))
    } else {
        None
    };

    // Once we have the essential matrix we don't need to compute this every time
    //   Check out  https://github.com/PoseLib/PoseLib
    // The above is the estimation of the absolute pose

    // Triangulate Points
    //   LOST  https://gtsam.org/2023/02/04/lost-triangulation.html
    //   look at lmvs plucker

    // Poselib P3P solver or LambdaTwist

    // Determine Absolute Pose - relative to the point cloud (vector of points)

    // P = K * [R,t]  // projection matrix
    // s * [u;v;1] = P * [x; y; z; 1]

    let millis = start.elapsed().unwrap().as_millis();

    //draw_matches();

    if let Some((trans_vec, rotation)) = decomposed_essential {
        println!("essential mtx ");
        println!("    translation : {}", trans_vec);
        println!("       rotation : {}", rotation);
        println!("------------------------------------------");
        println!("Total time (millis)           : {:?}", millis);
    }
}

//fn draw_matches() {
// let color = Rgba([255u8, 0, 0, 10]);
// let red: Rgb<u8> = Rgb([255u8, 0u8, 0u8]);

// for kp in matched_keypoints {
// 	let x = kp.0.x;
// 	let y = kp.0.y;

// 	let xx = kp.1.x;
// 	let yy = kp.1.y;

// 	let line_img = draw_line_segment(&mut img, (x, y), (xx, yy), color);
// }
//}

fn compute_kepoint_descriptors(
    img: &ImageBuffer<Rgb<u8>, Vec<u8>>,
) -> (Vec<KeyPoint>, Vec<Descriptor>) {
    //let blurred_img_a = image::imageops::blur(img_a, 3.0);

    let width = img.width() as usize;
    let height = img.height() as usize;

    let blurred_img = greyscale_gaussian_blur(
        img.as_raw(),
        img.width() as usize,
        img.height() as usize,
        3.0,
    );

    let kps_a = fast_detect::fast_keypoints(&blurred_img, width, height, 6);

    let key_points_with_orientation =
        fast_detect::compute_orientations(&blurred_img, width, &kps_a);

    let seed = 2523523;
    let mut random = Rand::new_with_seed(seed);

    // PHASE 3  -  Compute BRIEF descriptors for each keypoint so we can visually match them
    let sampling_pattern = descriptors::generate_sampling_pattern(&mut random, 100, 200);

    let descriptors_a = descriptors::compute_brief_descriptors(
        &blurred_img,
        width as u32,
        height as u32,
        &key_points_with_orientation,
        &sampling_pattern,
    );
    (key_points_with_orientation, descriptors_a)
}

pub fn read_image(filename: &str) -> RgbImage {
    let img = ImageReader::open(filename).unwrap().decode().unwrap();

    let gimg =
        image::RgbImage::from_raw(img.width(), img.height(), img.into_bytes().to_vec()).unwrap();

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
