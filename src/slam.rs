use nalgebra::Matrix3;
use nalgebra::Vector3;

use crate::common::*;
use crate::descriptors;
use crate::essential;
use crate::essential::decompose_essential_matrix;
use crate::fast_detect;
use crate::image_impl;
use crate::matcher;
use crate::rand::*;

pub struct Slam<'a> {
    image_a: Image<'a>,
    image_b: Image<'a>,
    random: Rand,
    patch_size: usize,
    num_pairs: usize,
    max_hamming_distance: usize,
    blur_radius: f32,
    essential_num_iterations: usize,
    essential_threshold: f32,
}

impl<'a> Slam<'a> {
    pub fn new(image_a: Image<'a>, image_b: Image<'a>) -> Slam<'a> {
        let seed = 2523523;
        let random = Rand::new_with_seed(seed);
        Slam {
            image_a,
            image_b,
            random,
            patch_size: 100,
            num_pairs: 500,
            max_hamming_distance: 300,
            blur_radius: 3.0,
            essential_num_iterations: 1000,
            essential_threshold: 10.0,
        }
    }

    pub fn calculate_pose(
        &mut self,
    ) -> (
        Option<(Matrix3<f64>, Vector3<f64>)>,
        Vec<(KeyPoint, KeyPoint)>,
        (Vec<KeyPoint>, Vec<Descriptor>, Vec<u8>),
        (Vec<KeyPoint>, Vec<Descriptor>),
    ) {
        let (key_points_with_orientation_a, blurred_image_a) = {
            let width = self.image_a.width;
            let height = self.image_a.height;

            // PHASE 1  -  Convert RGB image to greyscale and blur it with a Gaussian filter
            let greyscale = image_impl::rgb_to_grayscale(self.image_a.data, width, height);
            let blurred_img =
                image_impl::greyscale_gaussian_blur(&greyscale, width, height, self.blur_radius);
            let threshold: u8 = 30;

            // PHASE 2  -  Detect FAST keypoints and compute their orientations
            let keypoints = fast_detect::fast_keypoints(&blurred_img, width, height, threshold);
            let key_points_with_orientation =
                fast_detect::compute_orientations(&blurred_img, width, &keypoints);

            (key_points_with_orientation, blurred_img)
        };

        let (key_points_with_orientation_b, blurred_image_b) = {
            let width = self.image_b.width;
            let height = self.image_b.height;

            // PHASE 1  -  Convert RGB image to greyscale and blur it with a Gaussian filter
            let greyscale = image_impl::rgb_to_grayscale(self.image_b.data, width, height);
            let blurred_img =
                image_impl::greyscale_gaussian_blur(&greyscale, width, height, self.blur_radius);
            let threshold: u8 = 30;

            // PHASE 2  -  Detect FAST keypoints and compute their orientations
            let keypoints = fast_detect::fast_keypoints(&blurred_img, width, height, threshold);
            let key_points_with_orientation =
                fast_detect::compute_orientations(&blurred_img, width, &keypoints);

            (key_points_with_orientation, blurred_img)
        };

        // PHASE 3  -  Compute BRIEF descriptors for each keypoint so we can visually match them
        let sampling_pattern = descriptors::generate_sampling_pattern(
            &mut self.random,
            self.patch_size,
            self.num_pairs,
        );

        let descriptors_a = descriptors::compute_brief_descriptors(
            &blurred_image_a,
            self.image_a.width as u32,
            self.image_a.height as u32,
            &key_points_with_orientation_b,
            &sampling_pattern,
        );

        let descriptors_b = descriptors::compute_brief_descriptors(
            &blurred_image_b,
            self.image_b.width as u32,
            self.image_b.height as u32,
            &key_points_with_orientation_b,
            &sampling_pattern,
        );

        // PHASE 4  -  Match features between the two images

        let matched_keypoints = matcher::match_features(
            &key_points_with_orientation_a,
            &descriptors_a,
            &key_points_with_orientation_b,
            &descriptors_b,
            self.max_hamming_distance,
        );

        // PHASE 5  -  RANSAC to find the best rotation and translation using 8 point algorithm
        let essential_matrix = essential::estimate_essential_ransac(
            &matched_keypoints,
            *&self.essential_num_iterations,
            *&self.essential_threshold as f64,
            &mut self.random,
        );

        // PHASE 6  -  Decompose the essential matrix to find the rotation and translation
        let decomposed_essential = if let Some(essential) = essential_matrix {
            Some(decompose_essential_matrix(essential))
        } else {
            None
        };

        (
            decomposed_essential,
            matched_keypoints,
            (
                key_points_with_orientation_a,
                descriptors_a,
                blurred_image_a,
            ),
            (key_points_with_orientation_b, descriptors_b),
        )
    }
}
