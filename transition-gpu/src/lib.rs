#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]
#![allow(improper_ctypes_definitions)]

use cuda_std::*;
extern crate alloc;

#[kernel]
pub unsafe fn cross_fade(
    first_image: &[u8],
    second_image: &[u8],
    iterations: u16,
    output: *mut u8,
) {
    let idx = thread::index_1d();
    let image_length = first_image.len();

    // TODO: take a look at this
    if idx as usize >= image_length * iterations as usize {
        return;
    }

    let index = idx as usize % image_length;
    let current_iteration = idx as usize / image_length;

    // cuda_std::println!("{:#?}", idx);

    // cuda_std::println!("{}", iterations);

    // cuda_std::println!("{:#?}", first_image);

    // cuda_std::println!(
    //     "{}, {}, {}, {}",
    //     index,
    //     current_iteration,
    //     first_image[index],
    //     second_image[index]
    // );

    let elem = &mut *output.add(image_length * current_iteration + index);
    let alpha: f32 = current_iteration as f32 / f32::from(iterations - 1);

    // cuda_std::println!("{}, {}", first_image[index], index);
    let calculation =
        (1f32 - alpha) * f32::from(first_image[index]) + alpha * f32::from(second_image[index]);

    *elem = calculation.floor() as u8;
}
