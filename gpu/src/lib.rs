#![cfg_attr(
    target_os = "cuda",
    no_std,
    feature(register_attr),
    register_attr(nvvm_internal)
)]
#![allow(improper_ctypes_definitions)]

use cuda_std::{vek::num_traits::ToPrimitive, *};
extern crate alloc;

#[kernel]
pub unsafe fn cross_fade(
    first_image: &[u8],
    second_image: &[u8],
    iterations: u16,
    number_of_threads: u64,
    stride: u64,
    output: *mut u8,
) {
    let thread_id = thread::index_1d();

    // usize to u64 conversion is safe
    let image_length = first_image.len() as u64;

    for idx in (0..stride).map(|stride| stride * number_of_threads + u64::from(thread_id)) {
        if idx >= image_length * u64::from(iterations) {
            return;
        }

        let index = idx % image_length;
        let current_iteration = idx / image_length;

        let elem = &mut *output.add(
            (image_length * current_iteration + index)
                .to_usize()
                .unwrap(),
        );
        let alpha: f64 = current_iteration as f64 / (iterations - 1) as f64;

        let calculation =
            // since index is modulo image_length, as long as we're on 32 bit systems or above, we
            // should be fine casting it to usize
            (1f64 - alpha) * f64::from(first_image[index as usize]) + alpha * f64::from(second_image[index as usize]);

        *elem = calculation.floor() as u8;
    }
}
