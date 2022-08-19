use cust::{error::CudaError, memory::DeviceCopy, prelude::*};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransitionError {
    #[error("Sizes of the two images are not equal. First image size: {first_image_size}, Second image size: {second_image_size}")]
    SizeNotEqual {
        first_image_size: usize,
        second_image_size: usize,
    },
    #[error("gpu error")]
    GPUError(#[from] CudaError),
}

static PTX: &str = include_str!("../ptx/image.ptx");

pub fn cross_fade<T: DeviceCopy + std::fmt::Debug + Default>(
    first_image: &[T],
    second_image: &[T],
    iterations: usize,
) -> Result<Vec<T>, TransitionError> {
    let _ctx = cust::quick_init()?;
    let module = Module::from_ptx(PTX, &[])?;
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None)?;

    if first_image.len() != second_image.len() {
        return Err(TransitionError::SizeNotEqual {
            first_image_size: first_image.len(),
            second_image_size: second_image.len(),
        });
    }

    let image_size = first_image.len();

    let first_image_buffer = first_image.as_dbuf()?;
    let second_image_buffer = second_image.as_dbuf()?;

    // TODO: Add safety comment
    let mut output_buffer: UnifiedBuffer<T> =
        unsafe { UnifiedBuffer::uninitialized(image_size * iterations) }?;

    let func = module.get_function("cross_fade")?;
    let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;
    let grid_size = ((image_size * iterations) as u32 + block_size - 1) / block_size;

    unsafe {
        launch!(
            func<<<grid_size, block_size, 0, stream>>>(
                first_image_buffer.as_device_ptr(),
                first_image_buffer.len(),
                second_image_buffer.as_device_ptr(),
                second_image_buffer.len(),
                iterations,
                output_buffer.as_unified_ptr(),
            )
        )?;
    }

    stream.synchronize()?;

    // TODO: see if we can avoid an allocation here
    // This is safe since we're accessing the unified memory after stream synchonization
    let output_image = output_buffer.to_vec();

    Ok(output_image)
}

#[cfg(test)]
mod tests {
    use crate::cross_fade;

    #[test]
    fn test_cross_fade() {
        let first_image: &[u8] = &[100, 255, 5, 76];
        let second_image: &[u8] = &[28, 8, 245, 100];
        let iterations = 10;
        let output = cross_fade(first_image, second_image, iterations).unwrap();
        let split_output = output.chunks_exact(first_image.len()).collect::<Vec<_>>();
        println!("output: {:#?}", split_output);
        // assert_eq!(output, vec![5, 7, 9, 11]);
    }
}
