use cust::{error::CudaError, prelude::*};
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

pub enum GridStride {
    Default,
    Custom(u64),
}

static PTX: &str = include_str!("../ptx/image.ptx");

// This function takes the raw data of 2 images and generates all the intermediate frames necessary
// for a cross fade affect. The way this works,
// ```output_image = (1 - alpha) * first_image + alpha * second image```.
// We have a different alpha (which determines the transparency) for each iteration. We calculate
// what the value should be based on the number of iterations. The output vector is flat vector
// with all the raw intermediate frames concatenated together.
// This function is generic over a constant called STRIDE. This paramete
pub fn cross_fade(
    first_image: &[u8],
    second_image: &[u8],
    iterations: u16,
    stride: GridStride,
) -> Result<Vec<u8>, TransitionError> {
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

    // SAFETY: We are making sure not to access this buffer until we have passed it to the kernel
    // and written the values to it
    // iterations can be usize since we don't expect this to work on anything less than 32-bit
    // systems
    let mut output_buffer: UnifiedBuffer<u8> =
        unsafe { UnifiedBuffer::uninitialized(image_size * iterations as usize) }?;

    let func = module.get_function("cross_fade")?;
    let (_, block_size) = func.suggested_launch_configuration(0, 0.into())?;

    let stride_number = match stride {
        GridStride::Default => 100,
        GridStride::Custom(number) => number,
    };

    // We want the grid size to be large enough that we can parallely compute for all iterations of
    // the entire image. We divide it by stride so that we don't take up too much vram. We want
    // each thread to do multiple iterations
    // We are casting grid_size to u32 because that's the maximum supported by cuda. This should
    // likely be fine since we don't expect any of the values like image_size, block_size and
    // iterations to be larger than u32. The reason we are casting everything to u64 first is
    // because it's possible for ```image_size * iterations``` to go higher than u32 and we don't
    // want overflow to happen. But since we are later diving this number by ```block_size * stride```
    // we expect converting to u32 should usually work
    let grid_size = u32::try_from(
        (image_size as u64 * u64::from(iterations) + u64::from(block_size) * 100)
            / (u64::from(block_size) * stride_number),
    )
    .expect(&format!(
        "image size: {} or iterations: {} are too large",
        image_size, iterations
    ));

    unsafe {
        launch!(
            func<<<grid_size, block_size, 0, stream>>>(
                first_image_buffer.as_device_ptr(),
                first_image_buffer.len(),
                second_image_buffer.as_device_ptr(),
                second_image_buffer.len(),
                iterations,
                block_size as u64 * grid_size as u64,
                stride_number,
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
    use crate::GridStride;

    #[test]
    fn test_basic() {
        let first_image: &[u8] = &[100, 255, 5, 76];
        let second_image: &[u8] = &[28, 8, 245, 100];
        let iterations = 3;

        let output =
            cross_fade(first_image, second_image, iterations, GridStride::Default).unwrap();

        let split_output = output.chunks_exact(first_image.len()).collect::<Vec<_>>();

        assert_eq!(
            split_output,
            vec![[100, 255, 5, 76,], [64, 131, 125, 88,], [28, 8, 245, 100,]]
        )
    }

    #[test]
    fn test_large_random_image() {
        const N: usize = 11059200;
        let first_image: Vec<u8> = (0..N).map(|_| rand::random::<u8>()).collect();
        let second_image: Vec<u8> = (0..N).map(|_| rand::random::<u8>()).collect();
        let iterations = 720;

        let output =
            cross_fade(&first_image, &second_image, iterations, GridStride::Default).unwrap();

        let split_output = output.chunks_exact(first_image.len()).collect::<Vec<_>>();
        assert_ne!(split_output.last().unwrap(), &vec![0; first_image.len()])
    }
}
