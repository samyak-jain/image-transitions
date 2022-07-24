#![feature(restricted_std)]

use cust::prelude::*;
use std::error::Error;

static PTX: &str = include_str!("../ptx/image.ptx");

fn check() -> Result<(), Box<dyn Error>> {
    let _ctx = cust::quick_init().unwrap();

    let module = Module::from_ptx(PTX, &[]).unwrap();

    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let lhs = [1, 2, 3, 4];
    let rhs = [2, 3, 4, 5];

    let lhs_gpu = lhs.as_dbuf().unwrap();
    let rhs_gpu = rhs.as_dbuf().unwrap();

    let mut out = vec![0; 4];
    let out_buf = out.as_slice().as_dbuf().unwrap();

    let func = module.get_function("add").unwrap();

    let (_, block_size) = func.suggested_launch_configuration(0, 0.into()).unwrap();

    let grid_size = (4 + block_size - 1) / block_size;

    unsafe {
        launch!(
            // slices are passed as two parameters, the pointer and the length.
            func<<<grid_size, block_size, 0, stream>>>(
                lhs_gpu.as_device_ptr(),
                lhs_gpu.len(),
                rhs_gpu.as_device_ptr(),
                rhs_gpu.len(),
                out_buf.as_device_ptr(),
            )
        )
        .unwrap();
    }

    stream.synchronize().unwrap();

    out_buf.copy_to(&mut out).unwrap();

    println!("{:#?} + {:#?} = {:#?}", lhs, rhs, out);

    Ok(())
}

#[cfg(test)]
mod tests {
    use crate::check;

    #[test]
    fn it_works() {
        check().unwrap();
    }
}
