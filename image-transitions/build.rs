use cuda_builder::CudaBuilder;

fn main() {
    CudaBuilder::new("../gpu")
        .copy_to("ptx/image.ptx")
        .build()
        .unwrap();
}
