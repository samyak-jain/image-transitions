use std::time::Duration;

use criterion::{criterion_group, criterion_main, Criterion};
use image_transitions::{cross_fade, GridStride};

fn bench_basic(c: &mut Criterion) {
    let first_image: &[u8] = &[100, 255, 5, 76];
    let second_image: &[u8] = &[28, 8, 245, 100];
    let iterations = 3;

    c.bench_function("basic", |b| {
        b.iter(|| cross_fade(first_image, second_image, iterations, GridStride::Default).unwrap())
    });
}

fn bench_large_random_image(c: &mut Criterion) {
    const N: usize = 11059200;
    let first_image: Vec<u8> = (0..N).map(|_| rand::random::<u8>()).collect();
    let second_image: Vec<u8> = (0..N).map(|_| rand::random::<u8>()).collect();
    let iterations = 720;

    c.bench_function("images", |b| {
        b.iter(|| cross_fade(&first_image, &second_image, iterations, GridStride::Default).unwrap())
    });
}

criterion_group!(name = benches; 
    config = Criterion::default().sample_size(10).measurement_time(Duration::from_secs(65)); 
    targets = bench_basic, bench_large_random_image);
criterion_main!(benches);
