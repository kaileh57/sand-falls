//! Criterion benchmarks for the cell passes.
#![allow(missing_docs, reason = "benchmark harness")]

use criterion::{Criterion, criterion_group, criterion_main};
use fe_core::{Cell, IRect};
use fe_sim::World;
use std::hint::black_box;
use std::sync::Arc;

fn world(threads: usize, fill: &str) -> World {
    let mats = Arc::new(fe_materials::base_set());
    let id = mats.id(fill).unwrap_or_default();
    let floor = mats.id("bedrock").unwrap_or_default();
    let mut w = World::new(mats, 3, threads);
    w.ensure_rect(IRect::new(0, 0, 512, 512), Cell::AIR);
    w.fill_rect(IRect::new(0, 500, 512, 512), floor);
    w.fill_rect(IRect::new(0, 0, 512, 300), id);
    w
}

fn bench(c: &mut Criterion) {
    for (name, mat) in [("sand", "sand"), ("water", "water")] {
        for threads in [1, 4] {
            c.bench_function(&format!("{name}_falling_512x300_{threads}t"), |b| {
                b.iter_batched(
                    || world(threads, mat),
                    |mut w| {
                        for _ in 0..10 {
                            black_box(w.step_cells());
                            w.finish_tick();
                        }
                        w
                    },
                    criterion::BatchSize::LargeInput,
                );
            });
        }
    }
    c.bench_function("settled_sleeping_world", |b| {
        let mut w = world(1, "stone");
        for _ in 0..4 {
            w.step_cells();
            w.finish_tick();
        }
        b.iter(|| {
            black_box(w.step_cells());
            w.finish_tick();
        });
    });
}

criterion_group!(benches, bench);
criterion_main!(benches);
