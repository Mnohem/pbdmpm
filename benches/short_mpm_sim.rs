use criterion::{criterion_group, criterion_main, Criterion};
use prepixx::frame::World;

fn liquid_update_20(c: &mut Criterion) {
    let mut world = World::init_liquid_box();
    c.bench_function("liquid update x20", |b| {
        b.iter(|| {
            for _ in 0..20 {
                world.update()
            }
        })
    });
}

fn box_update_100(c: &mut Criterion) {
    let mut world = World::random_init();
    c.bench_function("box update x100", |b| {
        b.iter(|| {
            for _ in 0..100 {
                world.update()
            }
        })
    });
}

criterion_group!(benches, liquid_update_20, box_update_100);

criterion_main!(benches);
