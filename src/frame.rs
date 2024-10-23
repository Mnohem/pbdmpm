use crate::mls_mpm::{self, *};
use glam::*;
use rand::{
    distributions::{Distribution, Standard},
    thread_rng, Rng,
};
/// Color representation as abgr
// #[derive(Clone, Copy, Debug, Default)]
// #[repr(u32)]
// pub enum Color {
//     #[default]
//     Gray = 0xff999999,
//     Yellow = 0xff116090,
// }
//
// pub trait Frame<const WIDTH: usize, const HEIGHT: usize> {
//     fn get_frame(&self) -> &[[Color; WIDTH]; HEIGHT];
//     fn draw_at(&self, [x, y]: [usize; 2], canvas: &mut [u8]) {
//         let canvas: &mut [[Color; crate::CANVAS_WIDTH]; crate::CANVAS_HEIGHT] = unsafe {
//             &mut *(canvas.as_mut_ptr() as *mut [[Color; crate::CANVAS_WIDTH]; crate::CANVAS_HEIGHT])
//         };
//         // can always optimize with unchecked
//         for (canvas_row, frame_row) in canvas[y..(y + HEIGHT)]
//             .iter_mut()
//             .zip(self.get_frame().iter())
//         {
//             let canvas_row_section: &mut [Color; WIDTH] =
//                 unsafe { &mut *(canvas_row[x..(x + WIDTH)].as_mut_ptr() as *mut [Color; WIDTH]) };
//             *canvas_row_section = *frame_row;
//         }
//     }
// }
//
// impl<const WIDTH: usize, const HEIGHT: usize> Frame<WIDTH, HEIGHT> for [[Color; WIDTH]; HEIGHT] {
//     fn get_frame(&self) -> &Self {
//         self
//     }
// }

#[derive(Default)]
pub struct World {
    pub simulation: Simulation,
}

impl World {
    pub fn update(&mut self) {
        let iterations = 1;
        for _ in 0..iterations {
            self.simulation.step();
        }
    }

    pub fn draw(&self, canvas: &mut [u32]) {
        canvas.fill(0xff889911);

        for Particle { x, .. } in self.simulation.particles.iter() {
            debug_assert!(x.x < mls_mpm::GRID_WIDTH as f32 && x.y < mls_mpm::GRID_HEIGHT as f32);
            canvas[(x.x * mls_mpm::CELL_WIDTH as f32).floor() as usize
                + (x.y * mls_mpm::CELL_HEIGHT as f32).floor() as usize * crate::CANVAS_WIDTH] =
                0xffffffff;
        }
    }

    pub fn random_init() -> Self {
        let mut particles = vec![];
        let mut rng = thread_rng();
        // This gives us vecs in [-1, 1) x [-1, 1)
        let distr = Standard.map(|v: Vec2| {
            (v * 2.0 - 1.0) - Vec2::ZERO.with_y(1.0)
        });

        let spacing = 0.5;
        let box_origin = UVec2::new(16, 16);
        let box_size = 16;
        let mut x = 0.0;
        while x < box_size as f32 {
            let mut y = 0.0;
            while y < box_size as f32 {
                particles.push(Particle {
                    x: box_origin.as_vec2() + Vec2::new(x, y),
                    v: rng.sample(&distr),
                    ..Default::default()
                });
                y += spacing;
            }
            x += spacing;
        }
        let mut simulation = Simulation {
            particles,
            ..Default::default()
        };
        // scatter particle mass to the grid for computing initial volumes
        simulation.p2g(1.0);
        for p in simulation.particles.iter_mut() {
            let cell_idx = p.x.floor().as_uvec2();
            let weights = p.interpolated_weights();
            
            let mut density = 0.0;
            for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                // scatter our particle's momentum to the grid, using the cell's interpolation
                // weight calculated in 2.1
                let weight = weights[gx].x * weights[gy].y;
                let cell_index = (cell_idx.x as usize + gx - 1) + (cell_idx.y as usize + gy - 1) * mls_mpm::GRID_WIDTH;
                density += simulation.grid[cell_index].mass * weight;
            }

            p.initial_volume = p.mass / density;
        }

        World {
            simulation
        }
    }
    pub fn init_box() -> Self {
        let mut particles = vec![];

        let spacing = 0.5;
        let box_origin = UVec2::new(16, 16);
        let box_size = 32;
        let mut x = 0.0;
        while x < box_size as f32 {
            let mut y = 0.0;
            while y < box_size as f32 {
                particles.push(Particle {
                    x: box_origin.as_vec2() + Vec2::new(x, y),
                    ..Default::default()
                });
                y += spacing;
            }
            x += spacing;
        }
        let mut simulation = Simulation {
            particles,
            ..Default::default()
        };
        // scatter particle mass to the grid for computing initial volumes
        simulation.p2g(1.0);
        for p in simulation.particles.iter_mut() {
            let cell_idx = p.x.floor().as_uvec2();
            let weights = p.interpolated_weights();
            
            let mut density = 0.0;
            for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                let weight = weights[gx].x * weights[gy].y;
                let cell_index = (cell_idx.x as usize + gx - 1) + (cell_idx.y as usize + gy - 1) * mls_mpm::GRID_WIDTH;
                density += simulation.grid[cell_index].mass * weight;
            }

            p.initial_volume = p.mass / density;
        }

        World {
            simulation
        }
    }
}
