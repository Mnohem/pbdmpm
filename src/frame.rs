use crate::pbd_mpm::{self, *};
use glam::*;
use rayon::prelude::*;
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

pub struct World {
    pub simulation: Simulation,
}

impl World {
    pub fn update(&mut self) {
        self.simulation.step(0.1, 5);
    }

    pub fn draw(&self, canvas: &mut [u32]) {
        canvas.fill(0xff889911);

        for x in self.simulation.particle_x.iter() {
            debug_assert!(x.x < pbd_mpm::GRID_WIDTH as Real && x.y < pbd_mpm::GRID_HEIGHT as Real);
            canvas[(x.x * pbd_mpm::CELL_WIDTH as Real).floor() as usize
                + (x.y * pbd_mpm::CELL_HEIGHT as Real).floor() as usize * crate::CANVAS_WIDTH] =
                0xffffffff;
        }
    }

    pub fn random_init() -> Self {
        let mut particles = vec![];
        let mut rng = thread_rng();
        // This gives us vecs in [-1, 1) x [-1, 1)
        let distr = Standard.map(|v: Vector| v * 2.0 - 1.0);

        let spacing = 0.5;
        let box_origin = UVec2::new(8, 8);
        let box_size = 16;
        let mut x = 0.0;
        while x < box_size as Real {
            let mut y = 0.0;
            while y < box_size as Real {
                particles.push(Particle {
                    x: Into::<Vector>::into(box_origin.as_vec2()) + Vector::new(x, y),
                    d: rng.sample(&distr),
                    ..Default::default()
                });
                y += spacing;
            }
            x += spacing;
        }
        World {
            simulation: Simulation::new(particles),
        }
    }
    pub fn init_liquid_box() -> Self {
        let mut particles = vec![];

        let spacing = 0.5;
        let box_origin = UVec2::new(16, 16);
        let box_size = 32;
        let mut x = 0.0;
        while x < box_size as Real {
            let mut y = 0.0;
            while y < box_size as Real {
                particles.push(Particle {
                    x: Into::<Vector>::into(box_origin.as_vec2()) + Vector::new(x, y),
                    f: ConstrainedValue { liquid_density: 1.0 },
                    matter: Matter::Liquid,
                    ..Default::default()
                });
                y += spacing;
            }
            x += spacing;
        }
        World {
            simulation: Simulation::new(particles),
        }
    }
}
