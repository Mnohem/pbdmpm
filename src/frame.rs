use std::mem::take;

use crate::pbd_mpm::*;
use crate::canvas_dims::CANVAS_WIDTH;
use crate::{GRID_WIDTH, GRID_HEIGHT};
use crate::kernel::pbd_mpm::{Real, Vector, CELL_WIDTH, CELL_HEIGHT};
use bytemuck::{cast, cast_slice};
use glam::UVec2;
use krnl::buffer::Buffer;
use krnl::device::Device;
use rand::{
    distributions::{Distribution, Standard},
    thread_rng, Rng,
};

pub struct World {
    pub simulation: Simulation,
}

impl World {
    pub fn update(&mut self) {
        self.simulation.step(0.1, 5).unwrap();
    }

    pub fn draw(&self, canvas: &mut [u32]) {
        canvas.fill(0xff889911);

        let liquids_vec = self.simulation.liquids.particle_x.to_vec().unwrap();
        let elastics_vec = self.simulation.elastics.particle_x.to_vec().unwrap();
        let liquids_x: &[Vector] = cast_slice(&liquids_vec[..]);
        let elastics_x: &[Vector] = cast_slice(&elastics_vec[..]);

        for x in liquids_x.iter().chain(elastics_x.iter())
        {
            debug_assert!(x.x < GRID_WIDTH as Real && x.y < GRID_HEIGHT as Real);
            canvas[(x.x * CELL_WIDTH as Real).floor() as usize
                + (x.y * CELL_HEIGHT as Real).floor() as usize * CANVAS_WIDTH] =
                0xffffffff;
        }
    }

    pub fn spawn(
        &mut self,
        Particle {
            x,
            d,
            c,
            matter,
            mass,
        }: Particle,
    ) {
        self.simulation.num_particles += 1;
        match matter {
            Matter::Elastic {
                deformation_gradient: f,
            } => {
                let mut elastics_x = take(&mut self.simulation.elastics.particle_x).into_vec().unwrap();
                let mut elastics_d = take(&mut self.simulation.elastics.particle_d).into_vec().unwrap();
                let mut elastics_c = take(&mut self.simulation.elastics.particle_c).into_vec().unwrap();
                let mut elastics_f = take(&mut self.simulation.elastics.particle_f).into_vec().unwrap();
                let mut elastics_mass = take(&mut self.simulation.elastics.particle_mass).into_vec().unwrap();
                elastics_x.push(cast(x));
                elastics_d.push(cast(d));
                elastics_c.push(cast(c.x_axis));
                elastics_c.push(cast(c.y_axis));
                elastics_f.push(cast(f.x_axis));
                elastics_f.push(cast(f.y_axis));
                elastics_mass.push(mass);
                let d = self.simulation.device.clone();
                self.simulation.elastics.particle_x = Buffer::from_vec(elastics_x).into_device(d.clone()).unwrap();
                self.simulation.elastics.particle_d = Buffer::from_vec(elastics_d).into_device(d.clone()).unwrap();
                self.simulation.elastics.particle_c = Buffer::from_vec(elastics_c).into_device(d.clone()).unwrap();
                self.simulation.elastics.particle_f = Buffer::from_vec(elastics_f).into_device(d.clone()).unwrap();
                self.simulation.elastics.particle_mass = Buffer::from_vec(elastics_mass).into_device(d).unwrap();
            }
            Matter::Liquid { liquid_density } => {
                let mut liquids_x = take(&mut self.simulation.liquids.particle_x).into_vec().unwrap();
                let mut liquids_d = take(&mut self.simulation.liquids.particle_d).into_vec().unwrap();
                let mut liquids_c = take(&mut self.simulation.liquids.particle_c).into_vec().unwrap();
                let mut liquids_desired_density = take(&mut self.simulation.liquids.particle_desired_density).into_vec().unwrap();
                let mut liquids_mass = take(&mut self.simulation.liquids.particle_mass).into_vec().unwrap();
                liquids_x.push(cast(x));
                liquids_d.push(cast(d));
                liquids_c.push(cast(c.x_axis));
                liquids_c.push(cast(c.y_axis));
                liquids_desired_density.push(liquid_density);
                liquids_mass.push(mass);
                let d = self.simulation.device.clone();
                self.simulation.liquids.particle_x = Buffer::from_vec(liquids_x).into_device(d.clone()).unwrap();
                self.simulation.liquids.particle_d = Buffer::from_vec(liquids_d).into_device(d.clone()).unwrap();
                self.simulation.liquids.particle_c = Buffer::from_vec(liquids_c).into_device(d.clone()).unwrap();
                self.simulation.liquids.particle_desired_density = Buffer::from_vec(liquids_desired_density).into_device(d.clone()).unwrap();
                self.simulation.liquids.particle_mass = Buffer::from_vec(liquids_mass).into_device(d).unwrap();
            }
        }
    }

    pub fn random_init() -> Self {
        let mut particles = vec![];
        let mut rng = thread_rng();
        // This gives us vecs in [-1, 1) x [-1, 1)
        let distr = Standard.map(|x: f32| x * 2.0 - 1.0);

        let spacing = 0.5;
        let box_origin = UVec2::new(8, 8);
        let box_size = 16;
        let mut x = 0.0;
        while x < box_size as Real {
            let mut y = 0.0;
            while y < box_size as Real {
                particles.push(Particle {
                    x: Into::<Vector>::into(box_origin.as_vec2()) + Vector::new(x, y),
                    d: Vector::new(rng.sample(&distr), rng.sample(&distr)),
                    ..Default::default()
                });
                y += spacing;
            }
            x += spacing;
        }
        let device = Device::builder().build().unwrap_or(Device::host());
        World {
            simulation: Simulation::new(particles, device).unwrap(),
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
                    matter: Matter::Liquid {
                        liquid_density: 1.0,
                    },
                    ..Default::default()
                });
                y += spacing;
            }
            x += spacing;
        }
        let device = Device::builder().build().unwrap_or(Device::host());
        World {
            simulation: Simulation::new(particles, device).unwrap(),
        }
    }
}
