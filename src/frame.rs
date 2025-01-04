use crate::pbd_mpm::{self, *};
use glam::*;
use rand::{
    distributions::{Distribution, Standard},
    thread_rng, Rng,
};

pub struct World {
    pub simulation: Simulation,
}

impl World {
    pub fn update(&mut self) {
        self.simulation.step(0.1, 5);
    }

    pub fn draw(&self, canvas: &mut [u32]) {
        canvas.fill(0xff889911);

        for x in self.simulation.liquids.particle_x.iter().chain(self.simulation.elastics.particle_x.iter()) {
            debug_assert!(x.x < pbd_mpm::GRID_WIDTH as Real && x.y < pbd_mpm::GRID_HEIGHT as Real);
            canvas[(x.x * pbd_mpm::CELL_WIDTH as Real).floor() as usize
                + (x.y * pbd_mpm::CELL_HEIGHT as Real).floor() as usize * crate::CANVAS_WIDTH] =
                0xffffffff;
        }
    }
    
    pub fn spawn(&mut self, Particle { x, d, c, matter, mass }: Particle) {
        match matter {
            Matter::Elastic {
                deformation_gradient: f,
            } => {
                self.simulation.elastics.particle_x.push(x);
                self.simulation.elastics.particle_d.push(d);
                self.simulation.elastics.particle_c.push(c);
                self.simulation.elastics.particle_f.push(f);
                self.simulation.elastics.particle_mass.push(mass);
            }
            Matter::Liquid { liquid_density } => {
                self.simulation.liquids.particle_x.push(x);
                self.simulation.liquids.particle_d.push(d);
                self.simulation.liquids.particle_c.push(c);
                self.simulation.liquids.particle_desired_density.push(liquid_density);
                self.simulation.liquids.particle_mass.push(mass);
            }
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
                    matter: Matter::Liquid {
                        liquid_density: 1.0,
                    },
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
