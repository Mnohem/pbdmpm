use glam::*;
use crate::kernel::*;
use rayon::prelude::*;

const GRID_SIZE: usize = 64;
pub const GRID_WIDTH: usize = GRID_SIZE;
pub const GRID_HEIGHT: usize = GRID_SIZE;
// CANVAS_WIDTH and CANVAS_HEIGHT must be larger than GRID_WIDTH and GRID_HEIGHT respectively
// In the simulator the cell is a 1x1 square, this is the pixel width and height of a cell
pub const CELL_WIDTH: usize = crate::CANVAS_WIDTH / GRID_WIDTH;
pub const CELL_HEIGHT: usize = crate::CANVAS_HEIGHT / GRID_HEIGHT;

pub type Vector = Vec2;
pub type Matrix = Mat2;
pub type Real = f32;
pub type Atomic = std::sync::atomic::AtomicI32;
pub type FixedPoint = i32;

pub const GRAVITY: Real = -0.3;
pub const FRICTION: Real = 0.5;
pub const GRID_LOWER_BOUND: Vector = Vector::ONE;
pub const GRID_UPPER_BOUND: Vector = Vector::new(GRID_WIDTH as Real - 2.0, GRID_HEIGHT as Real - 2.0);

pub const LIQUID_RELAXATION: Real = 1.5;
pub const LIQUID_VISCOSITY: Real = 0.01;

#[derive(Clone, Copy)]
pub enum Matter {
    Elastic { deformation_gradient: Matrix },
    Liquid { liquid_density: Real },
}
pub struct Particle {
    pub x: Vector,
    pub d: Vector,
    pub c: Matrix, // deformation displacement (D) when multiplied by dt
    pub matter: Matter,
    pub mass: Real,
}
impl Default for Particle {
    fn default() -> Self {
        Self {
            x: Vector::ZERO,
            d: Vector::ZERO,
            c: Matrix::ZERO,
            matter: Matter::Elastic {
                deformation_gradient: Matrix::ZERO,
            },
            mass: 1.0,
        }
    }
}

pub struct LiquidStorage {
    pub particle_x: Vec<Vector>,
    pub particle_d: Vec<Vector>,
    pub particle_c: Vec<Matrix>,
    pub particle_desired_density: Vec<Real>,
    pub particle_mass: Vec<Real>,
}
pub struct ElasticStorage {
    pub particle_x: Vec<Vector>,
    pub particle_d: Vec<Vector>,
    pub particle_c: Vec<Matrix>,
    pub particle_f: Vec<Matrix>,
    pub particle_mass: Vec<Real>,
}
pub struct Simulation {
    grid_dx: [Atomic; GRID_WIDTH * GRID_HEIGHT],
    grid_dy: [Atomic; GRID_WIDTH * GRID_HEIGHT],
    grid_mass: [Atomic; GRID_WIDTH * GRID_HEIGHT],
    pub liquids: LiquidStorage,
    pub elastics: ElasticStorage,
}
impl Default for Simulation {
    fn default() -> Self {
        Self {
            grid_dx: [const { Atomic::new(0) }; GRID_WIDTH * GRID_HEIGHT],
            grid_dy: [const { Atomic::new(0) }; GRID_WIDTH * GRID_HEIGHT],
            grid_mass: [const { Atomic::new(0) }; GRID_WIDTH * GRID_HEIGHT],
            liquids: LiquidStorage {
                particle_x: vec![],
                particle_d: vec![],
                particle_c: vec![],
                particle_desired_density: vec![],
                particle_mass: vec![],
            },
            elastics: ElasticStorage {
                particle_x: vec![],
                particle_d: vec![],
                particle_c: vec![],
                particle_f: vec![],
                particle_mass: vec![],
            },
        }
    }
}

// Switching to a Position based model is beneficial for removing our dependence on the time delta,
// although this requires us to abandon our models of stress and constitutive equations, as without
// the time delta, we are no longer attempting to model the actual physics in this system, but
// rather using similar equations whilst constraining specific variables to do what we want
//
// Differences in our design with the seed pbmpm:
// 1. Starting point
// Ordering of steps is the same, but they start with the grid update, into g2p and so on g2p
// The semantics are still the same since they use alternating grids which they write and read from
// 2. Enforcing Boundaries
// They use something they call a guardian for boundaries, and also have collider shape collisions
// included in the grid update step
// 4. Currently no volume
// The volume calculation seems to only be used for the fluid material, so it will be implemented
// along with that
impl Simulation {
    pub fn new(particles: Vec<Particle>) -> Self {
        let mut sim = Self::default();
        for &Particle {
            x,
            d,
            c,
            matter,
            mass,
        } in particles.iter()
        {
            match matter {
                Matter::Elastic {
                    deformation_gradient: f,
                } => {
                    sim.elastics.particle_x.push(x);
                    sim.elastics.particle_d.push(d);
                    sim.elastics.particle_c.push(c);
                    sim.elastics.particle_f.push(f);
                    sim.elastics.particle_mass.push(mass);
                }
                Matter::Liquid { liquid_density } => {
                    sim.liquids.particle_x.push(x);
                    sim.liquids.particle_d.push(d);
                    sim.liquids.particle_c.push(c);
                    sim.liquids.particle_desired_density.push(liquid_density);
                    sim.liquids.particle_mass.push(mass);
                }
            }
        }
        sim
    }
    // TODO https://gafferongames.com/post/fix_your_timestep/
    pub fn step(&mut self, dt: Real, iterations: usize) {
        for _ in 0..iterations {
            self.grid_dx = [const { Atomic::new(0) }; GRID_WIDTH * GRID_HEIGHT];
            self.grid_dy = [const { Atomic::new(0) }; GRID_WIDTH * GRID_HEIGHT];
            self.grid_mass = [const { Atomic::new(0) }; GRID_WIDTH * GRID_HEIGHT];

            // PBD MPM, basically working to reset our deformation gradient gradually
            Self::liquid_constraints(
                &self.liquids.particle_desired_density,
                &mut self.liquids.particle_c,
            );
            Self::elastic_constraints(&self.elastics.particle_f, &mut self.elastics.particle_c);

            // 1. particle-to-grid P2G
            Self::p2g(
                &self.liquids.particle_x,
                &self.liquids.particle_d,
                &self.liquids.particle_c,
                &self.liquids.particle_mass,
                &self.grid_mass,
                &self.grid_dx,
                &self.grid_dy,
            );
            Self::p2g(
                &self.elastics.particle_x,
                &self.elastics.particle_d,
                &self.elastics.particle_c,
                &self.elastics.particle_mass,
                &self.grid_mass,
                &self.grid_dx,
                &self.grid_dy,
            );

            // 2. calculate grid displacements
            Self::update_grid(&self.grid_mass, &mut self.grid_dx, &mut self.grid_dy);

            // 3. grid-to-particle (G2P).
            Self::g2p(
                &self.grid_dx,
                &self.grid_dy,
                &self.liquids.particle_x,
                &mut self.liquids.particle_d,
                &mut self.liquids.particle_c,
            );
            Self::g2p(
                &self.grid_dx,
                &self.grid_dy,
                &self.elastics.particle_x,
                &mut self.elastics.particle_d,
                &mut self.elastics.particle_c,
            );
        }

        // 4. integrate our values to update our particle positions and deformation gradients
        Self::liquid_integrate(
            dt,
            &self.liquids.particle_c,
            &mut self.liquids.particle_x,
            &mut self.liquids.particle_d,
            &mut self.liquids.particle_desired_density,
        );
        Self::elastic_integrate(
            dt,
            &self.elastics.particle_c,
            &mut self.elastics.particle_x,
            &mut self.elastics.particle_d,
            &mut self.elastics.particle_f,
        );
    }
    pub fn elastic_constraints(particle_f: &[Matrix], particle_c: &mut [Matrix]) {
        particle_c
            .par_iter_mut()
            .zip_eq(particle_f)
            .for_each(|(c, f)| {
                elastic_constrain(c, f)
            })
    }
    pub fn liquid_constraints(particle_desired_density: &[Real], particle_c: &mut [Matrix]) {
        particle_c
            .par_iter_mut()
            .zip_eq(particle_desired_density)
            .for_each(|(c, &liquid_density)| {
                liquid_constrain(c, liquid_density)
            })
    }

    pub fn p2g(
        particle_x: &[Vector],
        particle_d: &[Vector],
        particle_c: &[Matrix],
        particle_mass: &[Real],
        grid_mass: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
        grid_dx: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
        grid_dy: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
    ) {
        // here we modify over the whole grid, but they are given through shared references which
        // is valid since we will only use atomic operations on the grid
        particle_x
            .par_iter()
            .zip_eq(particle_d)
            .zip_eq(particle_c)
            .zip_eq(particle_mass)
            .for_each(|(((&x, &d), &c), &mass)| {
                p2g_step(grid_mass, grid_dx, grid_dy, x, d, c, mass)
            })
    }
    pub fn update_grid(
        grid_mass: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
        grid_dx: &mut [Atomic; GRID_WIDTH * GRID_HEIGHT],
        grid_dy: &mut [Atomic; GRID_WIDTH * GRID_HEIGHT],
    ) {
        grid_dx
            .par_iter_mut()
            .zip_eq(grid_dy.par_iter_mut())
            .enumerate()
            .zip_eq(grid_mass)
            .for_each(|((i, (dx, dy)), mass)| {
                grid_update_at(i, mass, dx, dy)
            })
    }
    pub fn g2p(
        grid_dx: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
        grid_dy: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
        particle_x: &[Vector],
        particle_d: &mut [Vector],
        particle_c: &mut [Matrix],
    ) {
        particle_d
            .par_iter_mut()
            .zip_eq(particle_c.par_iter_mut())
            .zip_eq(particle_x)
            .for_each(|((d, c), &x)| {
                g2p_step(grid_dx, grid_dy, d, c, x)
            })
    }
    pub fn liquid_integrate(
        dt: Real,
        particle_c: &[Matrix],
        particle_x: &mut [Vector],
        particle_d: &mut [Vector],
        particle_desired_density: &mut [Real],
    ) {
        particle_x
            .par_iter_mut()
            .zip_eq(particle_d.par_iter_mut())
            .zip_eq(particle_desired_density.par_iter_mut())
            .zip_eq(particle_c)
            .for_each(|(((x, d), liquid_density), c)| {
                liquid_integrate_step(dt, x, d, liquid_density, c)
            })
    }
    pub fn elastic_integrate(
        dt: Real,
        particle_c: &[Matrix],
        particle_x: &mut [Vector],
        particle_d: &mut [Vector],
        particle_f: &mut [Matrix],
    ) {
        particle_x
            .par_iter_mut()
            .zip_eq(particle_d.par_iter_mut())
            .zip_eq(particle_f.par_iter_mut())
            .zip_eq(particle_c)
            .for_each(|(((x, d), f), c)| {
                elastic_integrate_step(dt, x, d, f, c)
            })
    }
}

