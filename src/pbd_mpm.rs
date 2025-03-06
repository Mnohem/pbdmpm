use crate::kernel::pbd_mpm::{FixedPoint, Matrix, Real, Vector, GRID_FLAT_LENGTH};
use crate::kernel::*;
use bytemuck::{cast, cast_slice, cast_slice_mut};
use krnl::{buffer::Buffer, device::Device, krnl_core::buffer::UnsafeSlice};
use rayon::prelude::*;

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
    pub particle_x: Buffer<u64>,
    pub particle_d: Buffer<u64>,
    pub particle_c: Buffer<u64>,
    pub particle_desired_density: Buffer<Real>,
    pub particle_mass: Buffer<Real>,
}
pub struct ElasticStorage {
    pub particle_x: Buffer<u64>,
    pub particle_d: Buffer<u64>,
    pub particle_c: Buffer<u64>,
    pub particle_f: Buffer<u64>,
    pub particle_mass: Buffer<Real>,
}
type Grid = Buffer<FixedPoint>;
pub struct Simulation {
    pub num_particles: u32,
    grid_dx: Grid,
    grid_dy: Grid,
    grid_mass: Grid,
    pub liquids: LiquidStorage,
    pub elastics: ElasticStorage,
    pub device: Device,
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
    pub fn new(particles: Vec<Particle>, device: Device) -> krnl::anyhow::Result<Self> {
        let mut liquid_particle_x = vec![];
        let mut liquid_particle_d = vec![];
        let mut liquid_particle_c = vec![];
        let mut liquid_particle_mass = vec![];
        let mut particle_f = vec![];
        let mut elastic_particle_x = vec![];
        let mut elastic_particle_d = vec![];
        let mut elastic_particle_c = vec![];
        let mut elastic_particle_mass = vec![];
        let mut particle_desired_density = vec![];
        let num_particles = particles.len() as u32;

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
                    elastic_particle_x.push(cast(x));
                    elastic_particle_d.push(cast(d));
                    elastic_particle_c.push(cast(c.x_axis));
                    elastic_particle_c.push(cast(c.y_axis));
                    particle_f.push(cast(f.x_axis));
                    particle_f.push(cast(f.y_axis));
                    elastic_particle_mass.push(mass);
                }
                Matter::Liquid { liquid_density } => {
                    liquid_particle_x.push(cast(x));
                    liquid_particle_d.push(cast(d));
                    liquid_particle_c.push(cast(c.x_axis));
                    liquid_particle_c.push(cast(c.y_axis));
                    particle_desired_density.push(liquid_density);
                    liquid_particle_mass.push(mass);
                }
            }
        }
        Ok(Self {
            num_particles,
            grid_dx: Buffer::zeros(device.clone(), GRID_FLAT_LENGTH)?,
            grid_dy: Buffer::zeros(device.clone(), GRID_FLAT_LENGTH)?,
            grid_mass: Buffer::zeros(device.clone(), GRID_FLAT_LENGTH)?,
            liquids: LiquidStorage {
                particle_x: Buffer::from_vec(liquid_particle_x).into_device(device.clone())?,
                particle_d: Buffer::from_vec(liquid_particle_d).into_device(device.clone())?,
                particle_c: Buffer::from_vec(liquid_particle_c).into_device(device.clone())?,
                particle_desired_density: Buffer::from_vec(particle_desired_density)
                    .into_device(device.clone())?,
                particle_mass: Buffer::from_vec(liquid_particle_mass)
                    .into_device(device.clone())?,
            },
            elastics: ElasticStorage {
                particle_x: Buffer::from_vec(elastic_particle_x).into_device(device.clone())?,
                particle_d: Buffer::from_vec(elastic_particle_d).into_device(device.clone())?,
                particle_c: Buffer::from_vec(elastic_particle_c).into_device(device.clone())?,
                particle_f: Buffer::from_vec(particle_f).into_device(device.clone())?,
                particle_mass: Buffer::from_vec(elastic_particle_mass)
                    .into_device(device.clone())?,
            },
            device,
        })
    }
    // TODO https://gafferongames.com/post/fix_your_timestep/
    pub fn step(&mut self, dt: Real, iterations: usize) -> krnl::anyhow::Result<()> {
        if self.num_particles == 0 {
            return Ok(());
        };
        if self.liquids.particle_x.len() == 0 {
            return Ok(());
        }
        #[cfg(feature = "gpu")]
        {
            for _ in 0..iterations {
                // println!("Iteration {i}");
                kernels::liquid_inner_loop::builder()?
                    .specialize(GRID_FLAT_LENGTH as u32)
                    // DONT CHANGE THREAD COUNT
                    .with_threads(64)
                    .build(self.device.clone())?
                    .with_global_threads(self.num_particles)
                    .dispatch(
                        self.liquids.particle_x.as_slice(),
                        self.liquids.particle_d.as_slice_mut(),
                        self.liquids.particle_mass.as_slice(),
                        self.liquids.particle_c.as_slice_mut(),
                        self.liquids.particle_desired_density.as_slice(),
                        self.grid_dx.as_slice_mut(),
                        self.grid_dy.as_slice_mut(),
                        self.grid_mass.as_slice_mut(),
                    )?;
            }

            kernels::liquid_integrate::builder()?
                .build(self.device.clone())?
                .with_global_threads(self.num_particles)
                .dispatch(
                    dt,
                    self.liquids.particle_x.as_slice_mut(),
                    self.liquids.particle_d.as_slice_mut(),
                    self.liquids.particle_desired_density.as_slice_mut(),
                    self.liquids.particle_c.as_slice(),
                )?;
        }
        #[cfg(not(feature = "gpu"))]
        {
            for _ in 0..iterations {
                self.grid_dx = [0; GRID_FLAT_LENGTH];
                self.grid_dy = [0; GRID_FLAT_LENGTH];
                self.grid_mass = [0; GRID_FLAT_LENGTH];

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
                    &mut self.grid_dx,
                    &mut self.grid_dy,
                    &mut self.grid_mass,
                );
                Self::p2g(
                    &self.elastics.particle_x,
                    &self.elastics.particle_d,
                    &self.elastics.particle_c,
                    &self.elastics.particle_mass,
                    &mut self.grid_dx,
                    &mut self.grid_dy,
                    &mut self.grid_mass,
                );

                // 2. calculate grid displacements
                Self::update_grid(&self.grid_mass, &mut self.grid_dx, &mut self.grid_dy);

                // 3. grid-to-particle (G2P).
                // (grid is not mutated here, but g2p_step takes an unsafe slice to work on for the gpu
                // impl, so we must also comply)
                Self::g2p(
                    &mut self.grid_dx,
                    &mut self.grid_dy,
                    &self.liquids.particle_x,
                    &mut self.liquids.particle_d,
                    &mut self.liquids.particle_c,
                );
                Self::g2p(
                    &mut self.grid_dx,
                    &mut self.grid_dy,
                    &self.elastics.particle_x,
                    &mut self.elastics.particle_d,
                    &mut self.elastics.particle_c,
                );
            }

            // 4. integrate our values to update our particle positions and deformation gradients
            Self::liquid_integrate(
                dt,
                cast_slice(
                    self.liquids
                        .particle_c
                        .as_slice_mut()
                        .as_host_slice_mut()
                        .unwrap(),
                ),
                cast_slice_mut(
                    self.liquids
                        .particle_x
                        .as_slice_mut()
                        .as_host_slice_mut()
                        .unwrap(),
                ),
                cast_slice_mut(
                    self.liquids
                        .particle_d
                        .as_slice_mut()
                        .as_host_slice_mut()
                        .unwrap(),
                ),
                self.liquids
                    .particle_desired_density
                    .as_slice_mut()
                    .as_host_slice_mut()
                    .unwrap(),
            );
            Self::elastic_integrate(
                dt,
                cast_slice(
                    self.elastics
                        .particle_c
                        .as_slice_mut()
                        .as_host_slice_mut()
                        .unwrap(),
                ),
                cast_slice_mut(
                    self.elastics
                        .particle_x
                        .as_slice_mut()
                        .as_host_slice_mut()
                        .unwrap(),
                ),
                cast_slice_mut(
                    self.elastics
                        .particle_d
                        .as_slice_mut()
                        .as_host_slice_mut()
                        .unwrap(),
                ),
                cast_slice_mut(
                    self.elastics
                        .particle_f
                        .as_slice_mut()
                        .as_host_slice_mut()
                        .unwrap(),
                ),
            );
        }
        Ok(())
    }
    fn elastic_constraints(particle_f: &[Matrix], particle_c: &mut [Matrix]) {
        particle_c
            .par_iter_mut()
            .zip_eq(particle_f)
            .for_each(|(c, f)| kernels::elastic_constrain(c, f))
    }
    fn liquid_constraints(particle_desired_density: &[Real], particle_c: &mut [Matrix]) {
        particle_c
            .par_iter_mut()
            .zip_eq(particle_desired_density)
            .for_each(|(c, &liquid_density)| kernels::liquid_constrain(c, liquid_density))
    }

    fn p2g(
        particle_x: &[Vector],
        particle_d: &[Vector],
        particle_c: &[Matrix],
        particle_mass: &[Real],
        grid_dx: &mut [FixedPoint],
        grid_dy: &mut [FixedPoint],
        grid_mass: &mut [FixedPoint],
    ) {
        // here we modify over the whole grid, but they are given through shared references which
        // is valid since we will only use atomic operations on the grid
        let grid_mass: UnsafeSlice<FixedPoint> = grid_mass.into();
        let grid_dx: UnsafeSlice<FixedPoint> = grid_dx.into();
        let grid_dy: UnsafeSlice<FixedPoint> = grid_dy.into();
        particle_x
            .par_iter()
            .zip_eq(particle_d)
            .zip_eq(particle_c)
            .zip_eq(particle_mass)
            .for_each(|(((&x, &d), &c), &mass)| {
                kernels::p2g_step(grid_dx, grid_dy, grid_mass, x, d, c, mass)
            })
    }
    fn update_grid(
        grid_mass: &[FixedPoint; GRID_FLAT_LENGTH],
        grid_dx: &mut [FixedPoint; GRID_FLAT_LENGTH],
        grid_dy: &mut [FixedPoint; GRID_FLAT_LENGTH],
    ) {
        grid_dx
            .par_iter_mut()
            .zip_eq(grid_dy.par_iter_mut())
            .enumerate()
            .zip_eq(grid_mass)
            .for_each(|((i, (dx, dy)), mass)| kernels::grid_update_at(i, mass, dx, dy))
    }
    fn g2p(
        grid_dx: &mut [FixedPoint],
        grid_dy: &mut [FixedPoint],
        particle_x: &[Vector],
        particle_d: &mut [Vector],
        particle_c: &mut [Matrix],
    ) {
        let grid_dx: UnsafeSlice<FixedPoint> = grid_dx.into();
        let grid_dy: UnsafeSlice<FixedPoint> = grid_dy.into();
        particle_d
            .par_iter_mut()
            .zip_eq(particle_c.par_iter_mut())
            .zip_eq(particle_x)
            .for_each(|((d, c), &x)| kernels::g2p_step(grid_dx, grid_dy, d, c, x))
    }
    fn liquid_integrate(
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
                kernels::liquid_integrate_step(dt, x, d, liquid_density, c)
            })
    }
    fn elastic_integrate(
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
            .for_each(|(((x, d), f), c)| kernels::elastic_integrate_step(dt, x, d, f, c))
    }
}
