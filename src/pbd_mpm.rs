use glam::*;
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

const GRAVITY: Real = -0.3;
const FRICTION: Real = 0.5;
const GRID_LOWER_BOUND: Vector = Vector::ONE;
const GRID_UPPER_BOUND: Vector = Vector::new(GRID_WIDTH as Real - 2.0, GRID_HEIGHT as Real - 2.0);

const LIQUID_RELAXATION: Real = 1.5;
const LIQUID_VISCOSITY: Real = 0.01;

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
            Self::elastic_constraints(
                &self.elastics.particle_f,
                &mut self.elastics.particle_c,
            );

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
            .zip_eq(particle_f.par_iter())
            .for_each(|(c, &f)| {
                let f_star = (*c + Matrix::IDENTITY) * f;

                let cdf = clamped_det(f_star);
                let vol = f_star / (cdf.abs().sqrt() * cdf.signum());
                let (u, _, vt) = svd2x2(f_star);
                let shape = vt * u;
                debug_assert_eq!(shape.determinant().round(), 1.0);

                // also called elasticity ratio
                let alpha = 1.0;
                let interp_term = alpha * shape + (1.0 - alpha) * vol;

                let elastic_relaxation = 1.5;
                let diff = (interp_term * f.inverse() - Matrix::IDENTITY) - *c;
                *c += elastic_relaxation * diff;
            })
    }
    pub fn liquid_constraints(particle_desired_density: &[Real], particle_c: &mut [Matrix]) {
        particle_c
            .par_iter_mut()
            .zip_eq(particle_desired_density)
            .for_each(|(c, &liquid_density)| {
                let deviatoric = -(*c + c.transpose());
                *c += (LIQUID_VISCOSITY * 0.5) * deviatoric;
                // Volume preservation constraint:
                // we want to generate hydrostatic impulses with the form alpha*I
                // and we want the liquid volume integration (see particleIntegrate) to yield 1 = (1+tr(alpha*I + D))*det(F) at the end of the timestep.
                // where det(F) is stored as particle.liquidDensity.
                // Rearranging, we get the below expression that drives the deformation displacement towards preserving the volume.
                let alpha = 0.5 * (1.0 / liquid_density - trace(c) - 1.0);
                *c += (LIQUID_RELAXATION * alpha) * Matrix::IDENTITY;
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
        // is valid since we will only use atomic stores
        particle_x
            .par_iter()
            .zip_eq(particle_d)
            .zip_eq(particle_c)
            .zip_eq(particle_mass)
            .for_each(|(((&x, &d), &c), &mass)| {
                let cell_idx = x.as_uvec2();
                let weights = interpolated_weights(x);

                for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                    let weight = weights[gx].x * weights[gy].y;

                    let cell_x = UVec2::new(cell_idx.x + gx as u32 - 1, cell_idx.y + gy as u32 - 1);
                    let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - x) + 0.5;

                    let weighted_mass = weight * mass;
                    let momentum = weighted_mass * (d + c * cell_dist);

                    let idx = cell_x.x as usize + cell_x.y as usize * GRID_WIDTH;
                    use std::sync::atomic::Ordering::*;
                    grid_mass[idx].fetch_add(encode_fixed_point(weighted_mass), Relaxed);
                    grid_dx[idx].fetch_add(encode_fixed_point(momentum.x), Relaxed);
                    grid_dy[idx].fetch_add(encode_fixed_point(momentum.y), Relaxed);
                }
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
            .filter(|(_, mass)| unsafe { decode_fixed_point(mass.as_ptr().read()) } > 1e-5)
            .for_each(|((i, (dx, dy)), mass)| {
                // calculate grid displacement based on momentum found in the P2G stage
                unsafe {
                    let d = Vector::new(
                        decode_fixed_point(dx.as_ptr().read()),
                        decode_fixed_point(dy.as_ptr().read()),
                    ) / decode_fixed_point(mass.as_ptr().read());
                    dx.as_ptr().write(encode_fixed_point(d.x));
                    dy.as_ptr().write(encode_fixed_point(d.y));
                };
                // enforce boundary conditions
                let x = i % GRID_WIDTH;
                let y = i / GRID_WIDTH;
                if x < GRID_LOWER_BOUND.x as usize + 1 || x > GRID_UPPER_BOUND.x as usize - 1 {
                    unsafe {
                        dx.as_ptr().write(0);
                    }
                }
                if y < GRID_LOWER_BOUND.y as usize + 1 || y > GRID_UPPER_BOUND.y as usize - 1 {
                    unsafe {
                        dy.as_ptr().write(0);
                    }
                }
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
                // reset particle velocity. we calculate it from scratch each step using the grid
                *d = Vector::ZERO;
                *c = Matrix::ZERO;
                // 4.1: calculate neighboring cell weights as in step 2.1
                // note: our particle's haven't moved on the grid at all by this point, so the weights
                // will be identical
                let cell_idx = x.floor().as_uvec2();
                let weights = interpolated_weights(x);
                // constructing affine per-particle momentum matrix from APIC / MLS-MPM.
                // APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf)
                // below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
                // where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
                let mut b = Matrix::ZERO;
                for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                    let weight = weights[gx].x * weights[gy].y;

                    let cell_x =
                        UVec2::new((cell_idx.x + gx as u32) - 1, (cell_idx.y + gy as u32) - 1);
                    let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - x) + 0.5;
                    // This is safe because we dont modify these atomics in g2p
                    let grid_d = unsafe {
                        Vector::new(
                            decode_fixed_point(
                                grid_dx[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH]
                                    .as_ptr()
                                    .read(),
                            ),
                            decode_fixed_point(
                                grid_dy[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH]
                                    .as_ptr()
                                    .read(),
                            ),
                        )
                    };
                    let weighted_displacement = grid_d * weight;

                    // APIC paper equation 10, constructing inner term for B
                    let term = Matrix::from_cols(
                        weighted_displacement * cell_dist.x,
                        weighted_displacement * cell_dist.y,
                    );
                    b += term;
                    *d += weighted_displacement;
                }
                *c = b * 4.0;
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
                // advect particle positions by their displacement
                *x = (*x + *d).clamp(GRID_LOWER_BOUND, GRID_UPPER_BOUND);

                *liquid_density *= trace(c) + 1.0;
                *liquid_density = liquid_density.max(0.05);

                // Add gravity here so that it only indirectly affects particles through p2g
                // We are using dt only when we want to create forces (or possibly also impulses?)
                d.y -= GRAVITY * dt * dt;
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
                // advect particle positions by their displacement
                *x = (*x + *d).clamp(GRID_LOWER_BOUND, GRID_UPPER_BOUND);

                *f = (*c + Matrix::IDENTITY) * *f;

                let (u, mut sigma, vt) = svd2x2(*f);

                sigma = sigma.clamp(Vector::splat(0.1), Vector::splat(10000.0));

                *f = u * Matrix::from_diagonal(sigma) * vt;

                if x.x <= GRID_LOWER_BOUND.x + 2.0 || x.x >= GRID_UPPER_BOUND.x - 2.0 {
                    d.y = d.y.lerp(0.0, FRICTION);
                }
                if x.y <= GRID_LOWER_BOUND.y + 2.0 || x.y >= GRID_UPPER_BOUND.y - 2.0 {
                    d.x = d.x.lerp(0.0, FRICTION);
                }
                // Add gravity here so that it only indirectly affects particles through p2g
                // We are using dt only when we want to create forces (or possibly also impulses?)
                d.y -= GRAVITY * dt * dt;
            })
    }
}

// i think larger exponent is more accuracy?
// Why are we even using fixed point? There is no native atomic float, so we convert our float into
// fixed point so that we may use atomic addition with signed integers instead
const FIXED_POINT_MULTIPLIER: Real = 1e7;
fn decode_fixed_point(fixed: FixedPoint) -> Real {
    fixed as Real / FIXED_POINT_MULTIPLIER
}
fn encode_fixed_point(float: Real) -> FixedPoint {
    (float * FIXED_POINT_MULTIPLIER) as FixedPoint
}

pub fn interpolated_weights(x: Vector) -> [Vector; 3] {
    let cell_diff = (x - x.floor()) - 0.5;
    [
        0.5 * (0.5 - cell_diff) * (0.5 - cell_diff),
        0.75 - cell_diff * cell_diff,
        0.5 * (0.5 + cell_diff) * (0.5 + cell_diff),
    ]
}

pub fn particle_volume_estimate(
    p_x: Vector,
    p_mass: Real,
    grid_mass: &[Real; GRID_WIDTH * GRID_HEIGHT],
) -> Real {
    let cell_idx = p_x.as_uvec2();
    let weights = interpolated_weights(p_x);
    // estimating particle volume by summing up neighbourhood's weighted mass contribution
    // MPM course, equation 152
    let mut density = 0.0;
    for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
        let weight = weights[gx].x * weights[gy].y;
        let cell_index =
            (cell_idx.x as usize + gx - 1) + (cell_idx.y as usize + gy - 1) * GRID_WIDTH;
        density += grid_mass[cell_index] * weight;
    }
    p_mass / density
}

// sigma is given as a vector of the diagonal
fn svd2x2(m: Matrix) -> (Matrix, Vector, Matrix) {
    let e = (m.col(0).x + m.col(1).y) * 0.5;
    let f = (m.col(0).x - m.col(1).y) * 0.5;
    let g = (m.col(0).y + m.col(1).x) * 0.5;
    let h = (m.col(0).y - m.col(1).x) * 0.5;

    let q = f32::sqrt(e * e + h * h);
    let r = f32::sqrt(f * f + g * g);
    let sx = q + r;
    let sy = q - r;

    let a1 = f32::atan2(g, f);
    let a2 = f32::atan2(h, e);

    let theta = (a2 - a1) * 0.5;
    let phi = (a2 + a1) * 0.5;

    (
        Matrix::from_angle(phi),
        Vector::new(sx, sy),
        Matrix::from_angle(theta),
    )
}
fn trace(a: &Matrix) -> Real {
    a.col(0).x + a.col(1).y
}
fn clamped_det(a: Matrix) -> Real {
    const LOWER_DET_BOUND: Real = 0.1;
    const UPPER_DET_BOUND: Real = 1000.0;
    let det = a.determinant();
    det.signum() * det.abs().clamp(LOWER_DET_BOUND, UPPER_DET_BOUND)
}
