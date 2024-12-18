use glam::*;

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

const GRAVITY: Real = -0.3;
const FRICTION: Real = 0.5;
const GRID_LOWER_BOUND: Vector = Vector::ONE;
const GRID_UPPER_BOUND: Vector = Vector::new(GRID_WIDTH as Real - 2.0, GRID_HEIGHT as Real - 2.0);

pub struct Particle {
    pub x: Vector,
    pub d: Vector,
    pub c: Matrix, // deformation displacement (D) when multiplied by dt
    pub f: Matrix, // deformation gradient
    pub mass: Real,
}
impl Particle {}
impl Default for Particle {
    fn default() -> Self {
        Self {
            x: Vector::ZERO,
            d: Vector::ZERO,
            c: Matrix::ZERO,
            f: Matrix::IDENTITY,
            mass: 1.0,
        }
    }
}

pub struct Simulation {
    grid_d: [Vector; GRID_WIDTH * GRID_HEIGHT],
    grid_mass: [Real; GRID_WIDTH * GRID_HEIGHT],
    pub particle_x: Vec<Vector>,
    particle_d: Vec<Vector>,
    particle_c: Vec<Matrix>,
    particle_f: Vec<Matrix>,
    particle_mass: Vec<Real>,
}
impl Default for Simulation {
    fn default() -> Self {
        Self {
            grid_d: [Vector::ZERO; GRID_WIDTH * GRID_HEIGHT],
            grid_mass: [0.0; GRID_WIDTH * GRID_HEIGHT],
            particle_x: vec![],
            particle_d: vec![],
            particle_c: vec![],
            particle_f: vec![],
            particle_mass: vec![],
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
        for &Particle { x, d, c, f, mass } in particles.iter() {
            sim.particle_x.push(x);
            sim.particle_d.push(d);
            sim.particle_c.push(c);
            sim.particle_f.push(f);
            sim.particle_mass.push(mass);
        }
        sim
    }
    // TODO https://gafferongames.com/post/fix_your_timestep/
    pub fn step(&mut self, dt: Real, iterations: usize) {
        for _ in 0..iterations {
            // 1. reset our scratch-pad grid completely
            self.grid_d.fill(Vector::ZERO);
            self.grid_mass.fill(0.0);

            // PBD MPM, basically working to reset our deformation gradient gradually
            Self::update_constraints(&self.particle_f, &mut self.particle_c);

            // 2. particle-to-grid P2G
            Self::p2g(
                &self.particle_x,
                &self.particle_d,
                &self.particle_c,
                &self.particle_mass,
                &mut self.grid_mass,
                &mut self.grid_d,
            );

            // 3. calculate grid displacements
            Self::update_grid(&self.grid_mass, &mut self.grid_d);

            // 4. grid-to-particle (G2P).
            Self::g2p(
                &self.grid_d,
                &self.particle_x,
                &mut self.particle_d,
                &mut self.particle_c,
            );
        }

        // 5. integrate our values to update our particle positions and deformation gradients
        Self::integrate(
            dt,
            &self.particle_c,
            &mut self.particle_x,
            &mut self.particle_d,
            &mut self.particle_f,
        );
    }
    pub fn update_constraints(particle_f: &[Matrix], particle_c: &mut [Matrix]) {
        for (c, &f) in particle_c.iter_mut().zip(particle_f.iter()) {
            debug_assert!(c.is_finite());
            debug_assert!(f.is_finite());
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
        }
    }

    pub fn p2g(
        particle_x: &[Vector],
        particle_d: &[Vector],
        particle_c: &[Matrix],
        particle_mass: &[Real],
        grid_mass: &mut [Real; GRID_WIDTH * GRID_HEIGHT],
        grid_d: &mut [Vector; GRID_WIDTH * GRID_HEIGHT],
    ) {
        for (((&x, &d), &c), &mass) in particle_x
            .iter()
            .zip(particle_d.iter())
            .zip(particle_c.iter())
            .zip(particle_mass.iter())
        {
            let cell_idx = x.as_uvec2();
            let weights = interpolated_weights(x);

            // estimating particle volume by summing up neighbourhood's weighted mass contribution
            // MPM course, equation 152
            // let volume = {
            //     let mut density = 0.0;
            //     for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
            //         let weight = weights[gx].x * weights[gy].y;
            //         let cell_index = (cell_idx.x as usize + gx - 1)
            //             + (cell_idx.y as usize + gy - 1) * GRID_WIDTH;
            //         density += self.grid[cell_index].mass * weight;
            //     }
            //     p.mass / density
            // };

            for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                let weight = weights[gx].x * weights[gy].y;

                let cell_x = UVec2::new(cell_idx.x + gx as u32 - 1, cell_idx.y + gy as u32 - 1);
                let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - x) + 0.5;

                let weighted_mass = weight * mass;
                let momentum = weighted_mass * (d + c * cell_dist);

                grid_mass[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH] += weighted_mass;
                grid_d[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH] += momentum;
            }
        }
    }
    pub fn update_grid(
        grid_mass: &[Real; GRID_WIDTH * GRID_HEIGHT],
        grid_d: &mut [Vector; GRID_WIDTH * GRID_HEIGHT],
    ) {
        // boundary conditions are done differently
        for ((i, d), mass) in grid_d
            .iter_mut()
            .enumerate()
            .zip(grid_mass.iter())
            .filter(|(_, &mass)| mass > 1e-5)
        {
            // calculate grid displacement based on momentum found in the P2G stage
            *d /= mass;

            // enforce boundary conditions
            let x = i % GRID_WIDTH;
            let y = i / GRID_WIDTH;
            if x < GRID_LOWER_BOUND.x as usize + 1 || x > GRID_UPPER_BOUND.x as usize - 1 {
                d.x = 0.0
            }
            if y < GRID_LOWER_BOUND.y as usize + 1 || y > GRID_UPPER_BOUND.y as usize - 1 {
                d.y = 0.0
            }
        }
    }
    pub fn g2p(
        grid_d: &[Vector; GRID_WIDTH * GRID_HEIGHT],
        particle_x: &[Vector],
        particle_d: &mut [Vector],
        particle_c: &mut [Matrix],
    ) {
        for ((d, c), &x) in particle_d
            .iter_mut()
            .zip(particle_c.iter_mut())
            .zip(particle_x.iter())
        {
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

                let cell_x = UVec2::new((cell_idx.x + gx as u32) - 1, (cell_idx.y + gy as u32) - 1);
                let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - x) + 0.5;
                let weighted_displacement =
                    grid_d[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH] * weight;

                // APIC paper equation 10, constructing inner term for B
                let term = Matrix::from_cols(
                    weighted_displacement * cell_dist.x,
                    weighted_displacement * cell_dist.y,
                );
                b += term;
                *d += weighted_displacement;
            }
            *c = b * 4.0;
        }
    }
    pub fn integrate(
        dt: Real,
        particle_c: &[Matrix],
        particle_x: &mut [Vector],
        particle_d: &mut [Vector],
        particle_f: &mut [Matrix],
    ) {
        for (((x, d), f), &c) in particle_x
            .iter_mut()
            .zip(particle_d.iter_mut())
            .zip(particle_f.iter_mut())
            .zip(particle_c.iter())
        {
            // deformation gradient update - MPM course, equation 181
            // Fp' = (I + p.C) * Fp
            *f = (c + Matrix::IDENTITY) * *f;

            let (u, mut sigma, vt) = svd2x2(*f);

            sigma = sigma.clamp(Vector::splat(0.1), Vector::splat(10000.0));

            *f = u * Matrix::from_diagonal(sigma) * vt;

            // advect particle positions by their displacement
            *x = (*x + *d).clamp(GRID_LOWER_BOUND, GRID_UPPER_BOUND);

            // safely clamp particle positions to be inside the grid
            if x.x <= GRID_LOWER_BOUND.x + 2.0 || x.x >= GRID_UPPER_BOUND.x - 2.0 {
                d.y = d.y.lerp(0.0, FRICTION);
            }
            if x.y <= GRID_LOWER_BOUND.y + 2.0 || x.y >= GRID_UPPER_BOUND.y - 2.0 {
                d.x = d.x.lerp(0.0, FRICTION);
            }
            // *d = d.lerp(Vector::ZERO, DAMPING);
            // Add gravity here so that it only indirectly affects particles through p2g
            // We are using dt only when we want to create forces (or possibly also impulses?)
            d.y -= GRAVITY * dt * dt;
        }
    }
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

pub fn interpolated_weights(x: Vector) -> [Vector; 3] {
    let cell_diff = (x - x.floor()) - 0.5;
    [
        0.5 * (0.5 - cell_diff) * (0.5 - cell_diff),
        0.75 - cell_diff * cell_diff,
        0.5 * (0.5 + cell_diff) * (0.5 + cell_diff),
    ]
}

fn clamped_det(a: Matrix) -> Real {
    const LOWER_DET_BOUND: Real = 0.1;
    const UPPER_DET_BOUND: Real = 1000.0;
    let det = a.determinant();
    det.signum() * det.abs().clamp(LOWER_DET_BOUND, UPPER_DET_BOUND)
}
