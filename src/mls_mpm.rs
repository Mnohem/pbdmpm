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

const GRAVITY: Real = 0.3;

pub struct Particle {
    pub x: Vector,
    pub v: Vector,
    pub c: Matrix, // deformation displacement (D) when multiplied by dt
    pub f: Matrix, // deformation gradient
    pub mass: Real,
}
impl Particle {
    pub fn interpolated_weights(&self) -> [Vector; 3] {
        let cell_diff = (self.x - self.x.floor()) - 0.5;
        [
            0.5 * (0.5 - cell_diff) * (0.5 - cell_diff),
            0.75 - cell_diff * cell_diff,
            0.5 * (0.5 + cell_diff) * (0.5 + cell_diff),
        ]
    }
}
impl Default for Particle {
    fn default() -> Self {
        Self {
            x: Vector::ZERO,
            v: Vector::ZERO,
            c: Matrix::ZERO,
            f: Matrix::IDENTITY,
            mass: 1.0,
        }
    }
}

#[derive(Clone, Copy, Default)]
pub struct Cell {
    v: Vector,
    pub mass: Real,
}

pub struct Simulation {
    pub grid: [Cell; GRID_WIDTH * GRID_HEIGHT],
    pub particles: Vec<Particle>,
}
impl Default for Simulation {
    fn default() -> Self {
        Self {
            grid: [Cell::default(); GRID_WIDTH * GRID_HEIGHT],
            particles: vec![],
        }
    }
}

impl Simulation {
    pub fn step(&mut self) {
        // TODO eventually dt will be a parameter
        // https://gafferongames.com/post/fix_your_timestep/
        let dt = 0.1;
        // let iterations = 5;

        // 1. reset our scratch-pad grid completely
        self.grid.fill(Cell::default());

        // PBD MPM, basically working to reset our deformation gradient gradually
        self.update_constraints(dt);

        // 2. particle-to-grid P2G
        // goal: transfers data from particles to our grid
        // 2.1 Scatter mass to the grid so that we can get volume in full p2g step
        self.scatter_mass();

        // 2.2 P2G with volume calculated for each particle
        self.p2g(dt);

        // 3. calculate grid velocities
        for (i, cell) in self
            .grid
            .iter_mut()
            .enumerate()
            .filter(|(_, c)| c.mass.is_normal())
        {
            // note: before this step, "cell.v" refers to MOMENTUM, not velocity!
            // 3.1: calculate grid velocity based on momentum found in the P2G stage
            cell.v /= cell.mass;
            cell.v += dt * Vector::new(0.0, GRAVITY);
            debug_assert!(cell.v.is_finite());

            // 3.2: enforce boundary conditions
            let x = i % GRID_WIDTH;
            let y = i / GRID_WIDTH;
            if x < 2 || x > GRID_WIDTH - 3 {
                cell.v.x = 0.0
            }
            // y == 0 is a dirty fix so that if a particle touches the top it doesn't get stuck
            if y == 0 || y > GRID_HEIGHT - 3 {
                cell.v.y = 0.0
            }
        }

        // 3.3: Soften particle velocities near the boundaries
        let wall_min: Real = 3.0;
        let wall_max_x = (GRID_WIDTH - 1) as Real - wall_min;
        let wall_max_y = (GRID_HEIGHT - 1) as Real - wall_min;

        for p in self.particles.iter_mut() {
            // safely clamp particle positions to be inside the grid
            p.x = p.x.clamp(
                Vector::ONE,
                Vector::new(GRID_WIDTH as Real - 2.0, GRID_HEIGHT as Real - 2.0),
            );

            // predictive boundary conditions that soften velocities near the domain's edges
            let x_n = p.x + p.v;
            if x_n.x < wall_min {
                p.v.x += wall_min - x_n.x
            } else if x_n.x > wall_max_x {
                p.v.x += wall_max_x - x_n.x
            }
            if x_n.y < wall_min {
                p.v.y += wall_min - x_n.y
            } else if x_n.y > wall_max_y {
                p.v.y += wall_max_y - x_n.y
            }
        }

        // 4. grid-to-particle (G2P).
        // goal: report our grid's findings back to our particles, and integrate their position +
        // velocity forward
        self.g2p(dt);

        // 5. integrate our values to update our particle positions and deformation gradients
        self.integrate(dt);
    }
    pub fn update_constraints(&mut self, dt: Real) {
        for p in self.particles.iter_mut() {
            debug_assert!(p.c.is_finite());
            debug_assert!(p.f.is_finite());
            let f_star = (p.c * dt + Matrix::IDENTITY) * p.f;

            let cdf = clamped_det(f_star);
            let vol = f_star / (cdf.abs().sqrt() * cdf.signum());
            let shape = polar_rotation(f_star);

            // also called elasticity ratio
            let alpha = 1.0;
            let interp_term = alpha * shape + (1.0 - alpha) * vol;

            let elastic_relaxation = 1.5;
            let diff = (interp_term * p.f.inverse() - Matrix::IDENTITY) / dt - p.c;
            p.c += elastic_relaxation * diff;
        }
    }

    pub fn scatter_mass(&mut self) {
        for p in self.particles.iter() {
            let cell_idx = p.x.as_uvec2();
            let weights = p.interpolated_weights();

            for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                // scatter our particle's mass to the grid, using the cell's interpolation
                // weight calculated in 2.1
                let weight = weights[gx].x * weights[gy].y;

                let cell_x = UVec2::new(cell_idx.x + gx as u32 - 1, cell_idx.y + gy as u32 - 1);
                let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - p.x) + 0.5;
                let q = p.c * cell_dist;
                debug_assert!(q.is_finite());

                let cell = &mut self.grid[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH];
                let weighted_mass = weight * p.mass;
                cell.mass += weighted_mass;
                cell.v += weighted_mass * (p.v + q);
                // This is just the force portion, not momentum portion which is derived from
                // constitutive equations ie neo-hookean
            }
        }
    }
    pub fn p2g(&mut self, dt: Real) {
        let elastic_lambda = 10.0;
        let elastic_mu = 20.0;

        for p in self.particles.iter() {
            // 2.1: calculate weights for the 3x3 neighbouring cells surrounding the particle's
            // position on the grid using an interpolation function
            let cell_idx = p.x.as_uvec2();
            let weights = p.interpolated_weights();

            // estimating particle volume by summing up neighbourhood's weighted mass contribution
            // MPM course, equation 152
            let volume = {
                let mut density = 0.0;
                for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                    let weight = weights[gx].x * weights[gy].y;
                    let cell_index = (cell_idx.x as usize + gx - 1)
                        + (cell_idx.y as usize + gy - 1) * GRID_WIDTH;
                    density += self.grid[cell_index].mass * weight;
                }
                p.mass / density
            };

            // 2.2: calculate quantities like e.g. stress based on constitutive equation
            let stress = {
                let j = clamped_det(p.f);
                let f_t = p.f.transpose();
                // MPM course equation 48
                let p_term_0 = elastic_mu * (p.f * f_t - Matrix::IDENTITY);
                let p_term_1 = Matrix::from_diagonal(Vector::splat(elastic_lambda * j.ln()));
                let p_term = p_term_0 + p_term_1;

                println!("j: {j}, j.ln(): {}, stress: {}", j.ln(), p_term / j);
                debug_assert!(j.ln().is_finite());
                // cauchy_stress = (1 / det(F)) * P * F_T
                // equation 38, MPM course
                // already incorporated F_T into p_terms to cancel out F_T_inv
                p_term / j
            };

            // (Mp)^-1 = 4, see APIC paper and MPM course page 42
            // this term is used in the MLS-MPM paper last equation with quadratic weights,
            // Mp = (1/4) * (delta_x)^2
            // in this simulation, delta_x = 1, because i scale the rendering of the domain rather
            // than the domain itself
            // we multiply by dt as part of the process of fusing the momentum and force update
            let last_eq_term_0 = -volume * 4.0 * stress * dt;

            // 2.3:
            for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                // scatter our particle's momentum to the grid, using the cell's interpolation
                // weight calculated in 2.1
                let weight = weights[gx].x * weights[gy].y;

                let cell_x = UVec2::new(cell_idx.x + gx as u32 - 1, cell_idx.y + gy as u32 - 1);
                let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - p.x) + 0.5;

                let cell = &mut self.grid[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH];
                let momentum = (last_eq_term_0 * weight) * cell_dist;
                cell.v += momentum;
                // note: currently "cell.v" refers to MOMENTUM, not velocity!
                // this gets corrected in the UpdateGrid step
            }
        }
    }
    pub fn g2p(&mut self, dt: Real) {
        for p in self.particles.iter_mut() {
            // reset particle velocity. we calculate it from scratch each step using the grid
            p.v = Vector::ZERO;
            p.c = Matrix::ZERO;
            // 4.1: calculate neighboring cell weights as in step 2.1
            // note: our particle's haven't moved on the grid at all by this point, so the weights
            // will be identical
            let cell_idx = p.x.floor().as_uvec2();
            let weights = p.interpolated_weights();
            // constructing affine per-particle momentum matrix from APIC / MLS-MPM.
            // APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf)
            // below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
            // where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
            for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                let weight = weights[gx].x * weights[gy].y;

                let cell_x = UVec2::new((cell_idx.x + gx as u32) - 1, (cell_idx.y + gy as u32) - 1);
                let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - p.x) + 0.5;
                let weighted_vel =
                    self.grid[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH].v * weight;
                debug_assert!(
                    !self.grid[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH]
                        .v
                        .is_nan()
                );
                debug_assert!(!weighted_vel.is_nan());
                debug_assert!(weighted_vel.is_finite());

                // APIC paper equation 10, constructing inner term for B
                let term =
                    Matrix::from_cols(weighted_vel * cell_dist.x, weighted_vel * cell_dist.y);

                p.v += weighted_vel;
                p.c += term * 4.0;
            }
        }
    }
    pub fn integrate(&mut self, dt: Real) {
        for p in self.particles.iter_mut() {
            // advect particle positions by their velocity
            debug_assert!(p.v.is_finite());
            p.x += p.v * dt;

            // safety clamp to ensure particles don't exit simulation domain
            p.x = p.x.clamp(
                Vector::ONE,
                Vector::new(GRID_WIDTH as Real - 2.0, GRID_HEIGHT as Real - 2.0),
            );

            // deformation gradient update - MPM course, equation 181
            // Fp' = (I + dt * p.C) * Fp
            p.f *= dt * p.c + Matrix::IDENTITY;
        }
    }
}

fn polar_rotation(a: Matrix) -> Matrix {
    let inner = a + clamped_det(a).abs() * a.transpose().inverse();
    inner / clamped_det(inner).abs().sqrt()
}

fn clamped_det(a: Matrix) -> Real {
    const LOWER_DET_BOUND: Real = 0.1;
    const UPPER_DET_BOUND: Real = 1000.0;
    let det = a.determinant();
    det.signum() * det.abs().clamp(LOWER_DET_BOUND, UPPER_DET_BOUND)
}
