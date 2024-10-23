use glam::*;

const GRID_SIZE: usize = 64;
pub const GRID_WIDTH: usize = GRID_SIZE;
pub const GRID_HEIGHT: usize = GRID_SIZE;
// CANVAS_WIDTH and CANVAS_HEIGHT must be larger than GRID_WIDTH and GRID_HEIGHT respectively
// In the simulator the cell is a 1x1 square, this is the pixel width and height of a cell
pub const CELL_WIDTH: usize = crate::CANVAS_WIDTH / GRID_WIDTH;
pub const CELL_HEIGHT: usize = crate::CANVAS_HEIGHT / GRID_HEIGHT;
const GRAVITY: f32 = -0.3;

pub struct Particle {
    pub x: Vec2,
    pub v: Vec2,
    pub c: Mat2,
    pub f: Mat2, // deformation gradient
    pub mass: f32,
    pub initial_volume: f32,
}
impl Particle {
    pub fn interpolated_weights(&self) -> [Vec2; 3] {
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
            x: Vec2::ZERO,
            v: Vec2::ZERO,
            c: Mat2::ZERO,
            f: Mat2::IDENTITY,
            mass: 1.0,
            initial_volume: 0.0,
        }
    }
}
#[derive(Clone, Copy, Default)]
pub struct Cell {
    v: Vec2,
    pub mass: f32,
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

        // 1. reset our scratch-pad grid completely
        self.grid.fill(Cell::default());

        // 2. particle-to-grid P2G
        // goal: transfers data from particles to our grid
        self.p2g(dt);

        // 3. calculate grid velocities
        for (i, cell) in self
            .grid
            .iter_mut()
            .enumerate()
            .filter(|(_, c)| c.mass > 0.0)
        {
            // 3.1: calculate grid velocity based on momentum found in the P2G stage
            cell.v /= cell.mass;
            cell.v += dt * Vec2::new(0.0, GRAVITY);

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

        // 4. grid-to-particle (G2P).
        // goal: report our grid's findings back to our particles, and integrate their position +
        // velocity forward
        self.g2p(dt);
    }
    pub fn p2g(&mut self, dt: f32) {
        let elastic_lambda = 10.0;
        let elastic_mu = 20.0;

        for p in self.particles.iter() {
            // 2.1: calculate weights for the 3x3 neighbouring cells surrounding the particle's
            // position on the grid using an interpolation function
            let cell_idx = p.x.floor().as_uvec2();
            let weights = p.interpolated_weights();

            // 2.2: calculate quantities like e.g. stress based on constitutive equation
            let j = p.f.determinant();
            debug_assert!(j != 0.0); // A is invertible iff det(A) != 0
                                     // MPM course, page 46
            let volume = p.initial_volume * j;
            // useful matrices for Neo-Hookean model
            let f_t = p.f.transpose();
            debug_assert!(!f_t.is_nan());
            let f_t_inv = f_t.inverse();
            debug_assert!(!f_t_inv.is_nan());
            let f_minus_f_t_inv = p.f - f_t_inv;
            // MPM course equation 48
            let p_term_0 = elastic_mu * f_minus_f_t_inv;
            debug_assert!(!p_term_0.is_nan());
            let p_term_1 = elastic_lambda * j.ln() * f_t_inv;
            println!("j: {j}, ln(j): {}", j.ln());
            debug_assert!(!p_term_1.is_nan());
            let p_term = p_term_0 + p_term_1;
            // cauchy_stress = (1 / det(F)) * P * F_T
            // equation 38, MPM course
            let stress = (1.0 / j) * (p_term * f_t);
            debug_assert!(!stress.is_nan());
            // (Mp)^-1 = 4, see APIC paper and MPM course page 42
            // this term is used in MLS-MPM papare eq. 16 with quadratic weights,
            // Mp = (1/4) * (delta_x)^2
            // in this simulation, delta_x = 1, because i scale the rendering of the domain rather
            // than the domain itself
            // we multiply by dt as part of the process of fusing the momentum and force update
            let eq_16_term_0 = -volume * 4.0 * stress * dt;
            debug_assert!(!eq_16_term_0.is_nan());

            // 2.3:
            for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                // scatter our particle's momentum to the grid, using the cell's interpolation
                // weight calculated in 2.1
                let weight = weights[gx].x * weights[gy].y;

                let cell_x = UVec2::new(cell_idx.x + gx as u32 - 1, cell_idx.y + gy as u32 - 1);
                let cell_dist = (cell_x.as_vec2() - p.x) + 0.5;
                let q = p.c * cell_dist;

                let cell = &mut self.grid[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH];
                let weighted_mass = weight * p.mass;
                cell.mass += weighted_mass;
                // fused force/momentum update form MLS_MPM
                // see MLS_MPM paper, equation listed after eqn. 28
                let momentum = (eq_16_term_0 * weight) * cell_dist;
                debug_assert!(!momentum.is_nan());
                cell.v += weighted_mass * (p.v + q) + momentum;
                // note: currently "cell.v" refers to MOMENTUM, not velocity!
                // this gets corrected in the UpdateGrid step below.
            }
        }
    }
    pub fn g2p(&mut self, dt: f32) {
        for p in self.particles.iter_mut() {
            // reset particle velocity. we calculate it from scratch each step using the grid
            p.v = Vec2::ZERO;
            // TODO 4.1: update particle's deformation gradient using MLS-MPM's velocity estimate
            // Reference: MLS-MPM paper, Eq. 17
            //
            // 4.2: calculate neighboring cell weights as in step 2.1
            // note: our particle's haven't moved on the grid at all by this point, so the weights
            // will be identical
            let cell_idx = p.x.floor().as_uvec2();
            let weights = p.interpolated_weights();
            // constructing affine per-particle momentum matrix from APIC / MLS-MPM.
            // APIC paper (https://web.archive.org/web/20190427165435/https://www.math.ucla.edu/~jteran/papers/JSSTS15.pdf)
            // below equation 11 for clarification. this is calculating C = B * (D^-1) for APIC equation 8,
            // where B is calculated in the inner loop at (D^-1) = 4 is a constant when using quadratic interpolation functions
            let mut b: Mat2 = Mat2::ZERO;
            for (gx, gy) in (0..3).flat_map(|x| std::iter::repeat(x).zip(0..3)) {
                let weight = weights[gx].x * weights[gy].y;
                debug_assert!(!weight.is_nan());

                let cell_x = UVec2::new((cell_idx.x + gx as u32) - 1, (cell_idx.y + gy as u32) - 1);
                let cell_dist = (cell_x.as_vec2() - p.x) + 0.5;
                let weighted_vel =
                    self.grid[cell_x.x as usize + cell_x.y as usize * GRID_WIDTH].v * weight;
                debug_assert!(!weighted_vel.is_nan());

                // APIC paper equation 10, constructing inner term for B
                let term = Mat2::from_cols(weighted_vel * cell_dist.x, weighted_vel * cell_dist.y);

                b += term;
                p.v += weighted_vel;
            }
            // 4.4: advect particle positions by their velocity
            p.x += p.v * dt;

            // safety clamp to ensure particles don't exit simulation domain
            p.x = p.x.clamp(Vec2::ONE, Vec2::splat(GRID_SIZE as f32 - 2.0));

            debug_assert!(!b.is_nan());
            p.c = b * 4.0;
            debug_assert!((dt * p.c + Mat2::IDENTITY).determinant() > 0.0);
            // deformation gradient update - MPM course, equation 181
            // Fp' = (I + dt * p.C) * Fp
            debug_assert!(p.f.determinant() > 0.0);
            p.f *= dt * p.c + Mat2::IDENTITY;
            debug_assert!(!p.f.determinant().is_nan());
            debug_assert!(p.f.determinant() > 0.0);
        }
    }
}
