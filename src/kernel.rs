use krnl::macros::module;

#[module]
pub mod pbd_mpm {
    const GRID_SIZE: usize = 64;
    pub const GRID_WIDTH: usize = GRID_SIZE;
    pub const GRID_HEIGHT: usize = GRID_SIZE;
    pub const GRID_FLAT_LENGTH: usize = GRID_WIDTH * GRID_HEIGHT;
    // CANVAS_WIDTH and CANVAS_HEIGHT must be larger than GRID_WIDTH and GRID_HEIGHT respectively
    // In the simulator the cell is a 1x1 square, this is the pixel width and height of a cell
    pub const CELL_WIDTH: usize = crate::canvas_dims::CANVAS_WIDTH / GRID_WIDTH;
    pub const CELL_HEIGHT: usize = crate::canvas_dims::CANVAS_HEIGHT / GRID_HEIGHT;

    pub type Vector = glam::Vec2;
    pub type Matrix = glam::Mat2;
    pub type Real = f32;
    pub type FixedPoint = i32;

    pub const GRID_LOWER_BOUND: Vector = Vector::ONE;
    pub const GRID_UPPER_BOUND: Vector =
        Vector::new(GRID_WIDTH as Real - 2.0, GRID_HEIGHT as Real - 2.0);

    pub const GRAVITY: Real = -0.3;
    pub const FRICTION: Real = 0.5;

    pub const LIQUID_RELAXATION: Real = 1.5;
    pub const LIQUID_VISCOSITY: Real = 0.01;
}

#[module]
pub mod kernels {
    #[cfg(not(target_arch = "spirv"))]
    use krnl::krnl_core;
    use krnl_core::buffer::{Slice, UnsafeIndex, UnsafeSlice};
    #[cfg(target_arch = "spirv")]
    use krnl_core::num_traits::*;
    use krnl_core::spirv_std::{arch::control_barrier, glam::*, memory::Scope, memory::Semantics};

    use super::pbd_mpm::{FixedPoint, Matrix, Real, Vector};
    use super::pbd_mpm::{
        FRICTION, GRAVITY, GRID_LOWER_BOUND, GRID_UPPER_BOUND, GRID_WIDTH, LIQUID_RELAXATION,
        LIQUID_VISCOSITY,
    };

    const G_NEIGHBORS: [[usize; 2]; 9] = [
        [0, 0],
        [0, 1],
        [0, 2],
        [1, 0],
        [1, 1],
        [1, 2],
        [2, 0],
        [2, 1],
        [2, 2],
    ];

    #[repr(transparent)]
    struct UnsafeVectorSlice<'a> {
        inner: UnsafeSlice<'a, u64>,
    }
    impl<'a> From<UnsafeSlice<'a, u64>> for UnsafeVectorSlice<'a> {
        fn from(slice: UnsafeSlice<'a, u64>) -> Self {
            Self { inner: slice }
        }
    }
    #[cfg_attr(not(target_arch = "spirv"), allow(dead_code))]
    impl UnsafeVectorSlice<'_> {
        fn len(&self) -> usize {
            self.inner.len()
        }
        unsafe fn load(&self, index: usize) -> Vector {
            reify_vec(*self.inner.unsafe_index(index))
        }
        unsafe fn store(&self, index: usize, vector: Vector) {
            *self.inner.unsafe_index_mut(index) =
                ((vector.x.to_bits() as u64) << 4) + vector.y.to_bits() as u64;
        }
    }
    #[repr(transparent)]
    struct UnsafeMatrixSlice<'a> {
        inner: UnsafeVectorSlice<'a>,
    }
    impl<'a> From<UnsafeVectorSlice<'a>> for UnsafeMatrixSlice<'a> {
        fn from(slice: UnsafeVectorSlice<'a>) -> Self {
            Self { inner: slice }
        }
    }
    impl<'a> From<UnsafeSlice<'a, u64>> for UnsafeMatrixSlice<'a> {
        fn from(slice: UnsafeSlice<'a, u64>) -> Self {
            Into::<UnsafeVectorSlice<'a>>::into(slice).into()
        }
    }
    #[cfg_attr(not(target_arch = "spirv"), allow(dead_code))]
    impl UnsafeMatrixSlice<'_> {
        fn len(&self) -> usize {
            self.inner.len() / 2
        }
        unsafe fn load(&self, item_id: usize) -> Matrix {
            let index = item_id * 2;
            Matrix::from_cols(self.inner.load(index), self.inner.load(index + 1))
        }
        unsafe fn store(&self, item_id: usize, matrix: Matrix) {
            let index = item_id * 2;
            self.inner.store(index, matrix.x_axis);
            self.inner.store(index + 1, matrix.y_axis);
        }
    }

    fn reify_mat(matrix_bag: Slice<u64>, item_id: usize) -> Matrix {
        Matrix::from_cols(
            reify_vec(matrix_bag[item_id * 2]),
            reify_vec(matrix_bag[item_id * 2 + 1]),
        )
    }
    fn reify_vec(f: u64) -> Vector {
        Vector::new(
            f32::from_bits((f >> 4) as u32),
            f32::from_bits((f & (u32::MAX as u64)) as u32),
        )
    }
    #[cfg_attr(not(target_arch = "spirv"), allow(dead_code))]
    unsafe fn total_barrier() {
        control_barrier::<
            { Scope::Workgroup as u32 },
            { Scope::Workgroup as u32 },
            { Semantics::UNIFORM_MEMORY.bits() },
        >()
    }

    #[allow(clippy::too_many_arguments)]
    #[krnl_core::macros::kernel]
    pub fn liquid_inner_loop<const GFL: u32>(
        #[global] x_bag: Slice<u64>,
        #[global] d_bag: UnsafeSlice<u64>,
        #[global] mass_bag: Slice<f32>,
        #[global] c_bag: UnsafeSlice<u64>,
        #[global] desired_density_bag: Slice<f32>,
        #[global] global_grid_dx: UnsafeSlice<i32>,
        #[global] global_grid_dy: UnsafeSlice<i32>,
        #[global] global_grid_mass: UnsafeSlice<i32>,
        #[group] scratch_grid_dx: UnsafeSlice<i32, { GFL as usize }>,
        #[group] scratch_grid_dy: UnsafeSlice<i32, { GFL as usize }>,
        #[group] scratch_grid_mass: UnsafeSlice<i32, { GFL as usize }>,
    ) {
        use super::pbd_mpm::GRID_FLAT_LENGTH;

        let item_id = kernel.global_id();
        let num_particles = x_bag.len();
        // when this is true, this threads particle calculations and grid calculations are both
        // used, when false, only the grid calculations are used
        let p_and_g_thread = item_id < num_particles;

        let d_bag: UnsafeVectorSlice = d_bag.into();
        let c_bag: UnsafeMatrixSlice = c_bag.into();

        let x = if p_and_g_thread {
            reify_vec(x_bag[item_id])
        } else {
            Vector::ZERO
        };
        let mut d: Vector = if p_and_g_thread {
            unsafe { d_bag.load(item_id) }
        } else {
            Vector::ZERO
        };
        let mass = if p_and_g_thread {
            mass_bag[item_id]
        } else {
            0.0
        };
        let mut c: Matrix = if p_and_g_thread {
            unsafe { c_bag.load(item_id) }
        } else {
            Matrix::ZERO
        };
        let desired_density = if p_and_g_thread {
            desired_density_bag[item_id]
        } else {
            0.0
        };

        // PBD MPM, basically working to reset our deformation gradient gradually
        liquid_constrain(&mut c, desired_density);

        // 1. particle-to-grid P2G
        p2g_step(
            scratch_grid_dx,
            scratch_grid_dy,
            scratch_grid_mass,
            x,
            d,
            c,
            mass,
        );

        // grid calculations start
        // each thread zeroes their section in the global grid
        // barrier
        // each group atomic adds their scratch grid to global grid
        //   each thread in the group is given a different section of their group local grid
        // each thread grid_updates their section in the global grid, same as zero section
        // barrier
        let dispatch_size = kernel.global_threads();
        let global_grid_section_size = if item_id < GRID_FLAT_LENGTH {
            (GRID_FLAT_LENGTH as usize / dispatch_size).max(1)
        } else {
            0
        };
        let global_chunk_start = item_id * global_grid_section_size;
        for i in global_chunk_start
            ..(global_chunk_start + global_grid_section_size).min(GRID_FLAT_LENGTH)
        {
            let idx = i + global_chunk_start;
            *unsafe { global_grid_dx.unsafe_index_mut(idx) } = 0;
            *unsafe { global_grid_dy.unsafe_index_mut(idx) } = 0;
            *unsafe { global_grid_mass.unsafe_index_mut(idx) } = 0;
        }

        // WE ARE GOING TO ASSUME GROUP_SIZE IS A DIVISOR OF THE GRID LENGTH, WE MUST ENSURE
        // THIS IN OUR KERNEL CONFIGURATION
        let group_size = kernel.threads();
        let group_grid_section_size = GRID_FLAT_LENGTH as usize / group_size;
        let group_chunk_start = kernel.thread_id() * group_grid_section_size;

        unsafe { total_barrier() };

        for i in group_chunk_start..(group_chunk_start + group_grid_section_size) {
            let idx = i + group_chunk_start;
            unsafe {
                atomic_add(
                    global_grid_dx.unsafe_index_mut(idx),
                    *scratch_grid_dx.unsafe_index(idx),
                );
                atomic_add(
                    global_grid_dy.unsafe_index_mut(idx),
                    *scratch_grid_dy.unsafe_index(idx),
                );
                atomic_add(
                    global_grid_mass.unsafe_index_mut(idx),
                    *scratch_grid_mass.unsafe_index(idx),
                );
            }
        }

        unsafe { total_barrier() };

        for i in global_chunk_start
            ..(global_chunk_start + global_grid_section_size).min(GRID_FLAT_LENGTH)
        {
            let idx = i + global_chunk_start;
            grid_update_at(
                idx,
                unsafe { global_grid_mass.unsafe_index(idx) },
                unsafe { global_grid_dx.unsafe_index_mut(idx) },
                unsafe { global_grid_dy.unsafe_index_mut(idx) },
            )
        }

        unsafe { total_barrier() };
        // grid calculations end

        // 3. grid-to-particle (G2P).
        g2p_step(global_grid_dx, global_grid_dy, &mut d, &mut c, x);

        if p_and_g_thread {
            unsafe { d_bag.store(item_id, d) };
            unsafe { c_bag.store(item_id, c) };
        }
    }

    #[allow(clippy::too_many_arguments)]
    #[krnl_core::macros::kernel]
    pub fn liquid_integrate(
        dt: f32,
        #[global] x_bag: UnsafeSlice<u64>,
        #[global] d_bag: UnsafeSlice<u64>,
        #[global] desired_density_bag: UnsafeSlice<f32>,
        #[global] c_bag: Slice<u64>,
    ) {
        let item_id = kernel.global_id();
        let num_particles = x_bag.len();
        // when this is true, this threads particle calculations and grid calculations are both
        // used, when false, only the grid calculations are used
        let p_and_g_thread = item_id < num_particles;

        let x_bag: UnsafeVectorSlice = x_bag.into();
        let d_bag: UnsafeVectorSlice = d_bag.into();
        let c: Matrix = reify_mat(c_bag, item_id);

        let mut x: Vector = if p_and_g_thread {
            unsafe { x_bag.load(item_id) }
        } else {
            Vector::ZERO
        };
        let mut d: Vector = if p_and_g_thread {
            unsafe { d_bag.load(item_id) }
        } else {
            Vector::ZERO
        };
        let mut desired_density = if p_and_g_thread {
            unsafe { *desired_density_bag.unsafe_index(item_id) }
        } else {
            0.0
        };

        liquid_integrate_step(dt, &mut x, &mut d, &mut desired_density, &c);

        if p_and_g_thread {
            unsafe { x_bag.store(item_id, x) };
            unsafe { d_bag.store(item_id, d) };
            unsafe { *desired_density_bag.unsafe_index_mut(item_id) = desired_density };
        }
    }

    pub fn elastic_constrain(c: &mut Matrix, f: &Matrix) {
        let f_star = (*c + Matrix::IDENTITY) * *f;

        let cdf = clamped_det(f_star);
        let vol = f_star.mul_scalar(1.0 / (cdf.abs().sqrt() * cdf.signum()));
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
    pub fn liquid_constrain(c: &mut Matrix, liquid_density: f32) {
        let deviatoric = -(*c + c.transpose());
        *c += (LIQUID_VISCOSITY * 0.5) * deviatoric;
        // Volume preservation constraint:
        // we want to generate hydrostatic impulses with the form alpha*I
        // and we want the liquid volume integration (see particleIntegrate) to yield 1 = (1+tr(alpha*I + D))*det(F) at the end of the timestep.
        // where det(F) is stored as particle.liquidDensity.
        // Rearranging, we get the below expression that drives the deformation displacement towards preserving the volume.
        let alpha = 0.5 * (1.0 / liquid_density - trace(c) - 1.0);
        *c += (LIQUID_RELAXATION * alpha) * Matrix::IDENTITY;
    }

    pub fn p2g_step(
        grid_dx: UnsafeSlice<FixedPoint>,
        grid_dy: UnsafeSlice<FixedPoint>,
        grid_mass: UnsafeSlice<FixedPoint>,
        x: Vector,
        d: Vector,
        c: Matrix,
        mass: Real,
    ) {
        let cell_idx = x.as_uvec2();
        let weights = interpolated_weights(x);

        #[allow(clippy::needless_range_loop)] // iters dont really work on spirv
        for i in 0..9 {
            let [gx, gy] = G_NEIGHBORS[i];
            let weight = weights[gx].x * weights[gy].y;

            let cell_x = UVec2::new(cell_idx.x + gx as u32 - 1, cell_idx.y + gy as u32 - 1);
            let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - x) + 0.5;

            let weighted_mass = weight * mass;
            let momentum = weighted_mass * (d + c * cell_dist);

            let idx = cell_x.x as usize + cell_x.y as usize * GRID_WIDTH;
            #[cfg(target_arch = "spirv")]
            unsafe {
                atomic_add(
                    grid_dx.unsafe_index_mut(idx),
                    encode_fixed_point(momentum.x),
                );
                atomic_add(
                    grid_dy.unsafe_index_mut(idx),
                    encode_fixed_point(momentum.y),
                );
                atomic_add(
                    grid_mass.unsafe_index_mut(idx),
                    encode_fixed_point(weighted_mass),
                );
            }
            #[cfg(not(target_arch = "spirv"))]
            unsafe {
                use core::sync::atomic::{AtomicI32, Ordering::Relaxed};
                AtomicI32::from_ptr(grid_dx.as_mut_ptr().add(idx))
                    .fetch_add(encode_fixed_point(momentum.x), Relaxed);
                AtomicI32::from_ptr(grid_dy.as_mut_ptr().add(idx))
                    .fetch_add(encode_fixed_point(momentum.y), Relaxed);
                AtomicI32::from_ptr(grid_mass.as_mut_ptr().add(idx))
                    .fetch_add(encode_fixed_point(weighted_mass), Relaxed);
            }
        }
    }

    pub fn grid_update_at(i: usize, mass: &FixedPoint, dx: &mut FixedPoint, dy: &mut FixedPoint) {
        // calculate grid displacement based on momentum found in the P2G stage
        let m = decode_fixed_point(*mass);
        let d = Vector::new(decode_fixed_point(*dx), decode_fixed_point(*dy))
            / if m > 1e-5 { m } else { 1.0 };
        *dx = encode_fixed_point(d.x);
        *dy = encode_fixed_point(d.y);
        // enforce boundary conditions
        let x = i % GRID_WIDTH;
        let y = i / GRID_WIDTH;
        if x < GRID_LOWER_BOUND.x as usize + 1 || x > GRID_UPPER_BOUND.x as usize - 1 {
            *dx = 0;
        }
        if y < GRID_LOWER_BOUND.y as usize + 1 || y > GRID_UPPER_BOUND.y as usize - 1 {
            *dy = 0;
        }
    }

    pub fn g2p_step(
        grid_dx: UnsafeSlice<FixedPoint>,
        grid_dy: UnsafeSlice<FixedPoint>,
        d: &mut Vector,
        c: &mut Matrix,
        x: Vector,
    ) {
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
        #[allow(clippy::needless_range_loop)] // iters dont really work on spirv
        for i in 0..9 {
            let [gx, gy] = G_NEIGHBORS[i];
            let weight = weights[gx].x * weights[gy].y;

            let cell_x = UVec2::new((cell_idx.x + gx as u32) - 1, (cell_idx.y + gy as u32) - 1);
            let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - x) + 0.5;

            let idx = cell_x.x as usize + cell_x.y as usize * GRID_WIDTH;
            // This is safe because we dont modify these atomics in g2p
            let grid_d = Vector::new(
                decode_fixed_point(*unsafe { grid_dx.unsafe_index(idx) }),
                decode_fixed_point(*unsafe { grid_dy.unsafe_index(idx) }),
            );
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
    }

    pub fn liquid_integrate_step(
        dt: Real,
        x: &mut Vector,
        d: &mut Vector,
        liquid_density: &mut Real,
        c: &Matrix,
    ) {
        // advect particle positions by their displacement
        *x = (*x + *d).clamp(GRID_LOWER_BOUND, GRID_UPPER_BOUND);

        *liquid_density *= trace(c) + 1.0;
        *liquid_density = liquid_density.max(0.05);

        // Add gravity here so that it only indirectly affects particles through p2g
        // We are using dt only when we want to create forces (or possibly also impulses?)
        d.y -= GRAVITY * dt * dt;
    }

    pub fn elastic_integrate_step(
        dt: Real,
        x: &mut Vector,
        d: &mut Vector,
        f: &mut Matrix,
        c: &Matrix,
    ) {
        // advect particle positions by their displacement
        *x = (*x + *d).clamp(GRID_LOWER_BOUND, GRID_UPPER_BOUND);

        *f = (*c + Matrix::IDENTITY) * *f;

        let (u, mut sigma, vt) = svd2x2(*f);

        sigma = sigma.clamp(Vector::splat(0.1), Vector::splat(10000.0));

        *f = u * Matrix::from_diagonal(sigma) * vt;

        if x.x <= GRID_LOWER_BOUND.x + 2.0 || x.x >= GRID_UPPER_BOUND.x - 2.0 {
            d.y = d.lerp(Vector::ZERO, FRICTION).y;
        }
        if x.y <= GRID_LOWER_BOUND.y + 2.0 || x.y >= GRID_UPPER_BOUND.y - 2.0 {
            d.x = d.lerp(Vector::ZERO, FRICTION).x;
        }
        // Add gravity here so that it only indirectly affects particles through p2g
        // We are using dt only when we want to create forces (or possibly also impulses?)
        d.y -= GRAVITY * dt * dt;
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

    fn interpolated_weights(x: Vector) -> [Vector; 3] {
        let cell_diff = (x - x.floor()) - 0.5;
        [
            0.5 * (0.5 - cell_diff) * (0.5 - cell_diff),
            0.75 - cell_diff * cell_diff,
            0.5 * (0.5 + cell_diff) * (0.5 + cell_diff),
        ]
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
    #[cfg(target_arch = "spirv")]
    unsafe fn atomic_add(n: &mut FixedPoint, y: FixedPoint) -> FixedPoint {
        krnl_core::spirv_std::arch::atomic_i_add::<
            FixedPoint,
            { Scope::Workgroup as u32 },
            { Semantics::NONE.bits() },
        >(&mut *n, y)
    }
}
