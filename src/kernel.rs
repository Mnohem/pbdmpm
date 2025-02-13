use crate::pbd_mpm::{Atomic, FixedPoint, Matrix, Real, Vector};
use crate::pbd_mpm::{FRICTION, LIQUID_RELAXATION, LIQUID_VISCOSITY, GRID_HEIGHT, GRID_WIDTH, GRID_LOWER_BOUND, GRID_UPPER_BOUND, GRAVITY};
use glam::{FloatExt, UVec2};

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

pub fn elastic_constrain(c: &mut Matrix, f: &Matrix) {
    let f_star = (*c + Matrix::IDENTITY) * *f;

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
    grid_mass: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
    grid_dx: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
    grid_dy: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
    x: Vector, d: Vector, c: Matrix, mass: Real
) {
    let cell_idx = x.as_uvec2();
    let weights = interpolated_weights(x);

    for [gx, gy] in G_NEIGHBORS {
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
}

pub fn grid_update_at(i: usize, mass: &Atomic, dx: &mut Atomic, dy: &mut Atomic) {
    // calculate grid displacement based on momentum found in the P2G stage
    unsafe {
        let m = decode_fixed_point(mass.as_ptr().read());
        let d = 
            Vector::new(
                decode_fixed_point(dx.as_ptr().read()),
                decode_fixed_point(dy.as_ptr().read()),
            ) / if m > 1e-5 {
                m
            } else {
                1.0
            };
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
}

pub fn g2p_step(
    grid_dx: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
    grid_dy: &[Atomic; GRID_WIDTH * GRID_HEIGHT],
    d: &mut Vector, c: &mut Matrix, x: Vector
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
    for [gx, gy] in G_NEIGHBORS {
        let weight = weights[gx].x * weights[gy].y;

        let cell_x =
            UVec2::new((cell_idx.x + gx as u32) - 1, (cell_idx.y + gy as u32) - 1);
        let cell_dist = (Into::<Vector>::into(cell_x.as_vec2()) - x) + 0.5;

        let idx = cell_x.x as usize + cell_x.y as usize * GRID_WIDTH;
        // This is safe because we dont modify these atomics in g2p
        let grid_d = unsafe {
            Vector::new(
                decode_fixed_point(grid_dx[idx].as_ptr().read()),
                decode_fixed_point(grid_dy[idx].as_ptr().read()),
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
}

pub fn liquid_integrate_step(dt: Real, x: &mut Vector, d: &mut Vector, liquid_density: &mut Real, c: &Matrix) {
    // advect particle positions by their displacement
    *x = (*x + *d).clamp(GRID_LOWER_BOUND, GRID_UPPER_BOUND);

    *liquid_density *= trace(c) + 1.0;
    *liquid_density = liquid_density.max(0.05);

    // Add gravity here so that it only indirectly affects particles through p2g
    // We are using dt only when we want to create forces (or possibly also impulses?)
    d.y -= GRAVITY * dt * dt;
}

pub fn elastic_integrate_step(dt: Real, x: &mut Vector, d: &mut Vector, f: &mut Matrix, c: &Matrix) {
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
}

// i think larger exponent is more accuracy?
// Why are we even using fixed point? There is no native atomic float, so we convert our float into
// fixed point so that we may use atomic addition with signed integers instead
const FIXED_POINT_MULTIPLIER: Real = 1e7;
pub fn decode_fixed_point(fixed: FixedPoint) -> Real {
    fixed as Real / FIXED_POINT_MULTIPLIER
}
pub fn encode_fixed_point(float: Real) -> FixedPoint {
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
pub fn svd2x2(m: Matrix) -> (Matrix, Vector, Matrix) {
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
pub fn trace(a: &Matrix) -> Real {
    a.col(0).x + a.col(1).y
}
pub fn clamped_det(a: Matrix) -> Real {
    const LOWER_DET_BOUND: Real = 0.1;
    const UPPER_DET_BOUND: Real = 1000.0;
    let det = a.determinant();
    det.signum() * det.abs().clamp(LOWER_DET_BOUND, UPPER_DET_BOUND)
}
