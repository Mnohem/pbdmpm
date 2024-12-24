pub mod frame;
pub mod pbd_mpm;

pub const WINDOW_WIDTH: usize = 512;
pub const WINDOW_HEIGHT: usize = 512;

pub const DOWN_SCALE: usize = 2;

pub const CANVAS_WIDTH: usize = WINDOW_WIDTH / DOWN_SCALE;
pub const CANVAS_HEIGHT: usize = WINDOW_HEIGHT / DOWN_SCALE;
