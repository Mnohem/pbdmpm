#![allow(unexpected_cfgs)]
pub mod frame;
pub mod kernel;
pub mod pbd_mpm;

use error_iter::ErrorIter as _;
use log::error;
use pixels::{Pixels, SurfaceTexture};
use winit::dpi::LogicalSize;
use winit::event::{ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ControlFlow, EventLoop};

use frame::World;
use glam::{UVec2, Vec2};
use kernel::pbd_mpm::{CELL_HEIGHT, CELL_WIDTH, GRID_HEIGHT, GRID_WIDTH};
use pbd_mpm::{Matter, Particle};

#[krnl::macros::module]
pub mod canvas_dims {
    pub const WINDOW_WIDTH: usize = 512;
    pub const WINDOW_HEIGHT: usize = 512;

    pub const DOWN_SCALE: usize = 2;

    pub const CANVAS_WIDTH: usize = WINDOW_WIDTH / DOWN_SCALE;
    pub const CANVAS_HEIGHT: usize = WINDOW_HEIGHT / DOWN_SCALE;
}
use canvas_dims::*;

fn main() -> Result<(), winit::error::EventLoopError> {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(World::init_liquid_box());
    event_loop.run_app(&mut app)
}

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        error!("  Caused by: {source}");
    }
}

use winit::application::ApplicationHandler;
use winit::event_loop::ActiveEventLoop;
use winit::window::{Window, WindowId};

struct App {
    window: Option<Window>,
    place_particle: bool,
    pixels: Option<Pixels>,
    world: World,
    cursor_position: UVec2,
}
impl App {
    fn new(world: World) -> Self {
        Self {
            place_particle: false,
            world,
            window: None,
            pixels: None,
            cursor_position: UVec2::ZERO,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let size = LogicalSize::new(WINDOW_WIDTH as f64, WINDOW_HEIGHT as f64);
        let window_attributes = Window::default_attributes()
            .with_title("Hello Pixels/winit")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .with_max_inner_size(size);
        self.window = event_loop.create_window(window_attributes).ok();
        self.pixels = {
            let window_size = self.window.as_ref().unwrap().inner_size();
            let surface_texture = SurfaceTexture::new(
                window_size.width,
                window_size.height,
                self.window.as_ref().unwrap(),
            );
            Pixels::new(CANVAS_WIDTH as u32, CANVAS_HEIGHT as u32, surface_texture).ok()
        };
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if let Err(err) = self
                    .pixels
                    .as_mut()
                    .unwrap()
                    .resize_surface(size.width, size.height)
                {
                    log_error("pixels.resize_surface", err);
                    event_loop.exit();
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = UVec2::new(
                    (position.x / DOWN_SCALE as f64) as u32,
                    (position.y / DOWN_SCALE as f64) as u32,
                );
            }
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button: MouseButton::Left,
                ..
            } => self.place_particle = true,
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => self.place_particle = false,
            WindowEvent::RedrawRequested => {
                self.world
                    .draw(unsafe { self.pixels.as_mut().unwrap().frame_mut().align_to_mut().1 });
                if let Err(err) = self.pixels.as_ref().unwrap().render() {
                    log_error("pixels.render", err);
                    event_loop.exit();
                }
            }

            _ => {}
        }
    }
    fn about_to_wait(&mut self, _: &ActiveEventLoop) {
        if self.place_particle {
            let sim_position =
                self.cursor_position.as_vec2() / Vec2::new(CELL_WIDTH as f32, CELL_HEIGHT as f32);
            if let (1..GRID_WIDTH, 1..GRID_HEIGHT) =
                (sim_position.x as usize, sim_position.y as usize)
            {
                self.world.spawn(Particle {
                    x: sim_position,
                    matter: Matter::Liquid {
                        liquid_density: 1.0,
                    },
                    ..Default::default()
                })
            }
        }

        self.world.update();

        self.window.as_ref().unwrap().request_redraw();
    }
}

#[cfg(test)]
mod tests {
    use krnl::{anyhow, buffer::Buffer, device::Device};

    use super::*;
    #[test]
    fn world_update() {
        let mut world = World::init_liquid_box();
        assert_eq!(
            world.simulation.num_particles,
            world.simulation.liquids.particle_x.len() as u32
        );
        world.update();
        assert_eq!(
            world.simulation.num_particles,
            world.simulation.liquids.particle_x.len() as u32
        );
    }
    #[test]
    fn world_update_and_draw() {
        let mut world = World::init_liquid_box();
        assert_eq!(
            world.simulation.num_particles,
            world.simulation.liquids.particle_x.len() as u32
        );
        world.update();
        assert_eq!(
            world.simulation.num_particles,
            world.simulation.liquids.particle_x.len() as u32
        );
        world.draw(&mut [0; CANVAS_WIDTH * CANVAS_HEIGHT]);
    }
    #[test]
    fn device_test() -> anyhow::Result<()> {
        let v = vec![1u64, 2, 3];
        let device = Device::builder().build()?;
        let buffer = Buffer::from_vec(v.clone()).into_device(device)?;
        assert_eq!(v.len(), buffer.len());
        Ok(())
    }
}
