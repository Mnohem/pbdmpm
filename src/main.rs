#![deny(clippy::all)]

pub mod frame;
pub mod mls_mpm;

use error_iter::ErrorIter as _;
use log::error;
use pixels::{Error, Pixels, SurfaceTexture};
use tao::dpi::LogicalSize;
use tao::event::{Event, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop};
use tao::window::WindowBuilder;

const WINDOW_WIDTH: usize = 512;
const WINDOW_HEIGHT: usize = 512;

const DOWN_SCALE: usize = 2;

pub const CANVAS_WIDTH: usize = WINDOW_WIDTH / DOWN_SCALE;
pub const CANVAS_HEIGHT: usize = WINDOW_HEIGHT / DOWN_SCALE;

fn main() -> Result<(), Error> {
    env_logger::init();
    let event_loop = EventLoop::new();
    let window = {
        let size = LogicalSize::new(WINDOW_WIDTH as f64, WINDOW_HEIGHT as f64);
        WindowBuilder::new()
            .with_title("Hello Pixels/Tao")
            .with_inner_size(size)
            .with_min_inner_size(size)
            .build(&event_loop)
            .unwrap()
    };

    let mut pixels = {
        let window_size = window.inner_size();
        let surface_texture = SurfaceTexture::new(window_size.width, window_size.height, &window);
        Pixels::new(CANVAS_WIDTH as u32, CANVAS_HEIGHT as u32, surface_texture)?
    };
    let mut world = frame::World::init_box();

    event_loop.run(move |event, _, control_flow| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                // Close the window
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }

                // Resize the window
                WindowEvent::Resized(size) => {
                    if let Err(err) = pixels.resize_surface(size.width, size.height) {
                        log_error("pixels.resize_surface", err);
                        *control_flow = ControlFlow::Exit;
                    }
                }

                _ => {}
            },

            // Update internal state and request a redraw
            Event::MainEventsCleared => {
                // std::thread::sleep(std::time::Duration::from_millis(500));
                world.update();

                window.request_redraw();
            }

            // Draw the current frame
            Event::RedrawRequested(_) => {
                world.draw(unsafe { pixels.frame_mut().align_to_mut().1 });
                if let Err(err) = pixels.render() {
                    log_error("pixels.render", err);
                    *control_flow = ControlFlow::Exit;
                }
            }

            _ => {}
        }
    });
}

fn log_error<E: std::error::Error + 'static>(method_name: &str, err: E) {
    error!("{method_name}() failed: {err}");
    for source in err.sources().skip(1) {
        error!("  Caused by: {source}");
    }
}
