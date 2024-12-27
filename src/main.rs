#![deny(clippy::all)]

use error_iter::ErrorIter as _;
use log::{error, info};
use pixels::{Error, Pixels, SurfaceTexture};
use tao::dpi::{LogicalSize, PhysicalPosition};
use tao::event::{ElementState, Event, MouseButton, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop};
use tao::window::WindowBuilder;

use prepixx::{frame::World, CANVAS_HEIGHT, CANVAS_WIDTH, WINDOW_HEIGHT, WINDOW_WIDTH, pbd_mpm::Particle};

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
    let mut world = World::init_liquid_box();

    let mut place_particle = false;

    event_loop.run(move |event, event_loop_window_target, control_flow| {
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
                WindowEvent::MouseInput {
                    state: ElementState::Pressed,
                    button: MouseButton::Left,
                    ..
                } => place_particle = true,
                WindowEvent::MouseInput {
                    state: ElementState::Released,
                    button: MouseButton::Left,
                    ..
                } => place_particle = false,

                _ => {}
            },

            // Update internal state and request a redraw
            Event::MainEventsCleared => {
                // std::thread::sleep(std::time::Duration::from_millis(500));
                world.update();

                if place_particle {
                    let PhysicalPosition { x, y } = event_loop_window_target.cursor_position().unwrap();

                    info!("Cursor physical position at {x:.2}, {y:.2}");
                    // world.spawn(Particle {
                    //     x: Vector::new(x as f32, y as f32),
                    //     f: ConstrainedValue {
                    //         liquid_density: 1.0,
                    //     },
                    //     matter: Matter::Liquid,
                    //     ..Default::default()
                    // });
                }

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
