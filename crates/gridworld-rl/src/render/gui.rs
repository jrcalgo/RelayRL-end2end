use iced::{
    mouse,
    widget::canvas::{self, Canvas, Frame, Path, Stroke},
    Element, Length, Point, Rectangle, Task, Theme,
};

use crate::render::stylesheet;

/// A backend-agnostic snapshot of the GridWorld state used for rendering.
///
/// Construct one manually or via [`GridWorldState::from_env`] (requires a live
/// [`crate::env::GridWorldEnv`]).
#[derive(Debug, Clone)]
pub struct GridWorldState {
    pub rows: usize,
    pub cols: usize,
    pub wall_positions: Vec<(usize, usize)>,
    pub end_position: (usize, usize),
    pub actor_positions: Vec<(usize, usize)>,
}

impl GridWorldState {
    pub fn from_env<B>(env: &crate::env::GridWorldEnv<B>) -> Self
    where
        B: relayrl_framework::prelude::tensor::burn::backend::Backend,
        B::Device: Clone,
    {
        GridWorldState {
            rows: env.length,
            cols: env.width,
            wall_positions: env
                .wall_positions
                .iter()
                .map(|&(r, c)| (r as usize, c as usize))
                .collect(),
            end_position: (env.end_position.0 as usize, env.end_position.1 as usize),
            actor_positions: env
                .get_actor_positions()
                .iter()
                .map(|&(r, c)| (r as usize, c as usize))
                .collect(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct GridWorldViewer {
    pub state: GridWorldState,
}

impl<Message> canvas::Program<Message> for GridWorldViewer {
    type State = ();

    fn draw(
        &self,
        _state: &Self::State,
        renderer: &iced::Renderer,
        _theme: &Theme,
        bounds: Rectangle,
        _cursor: mouse::Cursor,
    ) -> Vec<canvas::Geometry> {
        let mut frame = Frame::new(renderer, bounds.size());

        // Uniform tile size that fits the whole grid inside the canvas
        let tile_size = (bounds.width / self.state.cols as f32)
            .min(bounds.height / self.state.rows as f32);

        let offset_x = (bounds.width - tile_size * self.state.cols as f32) / 2.0;
        let offset_y = (bounds.height - tile_size * self.state.rows as f32) / 2.0;

        for row in 0..self.state.rows {
            for col in 0..self.state.cols {
                let x = offset_x + col as f32 * tile_size;
                let y = offset_y + row as f32 * tile_size;
                let pos = (row, col);
                let top_left = Point::new(x, y);
                let tile = iced::Size::new(tile_size, tile_size);

                if self.state.wall_positions.contains(&pos) {
                    // Black walls
                    frame.fill_rectangle(top_left, tile, stylesheet::WALL_COLOR);
                } else {
                    // White background, or red for the end cell.
                    let bg = if pos == self.state.end_position {
                        stylesheet::END_COLOR
                    } else {
                        stylesheet::TILE_BG_COLOR
                    };
                    frame.fill_rectangle(top_left, tile, bg);

                    // Black outline for all passable tiles.
                    let outline = Path::rectangle(top_left, tile);
                    frame.stroke(
                        &outline,
                        Stroke::default()
                            .with_color(stylesheet::TILE_OUTLINE_COLOR)
                            .with_width(stylesheet::TILE_OUTLINE_WIDTH),
                    );
                }
            }
        }

        for &(row, col) in &self.state.actor_positions {
            let cx = offset_x + col as f32 * tile_size + tile_size / 2.0;
            let cy = offset_y + row as f32 * tile_size + tile_size / 2.0;
            let radius = tile_size * stylesheet::AGENT_RADIUS_FRACTION;
            let circle = Path::circle(Point::new(cx, cy), radius);
            frame.fill(&circle, stylesheet::AGENT_COLOR);
        }

        vec![frame.into_geometry()]
    }
}

#[derive(Debug, Clone)]
enum Message {}

fn update(_viewer: &mut GridWorldViewer, msg: Message) -> Task<Message> {
    match msg {}
}

fn view(viewer: &GridWorldViewer) -> Element<Message> {
    Canvas::new(viewer.clone())
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

/// Open a blocking iced window that displays the provided [`GridWorldState`].
///
/// The function blocks until the window is closed.  Call it from a `#[test]`
/// or a thin binary wrapper.
pub fn run(state: GridWorldState) -> iced::Result {
    let initial = GridWorldViewer { state };
    iced::application(move || initial.clone(), update, view)
        .title("GridWorld Viewer")
        .run()
}

