/// Visual smoke-test for the GridWorld renderer with multiple agents.
///
/// Run with:
///   cargo run --example render_test_multiagent --features render
///
/// Close the window to exit.
fn main() -> iced::Result {
    use gridworld_rl::render::{GridWorldState, run};

    let state = GridWorldState {
        rows: 10,
        cols: 10,
        wall_positions: vec![
            (2, 1), (2, 2), (2, 3), (2, 4),
            (3, 4), (4, 4), (5, 4), (6, 4), (7, 4),
            (2, 6), (2, 7), (2, 8),
        ],
        end_position: (9, 9),
        actor_positions: vec![(0, 0), (0, 9), (9, 0), (5, 5)],
    };

    run(state)
}
