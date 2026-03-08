//! A Rust rewrite of the GridWorld environment commonly used as a Reinforcement Learning environment and benchmark.
//! 
//! This crate provides a 2D GridWorld simulation environment with user-defined rewards, agent counts, wall positions, start positions, and end position.
//! 
//! Uses `relayrl_framework` for the environment template traits and re-exported `Burn` tensors.
//! 
//! By default, the environment runs without rendering. To enable rendering with `iced` or to simply access the gui-related components for your own use-case, enable the `render` feature flag.
//! 
pub mod env;

#[cfg(feature = "render")]
pub mod render;