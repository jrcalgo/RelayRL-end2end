# Changelog

All notable changes to the `gridworld-rl` crate are documented in this file.

## [0.1.0] - 2026-03-08

### Added

- Core 2D GridWorld environment: `GridWorldEnv<B>` with configurable grid size, walls, end position, and Burn backend.
- Reward configuration via `RewardConfig` (collision, end-state, step rewards) with `Default`.
- Multi-agent support: configurable actor count, per-actor positions, observations, and rewards.
- Integration with `relayrl_framework`: `EnvironmentTrainingTrait` implementation, observations and returns as Burn tensors.
- Optional **render** feature (iced-based GUI), gated by the `render` feature flag.
