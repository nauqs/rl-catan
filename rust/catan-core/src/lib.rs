//! Catan game engine — a Rust port of the Python `engine/` package.
//!
//! Pure rules logic, no I/O or transport. Behavior matches the Python engine
//! exactly (verified by differential replay tests against recorded games).

pub mod board;
pub mod building;
pub mod dev_cards;
pub mod eval;
pub mod game;
pub mod players;
pub mod replay;
pub mod resources;
pub mod robber;
pub mod state;
pub mod trading;
