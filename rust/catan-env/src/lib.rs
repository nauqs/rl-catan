//! RL environment layer over catan-core.
//!
//! Built outward from the action codec: a fixed discrete action-id space
//! with an exact legality mask, the contract between the engine and any
//! policy network.

pub mod alpha;
pub mod codec;
pub mod env;
pub mod net;
pub mod obs;

pub use codec::{decode_action, encode_action, fill_action_mask, CODEC_VERSION, NUM_ACTIONS};
pub use env::{CatanEnv, EnvConfig, RewardConfig, StepResult, VecCatanEnv};
pub use obs::{encode_obs, Visibility, OBS_DIM, OBS_VERSION};
