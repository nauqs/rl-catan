//! PyO3 boundary: the batched Catan RL environment for Python trainers.
//!
//! Design: ONE Python call per training step for the whole batch. The GIL
//! is released while Rust steps all envs in parallel; results come back as
//! fresh NumPy arrays (one memcpy per buffer — microseconds against an NN
//! forward pass). Checkpoints must store `codec_version`/`obs_version` and
//! refuse to run against mismatched layouts.

use catan_env::env::SeatKind;
use catan_env::{
    CatanEnv, EnvConfig, RewardConfig, VecCatanEnv, Visibility, CODEC_VERSION, NUM_ACTIONS,
    OBS_DIM, OBS_VERSION,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

fn parse_seats(seats: Option<Vec<String>>, num_players: usize) -> PyResult<[SeatKind; 4]> {
    let mut kinds = [SeatKind::Policy; 4];
    if let Some(seats) = seats {
        if seats.len() != num_players {
            return Err(PyValueError::new_err(format!(
                "seats must list {num_players} entries"
            )));
        }
        for (i, s) in seats.iter().enumerate() {
            kinds[i] = match s.as_str() {
                "policy" => SeatKind::Policy,
                "heuristic" => SeatKind::HeuristicBot,
                "random" => SeatKind::RandomBot,
                "rollout" => SeatKind::RolloutBot,
                "heuristic_v2" => SeatKind::HeuristicV2Bot,
                other => {
                    return Err(PyValueError::new_err(format!(
                        "seat kind must be policy/heuristic/random/rollout, got '{other}'"
                    )))
                }
            };
        }
    }
    Ok(kinds)
}

fn parse_visibility(s: &str) -> PyResult<Visibility> {
    match s {
        "perfect" => Ok(Visibility::Perfect),
        "realistic" => Ok(Visibility::Realistic),
        other => Err(PyValueError::new_err(format!(
            "visibility must be 'perfect' or 'realistic', got '{other}'"
        ))),
    }
}

/// Batched vectorized Catan environment (see catan-env's VecCatanEnv).
#[pyclass]
struct VecEnv {
    inner: VecCatanEnv,
    obs: Vec<f32>,
    masks: Vec<bool>,
    seats: Vec<u32>,
    rewards: Vec<f32>,
    dones: Vec<bool>,
    terminals: Vec<f32>,
}

impl VecEnv {
    fn arrays<'py>(
        &self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<u32>>,
    )> {
        let n = self.inner.num_envs();
        let obs = PyArray1::from_slice_bound(py, &self.obs).reshape([n, OBS_DIM])?;
        let masks = PyArray1::from_slice_bound(py, &self.masks).reshape([n, NUM_ACTIONS])?;
        let seats = PyArray1::from_slice_bound(py, &self.seats);
        Ok((obs, masks, seats))
    }
}

#[pymethods]
impl VecEnv {
    #[new]
    #[pyo3(signature = (num_envs, num_players=4, victory_target=10, visibility="perfect",
                        vp_delta=0.0, win=1.0, loss=-1.0, auto_resolve_forced=true,
                        max_turns=1000, seed=0, seats=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        num_envs: usize,
        num_players: usize,
        victory_target: i32,
        visibility: &str,
        vp_delta: f32,
        win: f32,
        loss: f32,
        auto_resolve_forced: bool,
        max_turns: u32,
        seed: u64,
        seats: Option<Vec<String>>,
    ) -> PyResult<Self> {
        if num_envs == 0 {
            return Err(PyValueError::new_err("num_envs must be positive"));
        }
        if !(2..=4).contains(&num_players) {
            return Err(PyValueError::new_err("num_players must be 2-4"));
        }
        if !(3..=20).contains(&victory_target) {
            return Err(PyValueError::new_err("victory_target must be 3-20"));
        }
        let config = EnvConfig {
            num_players,
            victory_target,
            visibility: parse_visibility(visibility)?,
            reward: RewardConfig { win, loss, vp_delta },
            auto_resolve_forced,
            max_turns,
            seat_kinds: parse_seats(seats, num_players)?,
        };
        Ok(VecEnv {
            inner: VecCatanEnv::new(num_envs, config, seed),
            obs: vec![0.0; num_envs * OBS_DIM],
            masks: vec![false; num_envs * NUM_ACTIONS],
            seats: vec![0; num_envs],
            rewards: vec![0.0; num_envs],
            dones: vec![false; num_envs],
            terminals: vec![0.0; num_envs * 4],
        })
    }

    /// Current (obs[N, OBS_DIM], masks[N, NUM_ACTIONS], seats[N]) without
    /// stepping — prime the first batch with this.
    fn observe<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<u32>>,
    )> {
        let (inner, obs, masks, seats) =
            (&mut self.inner, &mut self.obs, &mut self.masks, &mut self.seats);
        py.allow_threads(|| inner.observe(obs, masks, seats));
        self.arrays(py)
    }

    /// Step every env. Returns (obs, masks, seats, rewards[N], dones[N],
    /// terminal_rewards[N, 4]); where dones[i], the obs/mask/seat already
    /// belong to env i's next auto-reset episode.
    #[allow(clippy::type_complexity)]
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: PyReadonlyArray1<'py, u32>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<bool>>,
        Bound<'py, PyArray1<u32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let n = self.inner.num_envs();
        let actions = actions.as_slice()?;
        if actions.len() != n {
            return Err(PyValueError::new_err(format!(
                "expected {n} actions, got {}",
                actions.len()
            )));
        }
        let actions: Vec<u32> = actions.to_vec();
        let (inner, obs, masks, seats, rewards, dones, terminals) = (
            &mut self.inner,
            &mut self.obs,
            &mut self.masks,
            &mut self.seats,
            &mut self.rewards,
            &mut self.dones,
            &mut self.terminals,
        );
        py.allow_threads(|| {
            inner.step_batch(&actions, obs, masks, seats, rewards, dones, terminals)
        });

        let (obs, masks, seats) = self.arrays(py)?;
        let rewards = PyArray1::from_slice_bound(py, &self.rewards);
        let dones = PyArray1::from_slice_bound(py, &self.dones);
        let terminals = PyArray1::from_slice_bound(py, &self.terminals).reshape([n, 4])?;
        Ok((obs, masks, seats, rewards, dones, terminals))
    }

    /// Record every episode as a CTRP replay (evaluation/video runs).
    fn enable_recording(&mut self) {
        self.inner.enable_recording();
    }

    /// (turns, winner, final_vp[4], hit_turn_cap) per episode finished
    /// since the last call — dashboard `game` events.
    fn take_episode_stats(&mut self) -> Vec<(u32, i8, [u8; 4], bool)> {
        self.inner.drain_episode_stats()
    }

    /// Harvest finished-episode CTRP replays as `bytes` objects.
    fn take_replays<'py>(&mut self, py: Python<'py>) -> Vec<Bound<'py, PyBytes>> {
        self.inner
            .drain_records()
            .into_iter()
            .map(|record| PyBytes::new_bound(py, &record.to_bytes()))
            .collect()
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.inner.num_envs()
    }
    #[getter]
    fn num_actions(&self) -> usize {
        NUM_ACTIONS
    }
    #[getter]
    fn obs_dim(&self) -> usize {
        OBS_DIM
    }
    #[getter]
    fn codec_version(&self) -> u32 {
        CODEC_VERSION
    }
    #[getter]
    fn obs_version(&self) -> u32 {
        OBS_VERSION
    }
}

/// Single-env handle for evaluation loops driving one game at a time.
#[pyclass]
struct Env {
    inner: CatanEnv,
}

#[pymethods]
impl Env {
    #[new]
    #[pyo3(signature = (num_players=4, victory_target=10, visibility="perfect", seed=0))]
    fn new(num_players: usize, victory_target: i32, visibility: &str, seed: u64) -> PyResult<Self> {
        let config = EnvConfig {
            num_players,
            victory_target,
            visibility: parse_visibility(visibility)?,
            ..EnvConfig::default()
        };
        Ok(Env { inner: CatanEnv::new(config, seed) })
    }

    fn reset(&mut self, seed: u64) {
        self.inner.reset(seed);
    }

    /// Returns (seat, reward, done, winner, terminal_rewards[4]).
    fn step(&mut self, action: usize) -> (usize, f32, bool, i8, [f32; 4]) {
        let r = self.inner.step(action);
        (r.seat, r.reward, r.done, r.winner, r.terminal_rewards)
    }

    fn obs<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let mut buf = vec![0.0f32; OBS_DIM];
        self.inner.write_obs(&mut buf);
        PyArray1::from_vec_bound(py, buf)
    }

    fn mask<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyArray1<bool>> {
        let mut buf = vec![false; NUM_ACTIONS];
        self.inner.write_mask(&mut buf);
        PyArray1::from_vec_bound(py, buf)
    }

    fn current_seat(&self) -> usize {
        self.inner.current_seat()
    }
}

#[pymodule]
fn catan_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<VecEnv>()?;
    m.add_class::<Env>()?;
    m.add("NUM_ACTIONS", NUM_ACTIONS)?;
    m.add("OBS_DIM", OBS_DIM)?;
    m.add("CODEC_VERSION", CODEC_VERSION)?;
    m.add("OBS_VERSION", OBS_VERSION)?;
    Ok(())
}
