//! AlphaZero-lite: the trained network inside the search loop.
//!
//! Root: one trunk forward on the current obs; the POLICY head ranks the
//! legal candidates and only the top `root_k` are searched (AlphaGo's prior
//! pruning). Per candidate: `samples` determinized short random rollouts
//! (`depth` turns — cheap chance averaging), each leaf scored by the VALUE
//! head from the acting seat's perspective (or the true ±1 if the game
//! ended). Highest mean wins.
//!
//! This fixes exactly what the plateau experiments isolated: the reactive
//! policy's one forward pass becomes a planner, and random-playout noise
//! becomes a trained evaluation.

use std::sync::Arc;

use catan_core::game::{Action, CatanGame};
use catan_core::players::Player;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::codec::encode_action;
use crate::net::{MlpNet, NetScratch};
use crate::obs::{encode_obs, Visibility, OBS_DIM};

pub struct AlphaBot {
    net: Arc<MlpNet>,
    rng: SmallRng,
    pub root_k: usize,
    pub samples: usize,
    pub depth: u32,
    obs: Vec<f32>,
    scratch: NetScratch,
    sim_valid: Vec<Action>,
    ranked: Vec<(f32, Action)>,
}

impl AlphaBot {
    pub fn new(seed: u64, net: Arc<MlpNet>, root_k: usize, samples: usize, depth: u32) -> AlphaBot {
        let scratch = NetScratch::new(&net);
        AlphaBot {
            net,
            rng: SmallRng::seed_from_u64(seed),
            root_k,
            samples,
            depth,
            obs: vec![0.0; OBS_DIM],
            scratch,
            sim_valid: Vec::with_capacity(256),
            ranked: Vec::with_capacity(256),
        }
    }

    /// Random playout, then value-head leaf evaluation.
    ///
    /// `depth == 0` is the in-distribution mode: play out only until `me` is
    /// next to act with a real choice — exactly the state class the value
    /// head trained on — and judge there. `depth > 0` plays that many turns
    /// before evaluating (off-distribution; kept for experiments).
    fn rollout_value(&mut self, base: &CatanGame, action: &Action, me: usize) -> f32 {
        let mut sim = base.clone();
        sim.record_history = false;
        if !sim.execute_action(action) {
            return -1.0;
        }
        let cap = if self.depth == 0 {
            1000
        } else {
            (sim.state.turn + self.depth).min(1000)
        };
        let mut acted = 0u32;
        while !sim.is_game_over() && sim.state.turn < cap {
            sim.fill_valid_actions(&mut self.sim_valid);
            if self.sim_valid.is_empty() {
                break;
            }
            if self.depth == 0
                && acted > 0
                && sim.current_player() == me
                && self.sim_valid.len() >= 2
            {
                break; // my next real decision: in-distribution leaf
            }
            let a = self.sim_valid[self.rng.gen_range(0..self.sim_valid.len())];
            sim.execute_action(&a);
            acted += 1;
        }
        let winner = sim.winner();
        if winner >= 0 {
            return if winner as usize == me { 1.0 } else { -1.0 };
        }
        if sim.state.turn >= 1000 {
            return 0.0;
        }
        encode_obs(&sim, me, Visibility::Perfect, &mut self.obs);
        self.net.trunk(&self.obs, &mut self.scratch);
        self.net.value_from(&self.scratch)
    }
}

impl Player for AlphaBot {
    fn choose_action(&mut self, game: &CatanGame, valid_actions: &[Action]) -> Action {
        let me = game.current_player();
        let candidates: Vec<Action> = {
            let filtered: Vec<Action> = valid_actions
                .iter()
                .copied()
                .filter(|a| !matches!(a, Action::ProposeTrade { .. }))
                .collect();
            if filtered.is_empty() {
                valid_actions.to_vec()
            } else {
                filtered
            }
        };
        if candidates.len() == 1 {
            return candidates[0];
        }

        // Root prior: rank candidates by policy logit, search only the top K.
        encode_obs(game, me, Visibility::Perfect, &mut self.obs);
        self.net.trunk(&self.obs, &mut self.scratch);
        self.ranked.clear();
        for &action in &candidates {
            let id = encode_action(game, &action);
            self.ranked.push((self.net.logit_from(&self.scratch, id), action));
        }
        self.ranked
            .sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let k = self.root_k.min(self.ranked.len());

        let searched: Vec<Action> = self.ranked[..k].iter().map(|r| r.1).collect();
        let mut best = searched[0];
        let mut best_score = f32::MIN;
        for action in searched {
            let mut total = 0.0;
            for _ in 0..self.samples {
                total += self.rollout_value(game, &action, me);
            }
            let score = total / self.samples as f32;
            if score > best_score {
                best_score = score;
                best = action;
            }
        }
        best
    }
}
