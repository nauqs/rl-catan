//! Position evaluation ("how good is this game state for player p?") and a
//! 1-ply greedy lookahead player built on it.
//!
//! Unlike `HeuristicPlayer` (which scores *actions* with local rules) this
//! scores *states*: the bot simulates every legal action on a cloned game and
//! picks the one whose resulting position evaluates best. True minimax does
//! not apply to 4-player stochastic Catan, so opponent awareness lives inside
//! the evaluation instead: the searched objective is
//! `V(me) - opp_lambda * max V(opponent)`, which makes robber placement,
//! blocking and discard choices fall out of one function.
//!
//! All weights are collected in `ValueParams` with to/from_array so the GA in
//! catan-sim can evolve them exactly like `HeuristicParams`.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::board::topology;
use crate::game::{Action, CatanGame};
use crate::players::Player;
use crate::state::{GameState, DEV_VICTORY_POINT, NUM_DEV_CARD_TYPES, NUM_RESOURCES};

pub const NUM_VALUE_PARAMS: usize = 17;

/// Tunable weights of the state evaluation. Indexes in `to_array` order.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ValueParams {
    /// Per victory point (the dominant term).
    pub vp: f32,
    /// Per unit of expected income (probability-weighted pips) per resource:
    /// wheat, sheep, wood, brick, stone.
    pub income: [f32; NUM_RESOURCES],
    /// Per distinct resource produced.
    pub diversity: f32,
    /// Flat bonus for owning a 3:1 port.
    pub port_any: f32,
    /// Per unit of income in a resource with a matching 2:1 port.
    pub port_matched: f32,
    /// Per card for the first 2 copies of each resource in hand.
    pub hand: f32,
    /// Per card beyond 2 copies of a resource (diminishing returns; this gap
    /// is what makes 4:1 bank trades of a surplus look profitable).
    pub hand_excess: f32,
    /// Per card above 7 in hand (discard exposure).
    pub overflow: f32,
    /// Per held non-VP development card.
    pub dev_card: f32,
    /// Per knight played (largest-army race; the award itself is in VP).
    pub knight: f32,
    /// Per edge of current longest-road length.
    pub road_len: f32,
    /// Times (best + 0.5 * second-best) production probability over the
    /// vertices where a settlement could legally go right now.
    pub expansion: f32,
    /// How much the strongest opponent's position subtracts from ours.
    pub opp_lambda: f32,
}

impl Default for ValueParams {
    fn default() -> Self {
        ValueParams {
            vp: 1.0,
            income: [2.2, 1.6, 2.0, 2.0, 2.2],
            diversity: 0.15,
            port_any: 0.1,
            port_matched: 1.0,
            hand: 0.05,
            hand_excess: 0.01,
            overflow: 0.04,
            dev_card: 0.15,
            knight: 0.1,
            road_len: 0.05,
            expansion: 1.5,
            opp_lambda: 0.5,
        }
    }
}

impl ValueParams {
    pub fn to_array(self) -> [f32; NUM_VALUE_PARAMS] {
        [
            self.vp,
            self.income[0], self.income[1], self.income[2], self.income[3], self.income[4],
            self.diversity, self.port_any, self.port_matched,
            self.hand, self.hand_excess, self.overflow,
            self.dev_card, self.knight, self.road_len,
            self.expansion, self.opp_lambda,
        ]
    }

    pub fn from_array(a: [f32; NUM_VALUE_PARAMS]) -> Self {
        ValueParams {
            vp: a[0],
            income: [a[1], a[2], a[3], a[4], a[5]],
            diversity: a[6],
            port_any: a[7],
            port_matched: a[8],
            hand: a[9],
            hand_excess: a[10],
            overflow: a[11],
            dev_card: a[12],
            knight: a[13],
            road_len: a[14],
            expansion: a[15],
            opp_lambda: a[16],
        }
    }
}

/// Expected per-roll income of one player, robber-aware: for every owned
/// building, each adjacent producing tile adds its roll probability
/// (doubled for cities) unless the robber sits on it.
fn income_by_resource(state: &GameState, player: usize) -> [f32; NUM_RESOURCES] {
    let topo = topology();
    let mut income = [0f32; NUM_RESOURCES];
    let mut mask = state.occupied_mask;
    while mask != 0 {
        let v = mask.trailing_zeros() as usize;
        mask &= mask - 1;
        if state.settlement_owner(v) != player as i8 {
            continue;
        }
        let mult = if state.is_city(v) { 2.0 } else { 1.0 };
        for &tile in &topo.vertex_tiles[v] {
            if tile < 0 || tile as u8 == state.robber_tile {
                continue;
            }
            let t = tile as usize;
            let r = state.tile_resources[t];
            if r >= 0 && (r as usize) < NUM_RESOURCES {
                income[r as usize] +=
                    mult * topo.number_probabilities[state.tile_numbers[t] as usize];
            }
        }
    }
    income
}

/// Absolute strength of one player's position (no opponent term).
pub fn position_score(state: &GameState, player: usize, p: &ValueParams) -> f32 {
    let mut score = p.vp * state.calculate_victory_points(player) as f32;

    let income = income_by_resource(state, player);
    let mut distinct = 0;
    for r in 0..NUM_RESOURCES {
        if income[r] > 0.0 {
            distinct += 1;
        }
        score += p.income[r] * income[r];
        if state.port_resource[player][r] {
            score += p.port_matched * income[r];
        }
    }
    score += p.diversity * distinct as f32;
    if state.port_any[player] {
        score += p.port_any;
    }

    let mut total_cards = 0i16;
    for r in 0..NUM_RESOURCES {
        let have = state.resources[player][r];
        total_cards += have;
        score += p.hand * have.min(2) as f32 + p.hand_excess * (have - 2).max(0) as f32;
    }
    score -= p.overflow * (total_cards - 7).max(0) as f32;

    let held_dev: i8 = (0..NUM_DEV_CARD_TYPES)
        .filter(|&c| c != DEV_VICTORY_POINT)
        .map(|c| state.dev_cards[player][c])
        .sum();
    score += p.dev_card * held_dev as f32;
    score += p.knight * state.knights_played[player] as f32;
    score += p.road_len * state.road_lengths[player] as f32;

    // Expansion potential: legal settlement spots (own road, vertex empty,
    // distance rule), ignoring affordability — resources are transient,
    // board position is not. Value the two best spots.
    let topo = topology();
    let (mut best, mut second) = (0f32, 0f32);
    let mut spots =
        state.vertex_road_mask[player] & !state.occupied_mask & ((1u64 << 54) - 1);
    while spots != 0 {
        let v = spots.trailing_zeros() as usize;
        spots &= spots - 1;
        if state.occupied_mask & topo.neighbor_mask[v] != 0 {
            continue;
        }
        let prob = state.vertex_probability(v);
        if prob > best {
            second = best;
            best = prob;
        } else if prob > second {
            second = prob;
        }
    }
    if state.settlements_built[player] < state.max_settlements {
        score += p.expansion * (best + 0.5 * second);
    }

    score
}

/// Relative evaluation searched by `GreedyValuePlayer`:
/// my position minus `opp_lambda` times the strongest opponent's.
/// Decisive terminal bonus so winning moves always dominate.
pub fn evaluate_state(state: &GameState, player: usize, p: &ValueParams) -> f32 {
    if state.winner >= 0 {
        return if state.winner as usize == player {
            1000.0
        } else {
            -1000.0
        };
    }
    let mine = position_score(state, player, p);
    let best_opp = (0..state.num_players)
        .filter(|&q| q != player)
        .map(|q| position_score(state, q, p))
        .fold(f32::MIN, f32::max);
    mine - p.opp_lambda * best_opp
}

/// 1-ply greedy lookahead: simulate every legal action on a cloned game and
/// take the argmax of `evaluate_state` (random tie-break).
///
/// Known modeling shortcuts, deliberate for a sparring bot:
/// - `RollDice` is scored as the pre-roll state (cloning the game clones the
///   RNG, so simulating the roll would let the bot peek at the actual dice).
/// - `BuyDevCard` / `StealResource` simulation does reveal the true draw the
///   same way; their value terms dominate, so the leak is minor.
/// - Like the other bots it never initiates trade proposals (it does respond
///   to and confirm them, via lookahead).
pub struct GreedyValuePlayer {
    rng: SmallRng,
    pub params: ValueParams,
}

impl GreedyValuePlayer {
    pub fn new(seed: u64) -> GreedyValuePlayer {
        Self::with_params(seed, ValueParams::default())
    }

    pub fn with_params(seed: u64, params: ValueParams) -> GreedyValuePlayer {
        GreedyValuePlayer {
            rng: SmallRng::seed_from_u64(seed),
            params,
        }
    }
}

impl Player for GreedyValuePlayer {
    fn choose_action(&mut self, game: &CatanGame, valid_actions: &[Action]) -> Action {
        assert!(!valid_actions.is_empty(), "no valid actions");
        let me = game.current_player();
        let baseline = evaluate_state(&game.state, me, &self.params);

        let mut best = None;
        let mut best_score = f32::MIN;
        let mut ties = 0u32;
        for &action in valid_actions {
            if matches!(action, Action::ProposeTrade { .. }) {
                continue;
            }
            let score = if matches!(action, Action::RollDice { .. }) {
                baseline
            } else {
                let mut sim = game.clone();
                sim.record_history = false;
                if !sim.execute_action(&action) {
                    continue;
                }
                evaluate_state(&sim.state, me, &self.params)
            };
            if score > best_score {
                best_score = score;
                best = Some(action);
                ties = 1;
            } else if score == best_score {
                // Reservoir sampling: uniform pick among tied maxima.
                ties += 1;
                if self.rng.gen_range(0..ties) == 0 {
                    best = Some(action);
                }
            }
        }
        best.unwrap_or(valid_actions[0])
    }
}
