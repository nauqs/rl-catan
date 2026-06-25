//! AI players: random baseline and heuristic strategy. Decision logic is
//! allocation-free. Players see the whole `CatanGame` (needed to evaluate
//! open trade offers); the RL observation encoder will need the same view.

use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

use crate::board::topology;
use crate::game::{Action, CatanGame};
use crate::state::GameState;

pub trait Player {
    fn choose_action(&mut self, game: &CatanGame, valid_actions: &[Action]) -> Action;
    fn on_game_start(&mut self, _state: &GameState, _player_idx: usize) {}
    fn on_game_end(&mut self, _state: &GameState, _winner: i8) {}
}

/// Drive a game to completion with the given strategies.
pub fn play_game(game: &mut CatanGame, players: &mut [Box<dyn Player>]) -> i8 {
    assert_eq!(players.len(), game.state.num_players);
    for (i, p) in players.iter_mut().enumerate() {
        p.on_game_start(&game.state, i);
    }
    let mut valid: Vec<Action> = Vec::with_capacity(128);
    while !game.is_game_over() && game.state.turn < 1000 {
        let idx = game.current_player();
        game.fill_valid_actions(&mut valid);
        if valid.is_empty() {
            break;
        }
        let action = players[idx].choose_action(game, &valid);
        game.execute_action(&action);
    }
    let winner = game.winner();
    for p in players.iter_mut() {
        p.on_game_end(&game.state, winner);
    }
    winner
}

pub struct RandomPlayer {
    rng: SmallRng,
}

impl RandomPlayer {
    pub fn new(seed: u64) -> RandomPlayer {
        RandomPlayer {
            rng: SmallRng::seed_from_u64(seed),
        }
    }
}

impl Player for RandomPlayer {
    fn choose_action(&mut self, _game: &CatanGame, valid_actions: &[Action]) -> Action {
        valid_actions[self.rng.gen_range(0..valid_actions.len())]
    }
}

/// Heuristic priority tiers, lowest value wins. Within a tier the choice is
/// score-maximizing, first-match, or uniform-random depending on the tier.
#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Tier {
    InitialSettlement = 0,
    InitialRoad = 1,
    RollDice = 2,
    Discard = 3,
    Respond = 4,
    Confirm = 5,
    MoveRobber = 6,
    Steal = 7,
    BuildSettlement = 8,
    BuildCity = 9,
    BuildRoad = 10,
    BuyDevCard = 11,
    PlayKnight = 12,
    TradeWithBank = 13,
    EndTurn = 14,
    /// Proposing trades and everything else ranks below EndTurn: the
    /// heuristic never initiates negotiation (it does respond).
    Other = 15,
}

fn tier_of(action: &Action) -> Tier {
    match action {
        Action::PlaceInitialSettlement { .. } => Tier::InitialSettlement,
        Action::PlaceInitialRoad { .. } => Tier::InitialRoad,
        Action::RollDice { .. } => Tier::RollDice,
        Action::DiscardResource { .. } => Tier::Discard,
        Action::RespondTrade { .. } => Tier::Respond,
        Action::ConfirmTrade { .. } => Tier::Confirm,
        Action::MoveRobber { .. } => Tier::MoveRobber,
        Action::StealResource { .. } => Tier::Steal,
        Action::BuildSettlement { .. } => Tier::BuildSettlement,
        Action::BuildCity { .. } => Tier::BuildCity,
        Action::BuildRoad { .. } => Tier::BuildRoad,
        Action::BuyDevCard { .. } => Tier::BuyDevCard,
        Action::PlayKnight { .. } => Tier::PlayKnight,
        Action::TradeWithBank { .. } => Tier::TradeWithBank,
        Action::EndTurn { .. } => Tier::EndTurn,
        _ => Tier::Other,
    }
}

/// Tunable weights of the heuristic strategy. `Default` is EXACTLY the
/// frozen Heuristic-v1 behavior; evolved variants use `with_params`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeuristicParams {
    pub diversity_bonus: f32,
    pub port_bonus: f32,
    pub robber_steal_bonus: f32,
    pub robber_self_penalty: f32,
    pub steal_cards_weight: f32,
    pub trade_accept_threshold: f32,
}

impl Default for HeuristicParams {
    fn default() -> Self {
        HeuristicParams {
            diversity_bonus: 0.1,
            port_bonus: 0.2,
            robber_steal_bonus: 0.1,
            robber_self_penalty: 2.0,
            steal_cards_weight: 0.1,
            trade_accept_threshold: 3.0,
        }
    }
}

impl HeuristicParams {
    pub fn to_array(self) -> [f32; 6] {
        [self.diversity_bonus, self.port_bonus, self.robber_steal_bonus,
         self.robber_self_penalty, self.steal_cards_weight, self.trade_accept_threshold]
    }
    pub fn from_array(a: [f32; 6]) -> Self {
        HeuristicParams {
            diversity_bonus: a[0], port_bonus: a[1], robber_steal_bonus: a[2],
            robber_self_penalty: a[3], steal_cards_weight: a[4],
            trade_accept_threshold: a[5],
        }
    }
}

/// Heuristic-v2: weights evolved by the GA (catan-sim --evolve, 2026-06-11),
/// 39.9% holdout vs three v1s (fair share 25%). FROZEN like v1: it is an
/// Elo anchor; never retune in place — evolve a v3 instead.
pub const HEURISTIC_V2_PARAMS: HeuristicParams = HeuristicParams {
    diversity_bonus: 0.126_215_38,
    port_bonus: 0.113_224_65,
    robber_steal_bonus: 0.081_000_64,
    robber_self_penalty: 1.079_534_2,
    steal_cards_weight: 0.027_256_077,
    trade_accept_threshold: 2.810_156_3,
};

pub struct HeuristicPlayer {
    rng: SmallRng,
    params: HeuristicParams,
}

impl HeuristicPlayer {
    pub fn new(seed: u64) -> HeuristicPlayer {
        HeuristicPlayer {
            rng: SmallRng::seed_from_u64(seed),
            params: HeuristicParams::default(),
        }
    }

    /// The frozen evolved Heuristic-v2.
    pub fn v2(seed: u64) -> HeuristicPlayer {
        Self::with_params(seed, HEURISTIC_V2_PARAMS)
    }

    /// Evolved/tuned variant (Heuristic-v2 candidates).
    pub fn with_params(seed: u64, params: HeuristicParams) -> HeuristicPlayer {
        HeuristicPlayer {
            rng: SmallRng::seed_from_u64(seed),
            params,
        }
    }

    fn evaluate_vertex(&self, state: &GameState, vertex: usize) -> f32 {
        let topo = topology();
        let mut prob_score = 0.0;
        let mut diversity = [false; 5];
        for &tile in &topo.vertex_tiles[vertex] {
            if tile >= 0 {
                let t = tile as usize;
                let resource = state.tile_resources[t];
                if resource < 5 {
                    prob_score += topo.number_probabilities[state.tile_numbers[t] as usize];
                    diversity[resource as usize] = true;
                }
            }
        }
        let diversity_bonus =
            diversity.iter().filter(|&&d| d).count() as f32 * self.params.diversity_bonus;
        let port_bonus = if topo.vertex_port_index[vertex] >= 0 {
            self.params.port_bonus
        } else {
            0.0
        };
        prob_score + diversity_bonus + port_bonus
    }

    fn score(&self, game: &CatanGame, action: &Action) -> f32 {
        let state = &game.state;
        match *action {
            Action::PlaceInitialSettlement { vertex, .. }
            | Action::BuildSettlement { vertex, .. }
            | Action::BuildCity { vertex, .. } => self.evaluate_vertex(state, vertex as usize),
            Action::MoveRobber { tile, .. } => {
                let topo = topology();
                let tile = tile as usize;
                let player = state.current_player as i8;
                let prob = topo.number_probabilities[state.tile_numbers[tile] as usize];
                let mut score = 0.0;
                for &vertex in &topo.tile_vertices[tile] {
                    let owner = state.settlement_owner(vertex as usize);
                    if owner >= 0 && owner != player {
                        score += prob;
                        if state.total_resources(owner as usize) > 0 {
                            score += self.params.robber_steal_bonus;
                        }
                    } else if owner == player {
                        score -= prob * self.params.robber_self_penalty;
                    }
                }
                score
            }
            Action::StealResource { victim, .. } => {
                if victim < 0 {
                    return f32::NEG_INFINITY;
                }
                let v = victim as usize;
                state.calculate_victory_points(v) as f32
                    + state.total_resources(v) as f32 * self.params.steal_cards_weight
            }
            // Discard the most plentiful resource first.
            Action::DiscardResource { player, resource } => {
                state.resources[player as usize][resource as usize] as f32
            }
            // Accept a trade only with a comfortable surplus of what's asked.
            Action::RespondTrade { player, accept } => {
                let wants_to_accept = game.trade_offer.is_some_and(|o| {
                    state.resources[player as usize][o.recv as usize] as f32
                        >= self.params.trade_accept_threshold
                });
                if accept == wants_to_accept {
                    1.0
                } else {
                    0.0
                }
            }
            _ => 0.0,
        }
    }
}

impl Player for HeuristicPlayer {
    fn choose_action(&mut self, game: &CatanGame, valid_actions: &[Action]) -> Action {
        assert!(!valid_actions.is_empty(), "no valid actions");
        let state = &game.state;

        // Single pass: find the best (lowest) tier present and, for
        // score-based tiers, the best-scoring action within it.
        let mut best_tier = Tier::Other;
        let mut tier_first: Option<Action> = None;
        let mut tier_count = 0usize;
        let mut tier_best: Option<Action> = None;
        let mut tier_best_score = -1.0f32;

        // Roads are only built while under the settlement limit; otherwise
        // the tier is skipped entirely.
        let road_tier_active = state.settlements_built[state.current_player] < 5;

        for action in valid_actions {
            let tier = tier_of(action);
            if tier == Tier::BuildRoad && !road_tier_active {
                continue;
            }
            if tier < best_tier {
                best_tier = tier;
                tier_first = Some(*action);
                tier_count = 1;
                tier_best = None;
                tier_best_score = -1.0;
            } else if tier == best_tier {
                tier_count += 1;
            } else {
                continue;
            }
            let s = self.score(game, action);
            if s > tier_best_score {
                tier_best_score = s;
                tier_best = Some(*action);
            }
        }

        match best_tier {
            // Score-maximizing tiers (first action wins ties).
            Tier::InitialSettlement
            | Tier::MoveRobber
            | Tier::Steal
            | Tier::BuildSettlement
            | Tier::BuildCity
            | Tier::Discard
            | Tier::Respond => tier_best.or(tier_first).unwrap(),

            // First-match tiers (Confirm: partners are listed before cancel,
            // so the first acceptor is chosen).
            Tier::RollDice
            | Tier::Confirm
            | Tier::BuyDevCard
            | Tier::PlayKnight
            | Tier::EndTurn => tier_first.unwrap(),

            // Uniform-random tiers: pick the k-th action of the tier.
            Tier::InitialRoad | Tier::BuildRoad | Tier::TradeWithBank => {
                let k = self.rng.gen_range(0..tier_count);
                valid_actions
                    .iter()
                    .filter(|a| tier_of(a) == best_tier)
                    .nth(k)
                    .copied()
                    .unwrap()
            }

            // Fallback: uniform over everything (only trade proposals land
            // here, and EndTurn outranks them, so this is unreachable).
            Tier::Other => valid_actions[self.rng.gen_range(0..valid_actions.len())],
        }
    }
}

/// Flat Monte Carlo search ("how Go was first cracked, before the neural
/// nets"): for each candidate action, simulate N random playouts to the end
/// (or a turn horizon) and pick the action with the best average outcome.
/// Dice and branching are non-issues — the simulations just roll them.
/// Trade proposals are excluded as candidates (like the heuristic, it never
/// initiates negotiation; it does evaluate responses).
pub struct RolloutBot {
    rng: SmallRng,
    pub rollouts: usize,
    pub horizon: u32,
    sim_valid: Vec<Action>,
}

impl RolloutBot {
    pub fn new(seed: u64, rollouts: usize, horizon: u32) -> RolloutBot {
        RolloutBot {
            rng: SmallRng::seed_from_u64(seed),
            rollouts,
            horizon,
            sim_valid: Vec::with_capacity(256),
        }
    }

    /// Random playout from `base`, truncated at the turn horizon. Returns a
    /// score in [0, 1] for `me`: win = 1, loss = 0, truncation = smooth VP
    /// differential vs the best opponent.
    fn playout(&mut self, base: &CatanGame, me: usize) -> f32 {
        let mut sim = base.clone();
        sim.record_history = false;
        let cap = (sim.state.turn + self.horizon).min(1000);
        while !sim.is_game_over() && sim.state.turn < cap {
            sim.fill_valid_actions(&mut self.sim_valid);
            if self.sim_valid.is_empty() {
                break;
            }
            let action = self.sim_valid[self.rng.gen_range(0..self.sim_valid.len())];
            sim.execute_action(&action);
        }
        let winner = sim.winner();
        if winner >= 0 {
            return if winner as usize == me { 1.0 } else { 0.0 };
        }
        let my_vp = sim.state.calculate_victory_points(me) as f32;
        let best_other = (0..sim.state.num_players)
            .filter(|&p| p != me)
            .map(|p| sim.state.calculate_victory_points(p))
            .max()
            .unwrap_or(0) as f32;
        (0.5 + 0.08 * (my_vp - best_other)).clamp(0.0, 1.0)
    }
}

impl Player for RolloutBot {
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
        let mut best = candidates[0];
        let mut best_score = f32::MIN;
        for &action in &candidates {
            let mut base = game.clone();
            base.record_history = false;
            if !base.execute_action(&action) {
                continue;
            }
            let mut total = 0.0;
            for _ in 0..self.rollouts {
                total += self.playout(&base, me);
            }
            let score = total / self.rollouts as f32;
            if score > best_score {
                best_score = score;
                best = action;
            }
        }
        best
    }
}
