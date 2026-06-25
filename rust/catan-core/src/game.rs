//! Turn management, action validation and execution.
//!
//! Every action is fully validated against the current phase, acting player,
//! and game rules: `execute_action` returns false and leaves the game
//! untouched for anything illegal. The set produced by `fill_valid_actions`
//! is exactly the set of actions that would execute successfully.

use crate::board::topology;
use crate::building::{
    build_city, build_road, build_settlement, city_placement_ok, for_each_valid_city_placement,
    for_each_valid_road_placement, for_each_valid_settlement_placement, has_free_road_placement,
    road_placement_ok, settlement_placement_ok,
};
use crate::dev_cards::{
    buy_dev_card, can_buy_dev_card, can_play_dev_card, play_knight, play_monopoly,
    play_road_building, play_year_of_plenty,
};
use crate::resources::{distribute_resources, roll_dice};
use crate::robber::{move_robber, steal_random_resource, steal_specific_resource, stealable_flags};
use crate::state::{
    GameState, DEV_DECK_SIZE, DEV_KNIGHT, DEV_MONOPOLY, DEV_ROAD_BUILDING, DEV_YEAR_OF_PLENTY,
};
use crate::trading::{for_each_bank_trade, trade_with_bank};

/// Offers a player may make per turn (keeps negotiation bounded).
pub const MAX_TRADES_PER_TURN: u8 = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum GamePhase {
    SetupForward = 0,
    SetupBackward = 1,
    Playing = 2,
    Finished = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum TurnPhase {
    PreRoll = 0,
    MustRoll = 1,
    RobberDiscard = 2,
    RobberMove = 3,
    RobberSteal = 4,
    Main = 5,
    RoadBuilding = 6,
    TradeResponse = 7,
    TradeChoose = 8,
}

/// An open player-trade offer: proposer gives `give_amount` of `give` and
/// receives 1 of `recv`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TradeOffer {
    pub proposer: u8,
    pub give: u8,
    pub give_amount: u8,
    pub recv: u8,
}

/// A game action. `forced` fields inject recorded randomness during replays;
/// live play passes `None` and the engine draws from its RNG.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Action {
    PlaceInitialSettlement {
        player: u8,
        vertex: u8,
    },
    PlaceInitialRoad {
        player: u8,
        edge: u8,
    },
    RollDice {
        player: u8,
        forced: Option<u8>,
    },
    BuildRoad {
        player: u8,
        edge: u8,
    },
    BuildSettlement {
        player: u8,
        vertex: u8,
    },
    BuildCity {
        player: u8,
        vertex: u8,
    },
    BuyDevCard {
        player: u8,
    },
    PlayKnight {
        player: u8,
    },
    PlayRoadBuilding {
        player: u8,
    },
    PlayYearOfPlenty {
        player: u8,
        r1: u8,
        r2: u8,
    },
    PlayMonopoly {
        player: u8,
        resource: u8,
    },
    MoveRobber {
        player: u8,
        tile: u8,
    },
    StealResource {
        player: u8,
        victim: i8,
        forced: Option<i8>,
    },
    /// Discard a single card; repeated until the 7-roll quota is met.
    DiscardResource {
        player: u8,
        resource: u8,
    },
    TradeWithBank {
        player: u8,
        give: u8,
        recv: u8,
    },
    ProposeTrade {
        player: u8,
        give: u8,
        give_amount: u8,
        recv: u8,
    },
    RespondTrade {
        player: u8,
        accept: bool,
    },
    /// Proposer picks an accepting partner, or -1 to cancel the trade.
    ConfirmTrade {
        player: u8,
        partner: i8,
    },
    EndTurn {
        player: u8,
    },
}

#[derive(Clone)]
pub struct CatanGame {
    pub state: GameState,
    pub game_phase: GamePhase,
    pub turn_phase: TurnPhase,
    pub setup_player_idx: i32,
    pub roads_to_place: i32,
    /// (player, cards still to discard) in seat order, set on a 7 roll.
    pub pending_discards: Vec<(usize, u8)>,
    pub discard_idx: usize,
    pub post_robber_phase: TurnPhase,
    pub trade_offer: Option<TradeOffer>,
    /// Seat offset (from proposer) of the responder currently being asked.
    pub trade_offset: usize,
    pub trade_accepts: [bool; 4],
    pub trades_proposed_this_turn: u8,
    /// When true (the default), every executed action is appended to
    /// `action_history`. Bulk simulation turns this off.
    pub record_history: bool,
    pub action_history: Vec<Action>,
}

impl CatanGame {
    pub fn new(num_players: usize, seed: u64) -> CatanGame {
        Self::with_state(GameState::new(num_players, seed))
    }

    /// New game with a custom win threshold (curriculum training: shorter
    /// first-to-7 games give denser reward signal than first-to-10).
    pub fn new_with_target(num_players: usize, seed: u64, victory_target: i32) -> CatanGame {
        assert!(
            (3..=20).contains(&victory_target),
            "victory target {victory_target} outside sane range 3-20"
        );
        let mut game = Self::new(num_players, seed);
        game.state.victory_target = victory_target;
        game
    }

    /// Build a game over an explicit board + dev deck (replay / service use).
    pub fn from_replay(
        num_players: usize,
        tile_resources: [i8; 19],
        tile_numbers: [i8; 19],
        port_types: [i8; 9],
        dev_deck: [i8; DEV_DECK_SIZE],
    ) -> CatanGame {
        Self::with_state(GameState::from_board(
            num_players,
            tile_resources,
            tile_numbers,
            port_types,
            dev_deck,
        ))
    }

    fn with_state(state: GameState) -> CatanGame {
        CatanGame {
            state,
            game_phase: GamePhase::SetupForward,
            turn_phase: TurnPhase::MustRoll,
            setup_player_idx: 0,
            roads_to_place: 0,
            // Pre-sized for 4 players: the hot loop never grows it.
            pending_discards: Vec::with_capacity(4),
            discard_idx: 0,
            post_robber_phase: TurnPhase::Main,
            trade_offer: None,
            trade_offset: 0,
            trade_accepts: [false; 4],
            trades_proposed_this_turn: 0,
            record_history: true,
            action_history: Vec::new(),
        }
    }

    /// The player who must act next (not always the turn owner: discards and
    /// trade responses are decided by other seats).
    pub fn current_player(&self) -> usize {
        match self.game_phase {
            GamePhase::SetupForward | GamePhase::SetupBackward => self.setup_player_idx as usize,
            _ => match self.turn_phase {
                TurnPhase::RobberDiscard => self
                    .pending_discards
                    .get(self.discard_idx)
                    .map_or(0, |&(p, _)| p),
                TurnPhase::TradeResponse => self.trade_responder().unwrap_or(0),
                _ => self.state.current_player,
            },
        }
    }

    fn trade_responder(&self) -> Option<usize> {
        let offer = self.trade_offer?;
        let n = self.state.num_players;
        Some((offer.proposer as usize + self.trade_offset) % n)
    }

    pub fn is_game_over(&self) -> bool {
        self.game_phase == GamePhase::Finished
    }

    pub fn winner(&self) -> i8 {
        self.state.winner
    }

    pub fn valid_actions(&self) -> Vec<Action> {
        let mut out = Vec::new();
        self.fill_valid_actions(&mut out);
        out
    }

    /// Fill `out` with the current player's valid actions. Clears `out`
    /// first; callers can reuse the buffer across turns to avoid allocation.
    pub fn fill_valid_actions(&self, out: &mut Vec<Action>) {
        out.clear();
        let player = self.current_player();
        match self.game_phase {
            GamePhase::Finished => {}
            GamePhase::SetupForward | GamePhase::SetupBackward => self.setup_actions(player, out),
            GamePhase::Playing => match self.turn_phase {
                TurnPhase::RobberDiscard => self.discard_actions(out),
                TurnPhase::RobberMove => self.robber_move_actions(player, out),
                TurnPhase::RobberSteal => self.steal_actions(player, out),
                TurnPhase::RoadBuilding => self.road_building_actions(player, out),
                TurnPhase::PreRoll => self.pre_roll_actions(player, out),
                TurnPhase::MustRoll => out.push(Action::RollDice {
                    player: player as u8,
                    forced: None,
                }),
                TurnPhase::Main => self.main_phase_actions(player, out),
                TurnPhase::TradeResponse => {
                    out.push(Action::RespondTrade {
                        player: player as u8,
                        accept: true,
                    });
                    out.push(Action::RespondTrade {
                        player: player as u8,
                        accept: false,
                    });
                }
                TurnPhase::TradeChoose => {
                    for partner in 0..self.state.num_players {
                        if self.trade_accepts[partner] {
                            out.push(Action::ConfirmTrade {
                                player: player as u8,
                                partner: partner as i8,
                            });
                        }
                    }
                    out.push(Action::ConfirmTrade {
                        player: player as u8,
                        partner: -1,
                    });
                }
            },
        }
    }

    fn pre_roll_actions(&self, player: usize, out: &mut Vec<Action>) {
        out.push(Action::RollDice {
            player: player as u8,
            forced: None,
        });
        if can_play_dev_card(&self.state, player, DEV_KNIGHT) {
            out.push(Action::PlayKnight {
                player: player as u8,
            });
        }
    }

    fn setup_actions(&self, player: usize, out: &mut Vec<Action>) {
        if self.expects_setup_settlement(player) {
            for_each_valid_settlement_placement(&self.state, player, true, |v| {
                out.push(Action::PlaceInitialSettlement {
                    player: player as u8,
                    vertex: v as u8,
                });
            });
        } else if self.expects_setup_road(player) {
            self.for_each_initial_road_placement(player, |e| {
                out.push(Action::PlaceInitialRoad {
                    player: player as u8,
                    edge: e as u8,
                });
            });
        }
    }

    fn expects_setup_settlement(&self, player: usize) -> bool {
        let expected = if self.game_phase == GamePhase::SetupForward {
            1
        } else {
            2
        };
        self.state.settlements_built[player] < expected
    }

    fn expects_setup_road(&self, player: usize) -> bool {
        !self.expects_setup_settlement(player)
            && self.state.roads_built[player] < self.state.settlements_built[player]
    }

    /// Edges adjacent to the player's settlement that has no road yet.
    fn for_each_initial_road_placement(&self, player: usize, mut f: impl FnMut(usize)) {
        let topo = topology();
        let player_i8 = player as i8;
        for vertex in 0..54 {
            if self.state.vertices[vertex] == player_i8 {
                let has_road = topo.vertex_edges[vertex]
                    .iter()
                    .any(|&e| e >= 0 && self.state.edges[e as usize] == player_i8);
                if !has_road {
                    for &e in &topo.vertex_edges[vertex] {
                        if e >= 0 && self.state.edges[e as usize] < 0 {
                            f(e as usize);
                        }
                    }
                }
            }
        }
    }

    fn initial_road_ok(&self, player: usize, edge: usize) -> bool {
        if edge >= 72 || self.state.edges[edge] >= 0 {
            return false;
        }
        let mut ok = false;
        self.for_each_initial_road_placement(player, |e| ok |= e == edge);
        ok
    }

    fn discard_actions(&self, out: &mut Vec<Action>) {
        if let Some(&(player, _)) = self.pending_discards.get(self.discard_idx) {
            for r in 0..5u8 {
                if self.state.resources[player][r as usize] > 0 {
                    out.push(Action::DiscardResource {
                        player: player as u8,
                        resource: r,
                    });
                }
            }
        }
    }

    fn robber_move_actions(&self, player: usize, out: &mut Vec<Action>) {
        for tile in 0..19u8 {
            if tile != self.state.robber_tile {
                out.push(Action::MoveRobber {
                    player: player as u8,
                    tile,
                });
            }
        }
    }

    fn steal_actions(&self, player: usize, out: &mut Vec<Action>) {
        let victims = stealable_flags(&self.state, self.state.robber_tile as usize);
        if victims.iter().any(|&v| v) {
            for (v, &stealable) in victims.iter().enumerate().take(self.state.num_players) {
                if stealable {
                    out.push(Action::StealResource {
                        player: player as u8,
                        victim: v as i8,
                        forced: None,
                    });
                }
            }
        } else {
            out.push(Action::StealResource {
                player: player as u8,
                victim: -1,
                forced: None,
            });
        }
    }

    fn road_building_actions(&self, player: usize, out: &mut Vec<Action>) {
        for_each_valid_road_placement(&self.state, player, true, |e| {
            out.push(Action::BuildRoad {
                player: player as u8,
                edge: e as u8,
            });
        });
    }

    fn main_phase_actions(&self, player: usize, out: &mut Vec<Action>) {
        let state = &self.state;
        let p = player as u8;

        for_each_valid_road_placement(state, player, false, |e| {
            out.push(Action::BuildRoad {
                player: p,
                edge: e as u8,
            });
        });
        for_each_valid_settlement_placement(state, player, false, |v| {
            out.push(Action::BuildSettlement {
                player: p,
                vertex: v as u8,
            });
        });
        for_each_valid_city_placement(state, player, |v| {
            out.push(Action::BuildCity {
                player: p,
                vertex: v as u8,
            });
        });
        if can_buy_dev_card(state, player) {
            out.push(Action::BuyDevCard { player: p });
        }
        if can_play_dev_card(state, player, DEV_KNIGHT) {
            out.push(Action::PlayKnight { player: p });
        }
        if self.can_play_road_building(player) {
            out.push(Action::PlayRoadBuilding { player: p });
        }
        if can_play_dev_card(state, player, DEV_YEAR_OF_PLENTY) {
            for r1 in 0..5u8 {
                for r2 in r1..5u8 {
                    let ok = if r1 == r2 {
                        state.bank[r1 as usize] >= 2
                    } else {
                        state.bank[r1 as usize] > 0 && state.bank[r2 as usize] > 0
                    };
                    if ok {
                        out.push(Action::PlayYearOfPlenty { player: p, r1, r2 });
                    }
                }
            }
        }
        if can_play_dev_card(state, player, DEV_MONOPOLY) {
            for r in 0..5u8 {
                out.push(Action::PlayMonopoly {
                    player: p,
                    resource: r,
                });
            }
        }
        for_each_bank_trade(state, player, |give, _amount, recv| {
            out.push(Action::TradeWithBank {
                player: p,
                give: give as u8,
                recv: recv as u8,
            });
        });
        if self.trades_proposed_this_turn < MAX_TRADES_PER_TURN {
            for give in 0..5u8 {
                for give_amount in 1..=2u8 {
                    if state.resources[player][give as usize] >= give_amount as i16 {
                        for recv in 0..5u8 {
                            if recv != give {
                                out.push(Action::ProposeTrade {
                                    player: p,
                                    give,
                                    give_amount,
                                    recv,
                                });
                            }
                        }
                    }
                }
            }
        }
        out.push(Action::EndTurn { player: p });
    }

    /// Road Building is only playable when it can place at least one road.
    fn can_play_road_building(&self, player: usize) -> bool {
        can_play_dev_card(&self.state, player, DEV_ROAD_BUILDING)
            && has_free_road_placement(&self.state, player)
    }

    pub fn execute_action(&mut self, action: &Action) -> bool {
        if self.game_phase == GamePhase::Finished {
            return false;
        }
        let in_setup = matches!(
            self.game_phase,
            GamePhase::SetupForward | GamePhase::SetupBackward
        );

        let ok = match *action {
            Action::PlaceInitialSettlement { player, vertex } => {
                in_setup
                    && player as usize == self.current_player()
                    && self.expects_setup_settlement(player as usize)
                    && self.exec_initial_settlement(player as usize, vertex as usize)
            }
            Action::PlaceInitialRoad { player, edge } => {
                in_setup
                    && player as usize == self.current_player()
                    && self.expects_setup_road(player as usize)
                    && self.initial_road_ok(player as usize, edge as usize)
                    && self.exec_initial_road(player as usize, edge as usize)
            }
            Action::RollDice { player, forced } => {
                !in_setup
                    && matches!(self.turn_phase, TurnPhase::PreRoll | TurnPhase::MustRoll)
                    && player as usize == self.state.current_player
                    && self.exec_roll_dice(forced)
            }
            Action::BuildRoad { player, edge } => {
                let p = player as usize;
                let free = self.turn_phase == TurnPhase::RoadBuilding;
                !in_setup
                    && matches!(self.turn_phase, TurnPhase::Main | TurnPhase::RoadBuilding)
                    && p == self.state.current_player
                    && road_placement_ok(&self.state, p, edge as usize, free)
                    && self.exec_build_road(p, edge as usize)
            }
            Action::BuildSettlement { player, vertex } => {
                let p = player as usize;
                !in_setup
                    && self.turn_phase == TurnPhase::Main
                    && p == self.state.current_player
                    && settlement_placement_ok(&self.state, p, vertex as usize, false)
                    && build_settlement(&mut self.state, p, vertex as usize, false)
            }
            Action::BuildCity { player, vertex } => {
                let p = player as usize;
                !in_setup
                    && self.turn_phase == TurnPhase::Main
                    && p == self.state.current_player
                    && city_placement_ok(&self.state, p, vertex as usize)
                    && build_city(&mut self.state, p, vertex as usize)
            }
            Action::BuyDevCard { player } => {
                !in_setup
                    && self.turn_phase == TurnPhase::Main
                    && player as usize == self.state.current_player
                    && buy_dev_card(&mut self.state, player as usize) >= 0
            }
            Action::PlayKnight { player } => {
                !in_setup
                    && matches!(self.turn_phase, TurnPhase::PreRoll | TurnPhase::Main)
                    && player as usize == self.state.current_player
                    && self.exec_play_knight(player as usize)
            }
            Action::PlayRoadBuilding { player } => {
                let p = player as usize;
                !in_setup
                    && self.turn_phase == TurnPhase::Main
                    && p == self.state.current_player
                    && self.can_play_road_building(p)
                    && {
                        let played = play_road_building(&mut self.state, p);
                        if played {
                            self.roads_to_place = 2;
                            self.turn_phase = TurnPhase::RoadBuilding;
                        }
                        played
                    }
            }
            Action::PlayYearOfPlenty { player, r1, r2 } => {
                !in_setup
                    && self.turn_phase == TurnPhase::Main
                    && player as usize == self.state.current_player
                    && r1 < 5
                    && r2 < 5
                    && play_year_of_plenty(
                        &mut self.state,
                        player as usize,
                        r1 as usize,
                        r2 as usize,
                    )
            }
            Action::PlayMonopoly { player, resource } => {
                !in_setup
                    && self.turn_phase == TurnPhase::Main
                    && player as usize == self.state.current_player
                    && play_monopoly(&mut self.state, player as usize, resource as usize) >= 0
            }
            Action::MoveRobber { player, tile } => {
                !in_setup
                    && self.turn_phase == TurnPhase::RobberMove
                    && player as usize == self.state.current_player
                    && self.exec_move_robber(tile as usize)
            }
            Action::StealResource {
                player,
                victim,
                forced,
            } => {
                !in_setup
                    && self.turn_phase == TurnPhase::RobberSteal
                    && player as usize == self.state.current_player
                    && self.steal_target_ok(victim)
                    && self.exec_steal(victim, forced)
            }
            Action::DiscardResource { player, resource } => {
                !in_setup
                    && self.turn_phase == TurnPhase::RobberDiscard
                    && resource < 5
                    && self.exec_discard(player as usize, resource as usize)
            }
            Action::TradeWithBank { player, give, recv } => {
                !in_setup
                    && self.turn_phase == TurnPhase::Main
                    && player as usize == self.state.current_player
                    && give < 5
                    && recv < 5
                    && trade_with_bank(
                        &mut self.state,
                        player as usize,
                        give as usize,
                        recv as usize,
                    )
            }
            Action::ProposeTrade {
                player,
                give,
                give_amount,
                recv,
            } => {
                !in_setup
                    && self.turn_phase == TurnPhase::Main
                    && player as usize == self.state.current_player
                    && self.exec_propose(player, give, give_amount, recv)
            }
            Action::RespondTrade { player, accept } => {
                !in_setup
                    && self.turn_phase == TurnPhase::TradeResponse
                    && Some(player as usize) == self.trade_responder()
                    && self.exec_respond(player as usize, accept)
            }
            Action::ConfirmTrade { player, partner } => {
                !in_setup
                    && self.turn_phase == TurnPhase::TradeChoose
                    && player as usize == self.state.current_player
                    && self.exec_confirm(partner)
            }
            Action::EndTurn { player } => {
                !in_setup
                    && self.turn_phase == TurnPhase::Main
                    && player as usize == self.state.current_player
                    && self.exec_end_turn()
            }
        };

        if ok {
            if self.record_history {
                self.action_history.push(*action);
            }
            // Victory is immediate, whatever the VP source: builds (roads via
            // longest road), dev card VPs, largest army via knights.
            if self.game_phase == GamePhase::Playing
                && matches!(
                    action,
                    Action::BuildRoad { .. }
                        | Action::BuildSettlement { .. }
                        | Action::BuildCity { .. }
                        | Action::BuyDevCard { .. }
                        | Action::PlayKnight { .. }
                )
            {
                self.check_victory(self.state.current_player);
            }
        }
        ok
    }

    fn exec_initial_settlement(&mut self, player: usize, vertex: usize) -> bool {
        let success = build_settlement(&mut self.state, player, vertex, true);
        if success && self.game_phase == GamePhase::SetupBackward {
            // Second settlement grants one resource per adjacent tile.
            for &tile in &topology().vertex_tiles[vertex] {
                if tile >= 0 {
                    let res = self.state.tile_resources[tile as usize];
                    if res < 5 {
                        self.state.resources[player][res as usize] += 1;
                        self.state.bank[res as usize] -= 1;
                    }
                }
            }
        }
        success
    }

    fn exec_initial_road(&mut self, player: usize, edge: usize) -> bool {
        let success = build_road(&mut self.state, player, edge, true);
        if success {
            self.advance_setup();
        }
        success
    }

    fn advance_setup(&mut self) {
        match self.game_phase {
            GamePhase::SetupForward => {
                self.setup_player_idx += 1;
                if self.setup_player_idx >= self.state.num_players as i32 {
                    self.game_phase = GamePhase::SetupBackward;
                    self.setup_player_idx = self.state.num_players as i32 - 1;
                }
            }
            GamePhase::SetupBackward => {
                self.setup_player_idx -= 1;
                if self.setup_player_idx < 0 {
                    self.game_phase = GamePhase::Playing;
                    self.state.phase = 1;
                    self.turn_phase = TurnPhase::PreRoll;
                }
            }
            _ => {}
        }
    }

    fn exec_roll_dice(&mut self, forced: Option<u8>) -> bool {
        let total = match forced {
            Some(t) => t,
            None => roll_dice(&mut self.state).2,
        };
        if !(2..=12).contains(&total) {
            return false;
        }
        self.state.dice_roll = total;
        self.state.has_rolled = true;

        if total == 7 {
            // Refill the reused buffer in place: zero allocations in steady
            // state (the zero_alloc test enforces this).
            self.pending_discards.clear();
            for p in 0..self.state.num_players {
                let held = self.state.total_resources(p);
                if held > 7 {
                    self.pending_discards.push((p, (held / 2) as u8));
                }
            }
            if !self.pending_discards.is_empty() {
                self.turn_phase = TurnPhase::RobberDiscard;
                self.discard_idx = 0;
            } else {
                self.turn_phase = TurnPhase::RobberMove;
            }
        } else {
            distribute_resources(&mut self.state, total);
            self.turn_phase = TurnPhase::Main;
        }
        true
    }

    fn exec_build_road(&mut self, player: usize, edge: usize) -> bool {
        let free = self.turn_phase == TurnPhase::RoadBuilding;
        let success = build_road(&mut self.state, player, edge, free);
        if success && self.turn_phase == TurnPhase::RoadBuilding {
            self.roads_to_place -= 1;
            // Place as many as you can: stop early when no placement remains.
            if self.roads_to_place <= 0 || !has_free_road_placement(&self.state, player) {
                self.roads_to_place = 0;
                self.turn_phase = TurnPhase::Main;
            }
        }
        success
    }

    fn exec_play_knight(&mut self, player: usize) -> bool {
        let success = play_knight(&mut self.state, player);
        if success {
            self.post_robber_phase = if self.turn_phase == TurnPhase::PreRoll {
                TurnPhase::MustRoll
            } else {
                TurnPhase::Main
            };
            self.turn_phase = TurnPhase::RobberMove;
        }
        success
    }

    fn exec_move_robber(&mut self, tile: usize) -> bool {
        let success = move_robber(&mut self.state, tile);
        if success {
            let any_victim = stealable_flags(&self.state, tile).iter().any(|&v| v);
            self.turn_phase = if any_victim {
                TurnPhase::RobberSteal
            } else {
                self.post_robber_phase
            };
        }
        success
    }

    fn steal_target_ok(&self, victim: i8) -> bool {
        let victims = stealable_flags(&self.state, self.state.robber_tile as usize);
        if victims.iter().any(|&v| v) {
            victim >= 0 && (victim as usize) < self.state.num_players && victims[victim as usize]
        } else {
            victim == -1
        }
    }

    fn exec_steal(&mut self, victim: i8, forced: Option<i8>) -> bool {
        if victim >= 0 {
            match forced {
                Some(resource) => {
                    steal_specific_resource(&mut self.state, victim as usize, resource);
                }
                None => {
                    steal_random_resource(&mut self.state, victim as usize);
                }
            }
        }
        self.turn_phase = self.post_robber_phase;
        true
    }

    fn exec_discard(&mut self, player: usize, resource: usize) -> bool {
        let Some(&(pending_player, remaining)) = self.pending_discards.get(self.discard_idx) else {
            return false;
        };
        if player != pending_player || self.state.resources[player][resource] < 1 {
            return false;
        }
        self.state.resources[player][resource] -= 1;
        self.state.bank[resource] += 1;

        let remaining = remaining - 1;
        self.pending_discards[self.discard_idx].1 = remaining;
        if remaining == 0 {
            self.discard_idx += 1;
            if self.discard_idx >= self.pending_discards.len() {
                self.turn_phase = TurnPhase::RobberMove;
            }
        }
        true
    }

    fn exec_propose(&mut self, player: u8, give: u8, give_amount: u8, recv: u8) -> bool {
        if give >= 5 || recv >= 5 || give == recv || !(1..=2).contains(&give_amount) {
            return false;
        }
        if self.trades_proposed_this_turn >= MAX_TRADES_PER_TURN {
            return false;
        }
        if self.state.resources[player as usize][give as usize] < give_amount as i16 {
            return false;
        }
        self.trades_proposed_this_turn += 1;
        self.trade_offer = Some(TradeOffer {
            proposer: player,
            give,
            give_amount,
            recv,
        });
        self.trade_accepts = [false; 4];
        match self.first_eligible_offset(1) {
            Some(k) => {
                self.trade_offset = k;
                self.turn_phase = TurnPhase::TradeResponse;
            }
            None => {
                // Nobody can accept: the offer fizzles (still counts toward
                // the cap).
                self.trade_offer = None;
            }
        }
        true
    }

    fn first_eligible_offset(&self, from: usize) -> Option<usize> {
        let offer = self.trade_offer?;
        let n = self.state.num_players;
        (from..n).find(|&k| {
            let seat = (offer.proposer as usize + k) % n;
            seat != offer.proposer as usize && self.state.resources[seat][offer.recv as usize] >= 1
        })
    }

    fn exec_respond(&mut self, responder: usize, accept: bool) -> bool {
        if accept {
            self.trade_accepts[responder] = true;
        }
        match self.first_eligible_offset(self.trade_offset + 1) {
            Some(next) => {
                self.trade_offset = next;
            }
            None => {
                if self.trade_accepts.iter().any(|&a| a) {
                    self.turn_phase = TurnPhase::TradeChoose;
                } else {
                    self.trade_offer = None;
                    self.turn_phase = TurnPhase::Main;
                }
            }
        }
        true
    }

    fn exec_confirm(&mut self, partner: i8) -> bool {
        let Some(offer) = self.trade_offer else {
            return false;
        };
        if partner >= 0 {
            let partner = partner as usize;
            if partner >= self.state.num_players || !self.trade_accepts[partner] {
                return false;
            }
            let proposer = offer.proposer as usize;
            let give = offer.give as usize;
            let recv = offer.recv as usize;
            let amount = offer.give_amount as i16;
            self.state.resources[proposer][give] -= amount;
            self.state.resources[partner][give] += amount;
            self.state.resources[partner][recv] -= 1;
            self.state.resources[proposer][recv] += 1;
        }
        self.trade_offer = None;
        self.turn_phase = TurnPhase::Main;
        true
    }

    fn exec_end_turn(&mut self) -> bool {
        self.state.turn += 1;
        self.state.current_player = (self.state.current_player + 1) % self.state.num_players;
        self.turn_phase = TurnPhase::PreRoll;
        self.post_robber_phase = TurnPhase::Main;
        self.state.has_rolled = false;
        self.state.dev_cards_bought_this_turn = [0; 5];
        self.state.dev_card_played_this_turn = false;
        self.trades_proposed_this_turn = 0;
        true
    }

    fn check_victory(&mut self, player: usize) {
        if self.state.calculate_victory_points(player) >= self.state.victory_target {
            self.state.winner = player as i8;
            self.state.phase = 2;
            self.game_phase = GamePhase::Finished;
        }
    }
}
