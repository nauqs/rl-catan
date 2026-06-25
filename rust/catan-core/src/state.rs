//! Game state: fixed-size arrays, ~500 bytes per game.

use rand::rngs::SmallRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::board::{topology, PORT_TYPES, RESOURCE_DESERT, TILE_NUMBERS, TILE_RESOURCES};

pub const NUM_RESOURCES: usize = 5;
pub const MAX_PLAYERS: usize = 4;
pub const INITIAL_RESOURCES_PER_TYPE: i16 = 19;
/// Default win threshold; per-game value lives on `GameState::victory_target`
/// (configurable for curriculum training: first-to-7 before first-to-10).
pub const VICTORY_POINTS_TO_WIN: i32 = 10;

pub const ROAD_COST: [i16; 5] = [0, 0, 1, 1, 0];
pub const SETTLEMENT_COST: [i16; 5] = [1, 1, 1, 1, 0];
pub const CITY_COST: [i16; 5] = [2, 0, 0, 0, 3];
pub const DEV_CARD_COST: [i16; 5] = [1, 1, 0, 0, 1];

pub const DEV_KNIGHT: usize = 0;
pub const DEV_VICTORY_POINT: usize = 1;
pub const DEV_ROAD_BUILDING: usize = 2;
pub const DEV_YEAR_OF_PLENTY: usize = 3;
pub const DEV_MONOPOLY: usize = 4;
pub const NUM_DEV_CARD_TYPES: usize = 5;
pub const DEV_CARD_COUNTS: [u8; 5] = [14, 5, 2, 2, 2];
pub const DEV_DECK_SIZE: usize = 25;

#[derive(Clone, Debug)]
pub struct GameState {
    pub num_players: usize,
    pub seed: u64,
    pub rng: SmallRng,

    /// 0 = setup, 1 = playing, 2 = finished (mirrors Python's state.phase).
    pub phase: u8,
    pub turn: u32,
    pub current_player: usize,
    pub winner: i8,

    pub tile_resources: [i8; 19],
    pub tile_numbers: [i8; 19],
    pub port_types: [i8; 9],

    /// -1 empty, 0-3 settlement by player, 4-7 city by player.
    pub vertices: [i8; 54],
    /// -1 empty, 0-3 road by player.
    pub edges: [i8; 72],

    pub settlements_built: [u8; MAX_PLAYERS],
    pub cities_built: [u8; MAX_PLAYERS],
    pub roads_built: [u8; MAX_PLAYERS],
    pub max_settlements: u8,
    pub max_cities: u8,
    pub max_roads: u8,

    /// Per player: wheat, sheep, wood, brick, stone.
    pub resources: [[i16; NUM_RESOURCES]; MAX_PLAYERS],
    pub bank: [i16; NUM_RESOURCES],

    pub dev_deck: [i8; DEV_DECK_SIZE],
    pub dev_deck_idx: usize,
    pub dev_cards: [[i8; NUM_DEV_CARD_TYPES]; MAX_PLAYERS],
    pub knights_played: [i8; MAX_PLAYERS],
    pub dev_cards_bought_this_turn: [i8; NUM_DEV_CARD_TYPES],
    pub dev_card_played_this_turn: bool,

    pub longest_road_player: i8,
    pub longest_road_length: u8,
    pub largest_army_player: i8,
    pub largest_army_size: i8,

    pub robber_tile: u8,
    pub dice_roll: u8,
    pub has_rolled: bool,

    /// Victory points needed to win this game (default 10; lowered for
    /// curriculum training).
    pub victory_target: i32,

    /// Port access cache, updated by `build_settlement`. Ports are never
    /// lost once gained (cities keep them), so these only ever flip to true.
    /// Bypassed if `vertices` is mutated directly without `build_settlement`.
    pub port_any: [bool; MAX_PLAYERS],
    pub port_resource: [[bool; NUM_RESOURCES]; MAX_PLAYERS],

    /// Tiles producing on each dice total (max 2 on a standard board; 3
    /// slots for safety). -1 = empty slot. Derived from `tile_numbers`.
    pub tiles_by_number: [[i8; 3]; 13],

    /// Bit v set = vertex v holds any building. Maintained by
    /// `build_settlement` (cities keep the bit). Bypassed by direct
    /// `vertices` mutation.
    pub occupied_mask: u64,
    /// Bit v set = the player owns a road on an edge touching vertex v.
    /// Maintained by `build_road`.
    pub vertex_road_mask: [u64; MAX_PLAYERS],
    /// Cached longest-road length per player, kept current by the build
    /// functions; the award logic reads this instead of recomputing everyone.
    pub road_lengths: [u8; MAX_PLAYERS],
}

impl GameState {
    pub fn new(num_players: usize, seed: u64) -> GameState {
        assert!((2..=4).contains(&num_players), "Catan supports 2-4 players");
        let mut rng = SmallRng::seed_from_u64(seed);

        let mut tile_resources = TILE_RESOURCES;
        tile_resources.shuffle(&mut rng);

        let mut numbers: Vec<i8> = TILE_NUMBERS.iter().copied().filter(|&n| n > 0).collect();
        numbers.shuffle(&mut rng);
        let mut tile_numbers = [0i8; 19];
        let mut number_idx = 0;
        for i in 0..19 {
            if tile_resources[i] != RESOURCE_DESERT {
                tile_numbers[i] = numbers[number_idx];
                number_idx += 1;
            }
        }

        let mut port_types = PORT_TYPES;
        port_types.shuffle(&mut rng);

        let mut deck = [0i8; DEV_DECK_SIZE];
        let mut i = 0;
        for (card_type, &count) in DEV_CARD_COUNTS.iter().enumerate() {
            for _ in 0..count {
                deck[i] = card_type as i8;
                i += 1;
            }
        }
        deck.shuffle(&mut rng);

        Self::from_parts(
            num_players,
            seed,
            rng,
            tile_resources,
            tile_numbers,
            port_types,
            deck,
        )
    }

    /// Build a state with explicit board + deck (replay / service use).
    pub fn from_board(
        num_players: usize,
        tile_resources: [i8; 19],
        tile_numbers: [i8; 19],
        port_types: [i8; 9],
        dev_deck: [i8; DEV_DECK_SIZE],
    ) -> GameState {
        let rng = SmallRng::seed_from_u64(0);
        Self::from_parts(
            num_players,
            0,
            rng,
            tile_resources,
            tile_numbers,
            port_types,
            dev_deck,
        )
    }

    fn from_parts(
        num_players: usize,
        seed: u64,
        rng: SmallRng,
        tile_resources: [i8; 19],
        tile_numbers: [i8; 19],
        port_types: [i8; 9],
        dev_deck: [i8; DEV_DECK_SIZE],
    ) -> GameState {
        let robber_tile = tile_resources
            .iter()
            .position(|&r| r == RESOURCE_DESERT)
            .expect("board has a desert") as u8;

        let mut tiles_by_number = [[-1i8; 3]; 13];
        for (tile, &number) in tile_numbers.iter().enumerate() {
            if number > 0 {
                let slots = &mut tiles_by_number[number as usize];
                let slot = slots
                    .iter_mut()
                    .find(|s| **s == -1)
                    .expect("more than 3 tiles share a number");
                *slot = tile as i8;
            }
        }

        GameState {
            num_players,
            seed,
            rng,
            phase: 0,
            turn: 0,
            current_player: 0,
            winner: -1,
            tile_resources,
            tile_numbers,
            port_types,
            vertices: [-1; 54],
            edges: [-1; 72],
            settlements_built: [0; MAX_PLAYERS],
            cities_built: [0; MAX_PLAYERS],
            roads_built: [0; MAX_PLAYERS],
            max_settlements: 5,
            max_cities: 4,
            max_roads: 15,
            resources: [[0; NUM_RESOURCES]; MAX_PLAYERS],
            bank: [INITIAL_RESOURCES_PER_TYPE; NUM_RESOURCES],
            dev_deck,
            dev_deck_idx: 0,
            dev_cards: [[0; NUM_DEV_CARD_TYPES]; MAX_PLAYERS],
            knights_played: [0; MAX_PLAYERS],
            dev_cards_bought_this_turn: [0; NUM_DEV_CARD_TYPES],
            dev_card_played_this_turn: false,
            longest_road_player: -1,
            longest_road_length: 0,
            largest_army_player: -1,
            largest_army_size: 0,
            robber_tile,
            dice_roll: 0,
            has_rolled: false,
            victory_target: VICTORY_POINTS_TO_WIN,
            port_any: [false; MAX_PLAYERS],
            port_resource: [[false; NUM_RESOURCES]; MAX_PLAYERS],
            tiles_by_number,
            occupied_mask: 0,
            vertex_road_mask: [0; MAX_PLAYERS],
            road_lengths: [0; MAX_PLAYERS],
        }
    }

    pub fn total_resources(&self, player: usize) -> i16 {
        self.resources[player].iter().sum()
    }

    /// Owner of the settlement/city at a vertex, or -1.
    pub fn settlement_owner(&self, vertex: usize) -> i8 {
        let val = self.vertices[vertex];
        if val < 0 {
            -1
        } else {
            val % 4
        }
    }

    pub fn is_city(&self, vertex: usize) -> bool {
        self.vertices[vertex] >= 4
    }

    pub fn calculate_victory_points(&self, player: usize) -> i32 {
        let mut vp = self.settlements_built[player] as i32;
        vp += 2 * self.cities_built[player] as i32;
        vp += self.dev_cards[player][DEV_VICTORY_POINT] as i32;
        if self.longest_road_player == player as i8 {
            vp += 2;
        }
        if self.largest_army_player == player as i8 {
            vp += 2;
        }
        vp
    }

    /// Port type at a vertex for THIS game (-1 none, 0 any/3:1, 1-5 = 2:1
    /// for resource type-1), resolved through the shuffled `port_types`.
    pub fn vertex_port_type(&self, vertex: usize) -> i8 {
        let idx = topology().vertex_port_index[vertex];
        if idx < 0 {
            -1
        } else {
            self.port_types[idx as usize]
        }
    }

    /// Total resource-production probability of a vertex (heuristic AI).
    pub fn vertex_probability(&self, vertex: usize) -> f32 {
        let topo = topology();
        let mut total = 0.0;
        for &tile in &topo.vertex_tiles[vertex] {
            if tile >= 0 {
                total += topo.number_probabilities[self.tile_numbers[tile as usize] as usize];
            }
        }
        total
    }
}
