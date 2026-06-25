//! Road Building card: place up to 2 free roads — as many as you can.
//! The card is not offered when it would place zero roads, and the game
//! never stalls.

use catan_core::board::topology;
use catan_core::building::build_settlement;
use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};
use catan_core::state::DEV_ROAD_BUILDING;

fn main_phase_game() -> CatanGame {
    let mut game = CatanGame::new(4, 53);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.turn_phase = TurnPhase::Main;
    game.state.current_player = 0;
    game.state.has_rolled = true;
    game.state.dev_cards[0][DEV_ROAD_BUILDING] = 1;
    game
}

#[test]
fn places_two_free_roads_normally() {
    let mut game = main_phase_game();
    assert!(build_settlement(&mut game.state, 0, 0, true));

    assert!(game.execute_action(&Action::PlayRoadBuilding { player: 0 }));
    assert_eq!(game.turn_phase, TurnPhase::RoadBuilding);

    for i in 0..2 {
        let valid = game.valid_actions();
        assert!(!valid.is_empty(), "placement {i} available");
        assert!(valid.iter().all(|a| matches!(a, Action::BuildRoad { .. })));
        assert!(game.execute_action(&valid[0]));
    }
    assert_eq!(game.turn_phase, TurnPhase::Main);
    assert_eq!(game.state.roads_built[0], 2);
    assert_eq!(game.state.total_resources(0), 0, "roads were free");
}

#[test]
fn not_offered_at_road_limit() {
    let mut game = main_phase_game();
    assert!(build_settlement(&mut game.state, 0, 0, true));
    game.state.roads_built[0] = game.state.max_roads;

    let valid = game.valid_actions();
    assert!(
        !valid
            .iter()
            .any(|a| matches!(a, Action::PlayRoadBuilding { .. })),
        "card with zero placeable roads must not be offered"
    );
    assert!(!game.execute_action(&Action::PlayRoadBuilding { player: 0 }));
    assert_eq!(
        game.state.dev_cards[0][DEV_ROAD_BUILDING], 1,
        "card not consumed"
    );
}

#[test]
fn places_only_one_road_when_only_one_fits() {
    let mut game = main_phase_game();
    // One settlement whose every adjacent region is walled off except a
    // single edge: occupy all but one edge around vertex 0's neighborhood
    // with opponent roads connected nowhere (placed directly).
    assert!(build_settlement(&mut game.state, 0, 0, true));
    let topo = topology();
    // Fill every empty edge except exactly one reachable edge with opponent
    // roads so player 0 has a single legal placement.
    let keep: usize = topo.vertex_edges[0]
        .iter()
        .find(|&&e| e >= 0)
        .map(|&e| e as usize)
        .unwrap();
    for e in 0..72 {
        if e != keep && game.state.edges[e] < 0 {
            game.state.edges[e] = 1;
        }
    }

    assert!(game.execute_action(&Action::PlayRoadBuilding { player: 0 }));
    assert_eq!(game.turn_phase, TurnPhase::RoadBuilding);
    let valid = game.valid_actions();
    assert_eq!(valid.len(), 1);
    assert!(game.execute_action(&valid[0]));
    // Second placement impossible: phase returns to Main instead of stalling.
    assert_eq!(game.turn_phase, TurnPhase::Main);
    assert_eq!(game.state.roads_built[0], 1);
}

#[test]
fn games_with_road_building_never_stall() {
    use catan_core::players::{play_game, Player, RandomPlayer};
    for seed in 0..40u64 {
        let mut game = CatanGame::new(4, seed);
        let mut players: Vec<Box<dyn Player>> = (0..4)
            .map(|i| Box::new(RandomPlayer::new(seed * 7 + i)) as Box<dyn Player>)
            .collect();
        play_game(&mut game, &mut players);
        assert!(
            game.is_game_over() || game.state.turn >= 1000,
            "seed {seed}: game stalled (phase {:?})",
            game.turn_phase
        );
    }
}
