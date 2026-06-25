//! End-to-end games with AI players: termination, win conditions, and
//! conservation invariants.

use catan_core::game::{CatanGame, GamePhase};
use catan_core::players::{play_game, HeuristicPlayer, Player, RandomPlayer};

fn check_invariants(game: &CatanGame) {
    let state = &game.state;

    // Resource conservation: every resource type sums to 19 across bank + hands.
    for r in 0..5 {
        let held: i16 = (0..state.num_players).map(|p| state.resources[p][r]).sum();
        assert_eq!(state.bank[r] + held, 19, "resource {r} not conserved");
        for p in 0..state.num_players {
            assert!(state.resources[p][r] >= 0, "negative resources");
        }
    }

    // Building counters match the board arrays.
    for p in 0..state.num_players {
        let s = state.vertices.iter().filter(|&&v| v == p as i8).count();
        let c = state.vertices.iter().filter(|&&v| v == p as i8 + 4).count();
        let r = state.edges.iter().filter(|&&e| e == p as i8).count();
        assert_eq!(state.settlements_built[p] as usize, s);
        assert_eq!(state.cities_built[p] as usize, c);
        assert_eq!(state.roads_built[p] as usize, r);
        assert!(s <= 5 && c <= 4 && r <= 15, "building limits exceeded");
    }

    // Dev cards: drawn cards equal cards in hands plus knights played.
    let drawn = state.dev_deck_idx as i32;
    let held: i32 = (0..state.num_players)
        .map(|p| state.dev_cards[p].iter().map(|&c| c as i32).sum::<i32>())
        .sum();
    let played: i32 = (0..state.num_players)
        .map(|p| state.knights_played[p] as i32)
        .sum();
    let road_building_and_others = drawn - held - played;
    assert!(
        road_building_and_others >= 0,
        "more dev cards in play than drawn"
    );

    // Robber is on the board.
    assert!((state.robber_tile as usize) < 19);
}

#[test]
fn random_games_terminate_with_valid_winners() {
    for seed in 0..30u64 {
        let mut game = CatanGame::new(4, seed);
        let mut players: Vec<Box<dyn Player>> = (0..4)
            .map(|i| Box::new(RandomPlayer::new(seed * 100 + i)) as Box<dyn Player>)
            .collect();
        let winner = play_game(&mut game, &mut players);

        assert!(
            game.game_phase == GamePhase::Finished || game.state.turn >= 1000,
            "seed {seed}: game neither finished nor hit the turn cap"
        );
        if winner >= 0 {
            let vp = game.state.calculate_victory_points(winner as usize);
            assert!(vp >= 10, "seed {seed}: winner has only {vp} VP");
        }
        check_invariants(&game);
    }
}

#[test]
fn heuristic_games_terminate_with_valid_winners() {
    for seed in 0..10u64 {
        let mut game = CatanGame::new(4, seed);
        let mut players: Vec<Box<dyn Player>> = vec![
            Box::new(HeuristicPlayer::new(seed)),
            Box::new(HeuristicPlayer::new(seed + 1)),
            Box::new(RandomPlayer::new(seed + 2)),
            Box::new(RandomPlayer::new(seed + 3)),
        ];
        let winner = play_game(&mut game, &mut players);
        if winner >= 0 {
            assert!(game.state.calculate_victory_points(winner as usize) >= 10);
        }
        check_invariants(&game);
    }
}

#[test]
fn three_player_games_work() {
    for seed in 0..10u64 {
        let mut game = CatanGame::new(3, seed);
        let mut players: Vec<Box<dyn Player>> = (0..3)
            .map(|i| Box::new(RandomPlayer::new(seed + i)) as Box<dyn Player>)
            .collect();
        play_game(&mut game, &mut players);
        check_invariants(&game);
    }
}

#[test]
fn same_seed_same_outcome() {
    let run = |seed: u64| {
        let mut game = CatanGame::new(4, seed);
        let mut players: Vec<Box<dyn Player>> = (0..4)
            .map(|i| Box::new(RandomPlayer::new(seed + i)) as Box<dyn Player>)
            .collect();
        let winner = play_game(&mut game, &mut players);
        (
            winner,
            game.state.turn,
            game.state.vertices,
            game.state.edges,
        )
    };
    assert_eq!(
        run(123),
        run(123),
        "identical seeds must reproduce identical games"
    );
}

#[test]
fn setup_phase_grants_resources_for_second_settlement() {
    // After full setup, each player has the resources from tiles adjacent to
    // their second settlement (0-3 cards each, desert gives nothing).
    let mut game = CatanGame::new(4, 77);
    let mut players: Vec<Box<dyn Player>> = (0..4)
        .map(|i| Box::new(RandomPlayer::new(77 + i)) as Box<dyn Player>)
        .collect();
    for p in 0..4usize {
        players[p].on_game_start(&game.state, p);
    }
    while game.game_phase != GamePhase::Playing {
        let valid = game.valid_actions();
        assert!(!valid.is_empty(), "setup must always offer actions");
        let player = game.current_player();
        let action = players[player].choose_action(&game, &valid);
        assert!(game.execute_action(&action), "setup action must succeed");
    }
    for p in 0..4 {
        let total: i16 = game.state.resources[p].iter().sum();
        assert!(total <= 3, "player {p} has more than 3 starting cards");
        assert_eq!(game.state.settlements_built[p], 2);
        assert_eq!(game.state.roads_built[p], 2);
    }
}
