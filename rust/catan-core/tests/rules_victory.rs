//! Victory: reaching 10 VP ends the game immediately, whatever the source.

use catan_core::board::topology;
use catan_core::building::build_settlement;
use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};
use catan_core::state::{DEV_KNIGHT, DEV_VICTORY_POINT};

/// Game in player 0's main phase with empty hands.
fn main_phase_game() -> CatanGame {
    let mut game = CatanGame::new(4, 23);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.turn_phase = TurnPhase::Main;
    game.state.current_player = 0;
    game.state.has_rolled = true;
    game
}

/// Place `count` extra settlements for player 0 wherever the distance rule
/// allows (lifting the build limit), to assemble a chosen VP total.
fn grant_settlements(game: &mut CatanGame, count: usize) {
    game.state.max_settlements = 10;
    let mut placed = 0;
    for v in 0..54 {
        if placed == count {
            break;
        }
        if build_settlement(&mut game.state, 0, v, true) {
            placed += 1;
        }
    }
    assert_eq!(
        placed, count,
        "board has room for {count} spaced settlements"
    );
}

fn edge_between(a: u8, b: u8) -> u8 {
    let key = [a.min(b), a.max(b)];
    (0..72u8)
        .find(|&e| topology().edge_vertices[e as usize] == key)
        .expect("edge exists")
}

#[test]
fn buying_the_winning_vp_card_ends_the_game_immediately() {
    let mut game = main_phase_game();
    grant_settlements(&mut game, 9);
    assert_eq!(game.state.calculate_victory_points(0), 9);

    // Stack the deck so the next card is a VP card.
    game.state.dev_deck[game.state.dev_deck_idx] = DEV_VICTORY_POINT as i8;
    game.state.resources[0] = [1, 1, 0, 0, 1];
    assert!(game.execute_action(&Action::BuyDevCard { player: 0 }));

    assert_eq!(game.state.calculate_victory_points(0), 10);
    assert!(
        game.is_game_over(),
        "10th VP from a dev card must end the game at once"
    );
    assert_eq!(game.winner(), 0);
}

#[test]
fn largest_army_completing_10_vp_ends_the_game_immediately() {
    let mut game = main_phase_game();
    grant_settlements(&mut game, 8);
    assert_eq!(game.state.calculate_victory_points(0), 8);

    // Two knights already played; the third grants largest army (+2 VP).
    game.state.knights_played[0] = 2;
    game.state.dev_cards[0][DEV_KNIGHT] = 1;
    assert!(game.execute_action(&Action::PlayKnight { player: 0 }));

    assert_eq!(game.state.calculate_victory_points(0), 10);
    assert!(
        game.is_game_over(),
        "largest army completing 10 VP must end the game"
    );
    assert_eq!(game.winner(), 0);
}

#[test]
fn longest_road_completing_10_vp_ends_on_the_road_build() {
    let mut game = main_phase_game();
    // Anchor settlement on tile 9's ring so the road chain is connected.
    let ring = topology().tile_vertices[9];
    assert!(build_settlement(&mut game.state, 0, ring[0] as usize, true));
    grant_settlements(&mut game, 7); // 8 VP total
    assert_eq!(game.state.calculate_victory_points(0), 8);

    game.state.resources[0] = [0, 0, 10, 10, 0];
    for i in 0..5 {
        let e = edge_between(ring[i], ring[(i + 1) % 6]);
        assert!(
            game.execute_action(&Action::BuildRoad { player: 0, edge: e }),
            "road {i} should build"
        );
        if game.is_game_over() {
            break;
        }
    }
    assert!(
        game.is_game_over(),
        "longest road pushing to 10 VP ends the game"
    );
    assert_eq!(game.winner(), 0);
    assert_eq!(game.state.longest_road_player, 0);
}

#[test]
fn game_over_blocks_all_further_actions() {
    let mut game = main_phase_game();
    game.state.winner = 0;
    game.state.phase = 2;
    game.game_phase = GamePhase::Finished;

    assert!(
        game.valid_actions().is_empty(),
        "finished game offers no actions"
    );
    assert!(!game.execute_action(&Action::EndTurn { player: 0 }));
    assert!(!game.execute_action(&Action::RollDice {
        player: 0,
        forced: None
    }));
}

#[test]
fn nine_points_does_not_win() {
    let mut game = main_phase_game();
    let ring = topology().tile_vertices[9];
    assert!(build_settlement(&mut game.state, 0, ring[0] as usize, true));
    grant_settlements(&mut game, 8); // 9 VP total
    assert_eq!(game.state.calculate_victory_points(0), 9);

    // Any non-scoring action leaves the game running.
    game.state.resources[0] = [0, 0, 1, 1, 0];
    let e = edge_between(ring[0], ring[1]);
    assert!(game.execute_action(&Action::BuildRoad { player: 0, edge: e }));
    assert!(!game.is_game_over(), "9 VP must not win");
}

#[test]
fn victory_target_is_configurable() {
    // Curriculum lever: a first-to-7 game must end at exactly 7 VP.
    let mut game = CatanGame::new_with_target(4, 23, 7);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.turn_phase = TurnPhase::Main;
    game.state.current_player = 0;
    game.state.has_rolled = true;

    grant_settlements(&mut game, 6);
    assert_eq!(game.state.calculate_victory_points(0), 6);
    assert!(!game.is_game_over(), "6 VP must not win a first-to-7 game");

    // The 7th point (a VP dev card) ends the game immediately.
    game.state.dev_deck[game.state.dev_deck_idx] = DEV_VICTORY_POINT as i8;
    game.state.resources[0] = [1, 1, 0, 0, 1];
    assert!(game.execute_action(&Action::BuyDevCard { player: 0 }));
    assert!(game.is_game_over(), "7 VP wins a first-to-7 game");
    assert_eq!(game.winner(), 0);
}

#[test]
fn lower_victory_target_shortens_games() {
    use catan_core::players::{play_game, Player, RandomPlayer};
    let avg_turns = |target: i32| -> f64 {
        let mut total = 0u32;
        for seed in 0..8u64 {
            let mut game = CatanGame::new_with_target(4, seed, target);
            let mut players: Vec<Box<dyn Player>> = (0..4)
                .map(|i| Box::new(RandomPlayer::new(seed * 5 + i)) as Box<dyn Player>)
                .collect();
            let winner = play_game(&mut game, &mut players);
            if winner >= 0 {
                assert!(
                    game.state.calculate_victory_points(winner as usize) >= target,
                    "winner below target {target}"
                );
            }
            total += game.state.turn;
        }
        total as f64 / 8.0
    };
    let t7 = avg_turns(7);
    let t10 = avg_turns(10);
    assert!(
        t7 < t10,
        "first-to-7 should be shorter than first-to-10 ({t7:.0} vs {t10:.0} turns)"
    );
}
