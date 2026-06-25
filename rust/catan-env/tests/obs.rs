//! The observation encoder is the policy network's only window into the
//! game. These tests pin the layout (frozen once training starts), prove
//! seat-relativity, prove the visibility modes differ in exactly the
//! opponent-private block, and bound every value on live games.

use catan_core::building::build_settlement;
use catan_core::game::{Action, CatanGame, GamePhase, TurnPhase};
use catan_core::players::{Player, RandomPlayer};
use catan_env::obs::{self, encode_obs, Visibility, OBS_DIM};

fn buf() -> Vec<f32> {
    vec![0.0; OBS_DIM]
}

#[test]
fn layout_spot_checks() {
    let mut game = CatanGame::new(4, 7);
    game.state.resources[0] = [5, 0, 2, 0, 0];
    game.state.resources[1] = [0, 3, 0, 0, 0];

    let mut o = buf();
    encode_obs(&game, 0, Visibility::Perfect, &mut o);

    // Robber flag on its tile, and nowhere else.
    let rt = game.state.robber_tile as usize;
    assert_eq!(o[obs::TILES + rt * obs::TILE_STRIDE + 7], 1.0);
    let robber_flags: f32 = (0..19)
        .map(|t| o[obs::TILES + t * obs::TILE_STRIDE + 7])
        .sum();
    assert_eq!(robber_flags, 1.0);

    // Tile resource one-hots: exactly one per tile, matching the board.
    for t in 0..19 {
        let base = obs::TILES + t * obs::TILE_STRIDE;
        let hot: Vec<usize> = (0..6).filter(|&r| o[base + r] == 1.0).collect();
        assert_eq!(hot, vec![game.state.tile_resources[t] as usize], "tile {t}");
    }

    // My hand in the self-private block, normalized /19.
    assert_eq!(o[obs::SELF_PRIVATE], 5.0 / 19.0);
    assert_eq!(o[obs::SELF_PRIVATE + 2], 2.0 / 19.0);

    // Seat 1's hand (3 sheep) in opp-private rel slot 0, sheep position.
    assert_eq!(o[obs::OPP_PRIVATE + 1], 3.0 / 19.0);

    // Public card counts in the player blocks: me 7, left-neighbor 3.
    assert_eq!(o[obs::PLAYERS + obs::PLAYER_CARDS], 7.0 / 19.0);
    assert_eq!(
        o[obs::PLAYERS + obs::PLAYER_STRIDE + obs::PLAYER_CARDS],
        3.0 / 19.0
    );

    // Bank starts full.
    for r in 0..5 {
        assert!((o[obs::BANK + r] - (game.state.bank[r] as f32 / 19.0)).abs() < 1e-6);
    }

    // Context: setup phase one-hot, victory target 10.
    assert_eq!(o[obs::CONTEXT + GamePhase::SetupForward as usize], 1.0);
    assert_eq!(o[obs::CONTEXT + 16], 1.0);
}

#[test]
fn buildings_are_seat_relative() {
    let mut game = CatanGame::new(4, 9);
    // Player 2 settles vertex 0 (free, direct).
    assert!(build_settlement(&mut game.state, 2, 0, true));

    // From seat 2's view that building is "mine" (rel 0, settlement slot).
    let mut o = buf();
    encode_obs(&game, 2, Visibility::Perfect, &mut o);
    assert_eq!(o[obs::VERTICES], 1.0, "rel-0 settlement slot at vertex 0");

    // From seat 1's view, player 2 is the seat to my left (rel 1).
    encode_obs(&game, 1, Visibility::Perfect, &mut o);
    assert_eq!(o[obs::VERTICES + 2], 1.0, "rel-1 settlement slot");
    assert_eq!(o[obs::VERTICES], 0.0);

    // City upgrade flips the slot within the same rel pair.
    game.state.resources[2] = [2, 0, 0, 0, 3];
    assert!(catan_core::building::build_city(&mut game.state, 2, 0));
    encode_obs(&game, 2, Visibility::Perfect, &mut o);
    assert_eq!(o[obs::VERTICES], 0.0);
    assert_eq!(o[obs::VERTICES + 1], 1.0, "rel-0 city slot");
}

#[test]
fn visibility_differs_in_exactly_the_opponent_private_block() {
    // A midgame state with real hidden information.
    let mut game = CatanGame::new(4, 21);
    game.record_history = false;
    let mut players: Vec<RandomPlayer> = (0..4u64).map(RandomPlayer::new).collect();
    let mut valid = Vec::with_capacity(256);
    for _ in 0..400 {
        if game.is_game_over() {
            break;
        }
        game.fill_valid_actions(&mut valid);
        let idx = game.current_player();
        let action = players[idx].choose_action(&game, &valid);
        game.execute_action(&action);
    }

    let (mut perfect, mut realistic) = (buf(), buf());
    encode_obs(&game, 0, Visibility::Perfect, &mut perfect);
    encode_obs(&game, 0, Visibility::Realistic, &mut realistic);

    let mut diffs = Vec::new();
    for i in 0..OBS_DIM {
        if perfect[i] != realistic[i] {
            diffs.push(i);
        }
    }
    assert!(!diffs.is_empty(), "midgame must contain hidden information");
    for &i in &diffs {
        assert!(
            (obs::OPP_PRIVATE..obs::BANK).contains(&i),
            "index {i} differs outside the opponent-private block"
        );
        assert_eq!(realistic[i], 0.0, "realistic mode must zero, not alter");
    }
}

#[test]
fn deterministic_and_bounded_on_live_games() {
    let mut phases = std::collections::HashSet::new();
    for seed in [3u64, 14, 33] {
        let num_players = if seed % 2 == 0 { 3 } else { 4 };
        let mut game = CatanGame::new(num_players, seed);
        game.record_history = false;
        let mut players: Vec<RandomPlayer> = (0..num_players as u64)
            .map(|i| RandomPlayer::new(seed + i))
            .collect();
        let mut valid = Vec::with_capacity(256);
        let (mut a, mut b) = (buf(), buf());

        for step in 0..3000 {
            if game.is_game_over() || game.state.turn >= 1000 {
                break;
            }
            phases.insert((game.game_phase as u8, game.turn_phase as u8));
            if step % 7 == 0 {
                for seat in 0..num_players {
                    encode_obs(&game, seat, Visibility::Perfect, &mut a);
                    encode_obs(&game, seat, Visibility::Perfect, &mut b);
                    assert_eq!(a, b, "seed {seed}: encoding must be deterministic");
                    for (i, &v) in a.iter().enumerate() {
                        assert!(v.is_finite(), "seed {seed}: obs[{i}] not finite");
                        assert!(
                            (-0.001..=1.5).contains(&v),
                            "seed {seed} seat {seat}: obs[{i}] = {v} out of range"
                        );
                    }
                    // Self block must match the seat's actual hand.
                    for r in 0..5 {
                        assert!(
                            (a[obs::SELF_PRIVATE + r]
                                - game.state.resources[seat][r] as f32 / 19.0)
                                .abs()
                                < 1e-6
                        );
                    }
                }
            }
            game.fill_valid_actions(&mut valid);
            let idx = game.current_player();
            let action = players[idx].choose_action(&game, &valid);
            game.execute_action(&action);
        }
    }
    assert!(phases.len() >= 8, "only {} phases reached", phases.len());
}

#[test]
fn trade_offer_block_encodes_relative_to_the_viewer() {
    let mut game = CatanGame::new(4, 11);
    game.game_phase = GamePhase::Playing;
    game.state.phase = 1;
    game.turn_phase = TurnPhase::Main;
    game.state.current_player = 0;
    game.state.has_rolled = true;
    game.state.resources[0] = [2, 0, 0, 0, 0];
    game.state.bank[0] -= 2;
    game.state.resources[2] = [0, 1, 0, 0, 0];
    game.state.bank[1] -= 1;

    // Player 0 offers 2 wheat for 1 sheep; player 2 is the eligible responder.
    assert!(game.execute_action(&Action::ProposeTrade {
        player: 0,
        give: 0,
        give_amount: 2,
        recv: 1
    }));
    assert_eq!(game.current_player(), 2);

    let mut o = buf();
    encode_obs(&game, 2, Visibility::Realistic, &mut o);
    assert_eq!(o[obs::TRADE], 1.0, "offer present");
    // Proposer (seat 0) is rel 2 from seat 2's perspective.
    assert_eq!(o[obs::TRADE + 1 + 2], 1.0, "proposer rel one-hot");
    assert_eq!(o[obs::TRADE + 5], 1.0, "give = wheat");
    assert_eq!(o[obs::TRADE + 10], 1.0, "amount 2/2");
    assert_eq!(o[obs::TRADE + 11 + 1], 1.0, "recv = sheep");

    // No offer -> block is all zeros.
    assert!(game.execute_action(&Action::RespondTrade {
        player: 2,
        accept: false
    }));
    encode_obs(&game, 2, Visibility::Realistic, &mut o);
    for i in obs::TRADE..OBS_DIM {
        assert_eq!(o[i], 0.0, "trade block must clear after resolution");
    }
}
