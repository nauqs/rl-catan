//! Times the net forward pass against a real CTNN file, on a real
//! mid-game observation (sparsity matters for the column-skip path).
//!
//!     cargo run --release -p catan-env --example netbench -- models/catan-512.ctnn

use std::hint::black_box;
use std::time::Instant;

use catan_core::game::{Action, CatanGame};
use catan_core::players::{Player, RandomPlayer};
use catan_env::net::{MlpNet, NetScratch};
use catan_env::obs::{encode_obs, Visibility, OBS_DIM};

fn main() {
    let path = std::env::args().nth(1).expect("usage: netbench <net.ctnn>");
    let net = MlpNet::load(std::path::Path::new(&path));
    let mut scratch = NetScratch::new(&net);

    // Play a random game forward a while so the board is realistically full.
    let mut game = CatanGame::new(4, 7);
    let mut player = RandomPlayer::new(12345);
    let mut valid: Vec<Action> = Vec::new();
    for _ in 0..400 {
        if game.is_game_over() {
            break;
        }
        game.fill_valid_actions(&mut valid);
        if valid.is_empty() {
            break;
        }
        let action = player.choose_action(&game, &valid);
        game.execute_action(&action);
    }
    let mut obs = vec![0.0f32; OBS_DIM];
    encode_obs(&game, 0, Visibility::Perfect, &mut obs);
    let nonzero = obs.iter().filter(|&&x| x != 0.0).count();

    const WARMUP: u32 = 50;
    const ITERS: u32 = 500;
    for _ in 0..WARMUP {
        net.trunk(black_box(&obs), &mut scratch);
    }
    let t = Instant::now();
    let mut sink = 0.0f32;
    for _ in 0..ITERS {
        net.trunk(black_box(&obs), &mut scratch);
        sink += net.value_from(&scratch);
    }
    let dt = t.elapsed();
    let macs = (OBS_DIM * net.hidden + net.hidden * net.hidden) as f64;
    println!(
        "obs nonzero: {nonzero}/{OBS_DIM}; trunk+value: {:.1} us/call, {:.2} dense-GMAC/s (hidden {}, sink {sink:.3})",
        dt.as_micros() as f64 / ITERS as f64,
        macs * ITERS as f64 / dt.as_nanos() as f64,
        net.hidden,
    );
}
