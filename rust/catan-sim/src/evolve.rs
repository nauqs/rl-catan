//! Genetic algorithm over the heuristic's weights: fitness = win rate vs
//! three frozen Heuristic-v1 opponents. Every individual in a generation
//! faces the SAME game seeds (common random numbers), so fitness gaps
//! reflect skill, not board luck. Output: a Heuristic-v2 candidate.

use catan_core::game::CatanGame;
use catan_core::players::{play_game, HeuristicParams, HeuristicPlayer, Player};
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;

const POP: usize = 20;
const ELITE: usize = 4;
const GENS: usize = 12;

fn fitness(params: HeuristicParams, games: u64, seed_base: u64) -> f32 {
    let wins: u32 = (0..games)
        .map(|g| {
            let game_seed = seed_base.wrapping_add(g.wrapping_mul(0x9E3779B97F4A7C15));
            // Rotate the candidate's seat so seat advantage washes out.
            let my_seat = (g % 4) as usize;
            let mut game = CatanGame::new_with_target(4, game_seed, 7);
            game.record_history = false;
            let mut players: Vec<Box<dyn Player>> = (0..4)
                .map(|s| -> Box<dyn Player> {
                    if s == my_seat {
                        Box::new(HeuristicPlayer::with_params(game_seed ^ 1, params))
                    } else {
                        Box::new(HeuristicPlayer::new(game_seed ^ (s as u64 + 2)))
                    }
                })
                .collect();
            u32::from(play_game(&mut game, &mut players) == my_seat as i8)
        })
        .sum();
    wins as f32 / games as f32
}

fn mutate(params: HeuristicParams, rng: &mut SmallRng, sigma: f32) -> HeuristicParams {
    let defaults = HeuristicParams::default().to_array();
    let mut a = params.to_array();
    for (i, v) in a.iter_mut().enumerate() {
        // Gaussian-ish noise scaled to each gene's natural magnitude.
        let scale = defaults[i].abs().max(0.05);
        let noise: f32 = (0..4).map(|_| rng.gen::<f32>()).sum::<f32>() / 2.0 - 1.0;
        *v = (*v + noise * sigma * scale).max(0.0);
    }
    HeuristicParams::from_array(a)
}

fn crossover(a: HeuristicParams, b: HeuristicParams, rng: &mut SmallRng) -> HeuristicParams {
    let (aa, ba) = (a.to_array(), b.to_array());
    let mut child = [0f32; 6];
    for i in 0..6 {
        child[i] = if rng.gen::<bool>() { aa[i] } else { ba[i] };
    }
    HeuristicParams::from_array(child)
}

pub fn evolve(games_per_eval: u64, seed: u64) {
    let mut rng = SmallRng::seed_from_u64(seed);
    // Individual 0 is v1 itself: evolution must BEAT the incumbent to count.
    let mut population: Vec<HeuristicParams> = vec![HeuristicParams::default()];
    while population.len() < POP {
        let base = HeuristicParams::default();
        population.push(mutate(base, &mut rng, 0.6));
    }

    for gen in 0..GENS {
        let seed_base = seed + gen as u64 * 1_000_003; // CRN per generation
        let mut scored: Vec<(f32, HeuristicParams)> = population
            .par_iter()
            .map(|&p| (fitness(p, games_per_eval, seed_base), p))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let best = scored[0];
        let mean: f32 = scored.iter().map(|s| s.0).sum::<f32>() / scored.len() as f32;
        println!(
            "gen {gen:2}: best {:.1}% | mean {:.1}% | best params {:?}",
            best.0 * 100.0,
            mean * 100.0,
            best.1.to_array().map(|v| (v * 100.0).round() / 100.0),
        );

        let sigma = 0.4 * (1.0 - gen as f32 / GENS as f32) + 0.08;
        let elites: Vec<HeuristicParams> = scored.iter().take(ELITE).map(|s| s.1).collect();
        population = elites.clone();
        while population.len() < POP {
            let a = elites[rng.gen_range(0..ELITE)];
            let b = scored[rng.gen_range(0..POP / 2)].1;
            population.push(mutate(crossover(a, b, &mut rng), &mut rng, sigma));
        }
    }

    // Holdout: fresh seeds, larger sample, champion vs the v1 incumbent.
    let champion = {
        let seed_base = seed + 777_777;
        let mut scored: Vec<(f32, HeuristicParams)> = population
            .par_iter()
            .map(|&p| (fitness(p, games_per_eval, seed_base), p))
            .collect();
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        scored[0].1
    };
    let holdout_champ = fitness(champion, games_per_eval * 6, seed + 31_337);
    let holdout_v1 = fitness(HeuristicParams::default(), games_per_eval * 6, seed + 31_337);
    println!("\nholdout ({} fresh games, seat-rotated, vs 3x Heuristic-v1):", games_per_eval * 6);
    println!("  v1 baseline : {:.1}% (fair share = 25%)", holdout_v1 * 100.0);
    println!("  evolved     : {:.1}%", holdout_champ * 100.0);
    println!("  params      : {:?}", champion);
    let json = format!(
        "{{\"diversity_bonus\":{},\"port_bonus\":{},\"robber_steal_bonus\":{},\
         \"robber_self_penalty\":{},\"steal_cards_weight\":{},\"trade_accept_threshold\":{},\
         \"holdout_win_rate\":{}}}",
        champion.diversity_bonus, champion.port_bonus, champion.robber_steal_bonus,
        champion.robber_self_penalty, champion.steal_cards_weight,
        champion.trade_accept_threshold, holdout_champ,
    );
    std::fs::write("heuristic_v2_candidate.json", &json).ok();
    println!("written to heuristic_v2_candidate.json");
}
