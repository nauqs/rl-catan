//! Board topology: pre-computed adjacency for the standard 19-tile board.
//!
//! Numbering matches the Python engine: tiles row by row (3-4-5-4-3),
//! vertices in first-encounter order iterating tiles then their 6 corners
//! clockwise from the top, edges in sorted (v1, v2) order.

use std::collections::BTreeSet;
use std::sync::OnceLock;

pub const NUM_TILES: usize = 19;
pub const NUM_VERTICES: usize = 54;
pub const NUM_EDGES: usize = 72;
pub const NUM_PORTS: usize = 9;

pub const RESOURCE_WHEAT: i8 = 0;
pub const RESOURCE_SHEEP: i8 = 1;
pub const RESOURCE_WOOD: i8 = 2;
pub const RESOURCE_BRICK: i8 = 3;
pub const RESOURCE_STONE: i8 = 4;
pub const RESOURCE_DESERT: i8 = 5;

pub const TILE_RESOURCES: [i8; 19] = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5];
pub const TILE_NUMBERS: [i8; 19] = [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12, 0];
pub const PORT_TYPE_ANY: i8 = 0;
pub const PORT_TYPES: [i8; 9] = [0, 0, 0, 0, 1, 2, 3, 4, 5];

pub struct Topology {
    pub tile_vertices: [[u8; 6]; NUM_TILES],
    pub vertex_tiles: [[i8; 3]; NUM_VERTICES],
    pub vertex_neighbors: [[i8; 3]; NUM_VERTICES],
    pub edge_vertices: [[u8; 2]; NUM_EDGES],
    pub vertex_edges: [[i8; 3]; NUM_VERTICES],
    /// Bitmask (bit v) of the vertices adjacent to each vertex.
    pub neighbor_mask: [u64; NUM_VERTICES],
    pub port_vertices: [[u8; 2]; NUM_PORTS],
    /// Port slot index (0-8) touching each vertex, or -1. The port's
    /// *type* is per-game: `GameState::vertex_port_type` maps through the
    /// shuffled `port_types`.
    pub vertex_port_index: [i8; NUM_VERTICES],
    /// Probability weight (out of 36) for each dice total 0-12; 7 is 0.
    pub number_probabilities: [f32; 13],
}

pub fn topology() -> &'static Topology {
    static TOPO: OnceLock<Topology> = OnceLock::new();
    TOPO.get_or_init(Topology::build)
}

impl Topology {
    fn build() -> Topology {
        // Integer hex geometry: x in units of half-width h, y in units of R/2.
        // Pointy-top corner offsets clockwise from top.
        const CORNERS: [(i32, i32); 6] = [(0, -2), (1, -1), (1, 1), (0, 2), (-1, 1), (-1, -1)];
        const ROWS: [usize; 5] = [3, 4, 5, 4, 3];

        let mut coord_to_vertex: std::collections::HashMap<(i32, i32), u8> =
            std::collections::HashMap::new();
        let mut tile_vertices = [[0u8; 6]; NUM_TILES];
        let mut tile = 0usize;
        for (row, &n) in ROWS.iter().enumerate() {
            let y = 3 * (row as i32 - 2);
            let x_start = -((n as i32) - 1);
            for col in 0..n {
                let cx = x_start + 2 * col as i32;
                for (i, (dx, dy)) in CORNERS.iter().enumerate() {
                    let key = (cx + dx, y + dy);
                    let next = coord_to_vertex.len() as u8;
                    let id = *coord_to_vertex.entry(key).or_insert(next);
                    tile_vertices[tile][i] = id;
                }
                tile += 1;
            }
        }
        assert_eq!(coord_to_vertex.len(), NUM_VERTICES);

        let mut vertex_tiles = [[-1i8; 3]; NUM_VERTICES];
        for t in 0..NUM_TILES {
            for &v in &tile_vertices[t] {
                let slots = &mut vertex_tiles[v as usize];
                if let Some(slot) = slots.iter_mut().find(|s| **s == -1) {
                    *slot = t as i8;
                }
            }
        }

        let mut neighbor_sets: Vec<BTreeSet<u8>> = vec![BTreeSet::new(); NUM_VERTICES];
        let mut edge_set: BTreeSet<(u8, u8)> = BTreeSet::new();
        for t in 0..NUM_TILES {
            for i in 0..6 {
                let v1 = tile_vertices[t][i];
                let v2 = tile_vertices[t][(i + 1) % 6];
                neighbor_sets[v1 as usize].insert(v2);
                neighbor_sets[v2 as usize].insert(v1);
                edge_set.insert((v1.min(v2), v1.max(v2)));
            }
        }
        assert_eq!(edge_set.len(), NUM_EDGES);

        let mut vertex_neighbors = [[-1i8; 3]; NUM_VERTICES];
        for (v, nbrs) in neighbor_sets.iter().enumerate() {
            for (i, &n) in nbrs.iter().take(3).enumerate() {
                vertex_neighbors[v][i] = n as i8;
            }
        }

        let mut edge_vertices = [[0u8; 2]; NUM_EDGES];
        for (e, &(v1, v2)) in edge_set.iter().enumerate() {
            edge_vertices[e] = [v1, v2];
        }

        let mut vertex_edges = [[-1i8; 3]; NUM_VERTICES];
        for e in 0..NUM_EDGES {
            for &v in &edge_vertices[e] {
                let slots = &mut vertex_edges[v as usize];
                if let Some(slot) = slots.iter_mut().find(|s| **s == -1) {
                    *slot = e as i8;
                }
            }
        }

        let mut neighbor_mask = [0u64; NUM_VERTICES];
        for v in 0..NUM_VERTICES {
            for &n in &vertex_neighbors[v] {
                if n >= 0 {
                    neighbor_mask[v] |= 1u64 << n;
                }
            }
        }

        let port_vertices: [[u8; 2]; NUM_PORTS] = [
            [0, 1],
            [7, 10],
            [12, 22],
            [35, 36],
            [45, 46],
            [50, 53],
            [47, 48],
            [39, 40],
            [27, 28],
        ];
        let mut vertex_port_index = [-1i8; NUM_VERTICES];
        for (p, verts) in port_vertices.iter().enumerate() {
            for &v in verts {
                vertex_port_index[v as usize] = p as i8;
            }
        }

        let mut number_probabilities = [0f32; 13];
        let weights = [0, 0, 1, 2, 3, 4, 5, 0, 5, 4, 3, 2, 1];
        for (i, &w) in weights.iter().enumerate() {
            number_probabilities[i] = w as f32 / 36.0;
        }

        Topology {
            tile_vertices,
            vertex_tiles,
            vertex_neighbors,
            edge_vertices,
            vertex_edges,
            neighbor_mask,
            port_vertices,
            vertex_port_index,
            number_probabilities,
        }
    }
}
