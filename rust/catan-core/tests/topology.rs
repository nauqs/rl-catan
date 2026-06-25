//! Board topology must match the Python engine's numbering exactly.
//! The fixture was captured once from the original Python engine's
//! BoardTopology and is permanent: the board numbering is frozen.

use catan_core::board::{topology, NUM_EDGES, NUM_TILES, NUM_VERTICES};

fn fixture() -> serde_json::Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/golden/topology.json");
    let data = std::fs::read_to_string(path)
        .expect("topology fixture missing from tests/golden/topology.json");
    serde_json::from_str(&data).unwrap()
}

fn as_i64_matrix(v: &serde_json::Value) -> Vec<Vec<i64>> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|row| {
            row.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_i64().unwrap())
                .collect()
        })
        .collect()
}

#[test]
fn tile_vertices_match_python() {
    let fix = fixture();
    let expected = as_i64_matrix(&fix["tile_vertices"]);
    let topo = topology();
    for t in 0..NUM_TILES {
        for i in 0..6 {
            assert_eq!(
                topo.tile_vertices[t][i] as i64, expected[t][i],
                "tile {t} slot {i}"
            );
        }
    }
}

#[test]
fn edge_vertices_match_python() {
    let fix = fixture();
    let expected = as_i64_matrix(&fix["edge_vertices"]);
    let topo = topology();
    for e in 0..NUM_EDGES {
        assert_eq!(topo.edge_vertices[e][0] as i64, expected[e][0], "edge {e}");
        assert_eq!(topo.edge_vertices[e][1] as i64, expected[e][1], "edge {e}");
    }
}

#[test]
fn vertex_adjacency_matches_python() {
    let fix = fixture();
    let topo = topology();
    let neighbors = as_i64_matrix(&fix["vertex_neighbors"]);
    let tiles = as_i64_matrix(&fix["vertex_tiles"]);
    let edges = as_i64_matrix(&fix["vertex_edges"]);
    for v in 0..NUM_VERTICES {
        for i in 0..3 {
            assert_eq!(
                topo.vertex_neighbors[v][i] as i64, neighbors[v][i],
                "neighbors v{v} s{i}"
            );
            assert_eq!(
                topo.vertex_tiles[v][i] as i64, tiles[v][i],
                "tiles v{v} s{i}"
            );
            assert_eq!(
                topo.vertex_edges[v][i] as i64, edges[v][i],
                "edges v{v} s{i}"
            );
        }
    }
}

#[test]
fn ports_match_python() {
    let fix = fixture();
    let topo = topology();
    let ports = as_i64_matrix(&fix["port_vertices"]);
    for p in 0..9 {
        assert_eq!(topo.port_vertices[p][0] as i64, ports[p][0]);
        assert_eq!(topo.port_vertices[p][1] as i64, ports[p][1]);
    }
    // Each port's two vertices carry that port's slot index; all others -1.
    let mut expected_index = [-1i64; NUM_VERTICES];
    for (p, verts) in ports.iter().enumerate() {
        for &v in verts {
            expected_index[v as usize] = p as i64;
        }
    }
    for v in 0..NUM_VERTICES {
        assert_eq!(
            topo.vertex_port_index[v] as i64, expected_index[v],
            "port index v{v}"
        );
    }
}

#[test]
fn structural_invariants() {
    let topo = topology();
    // Every edge connects two distinct, valid, adjacent vertices.
    for e in 0..NUM_EDGES {
        let [v1, v2] = topo.edge_vertices[e];
        assert!(v1 < v2, "edge vertices are sorted");
        assert!((v2 as usize) < NUM_VERTICES);
        assert!(topo.vertex_neighbors[v1 as usize].contains(&(v2 as i8)));
    }
    // vertex_edges is consistent with edge_vertices, and every vertex has 2-3 edges.
    for v in 0..NUM_VERTICES {
        let count = topo.vertex_edges[v].iter().filter(|&&e| e >= 0).count();
        assert!((2..=3).contains(&count), "vertex {v} has {count} edges");
        for &e in topo.vertex_edges[v].iter().filter(|&&e| e >= 0) {
            assert!(topo.edge_vertices[e as usize].contains(&(v as u8)));
        }
    }
}
