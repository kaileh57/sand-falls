//! Structural support: finds structure that lost its connection to an anchor.

use crate::world::World;
use fe_core::{Cell, CellPos, MaterialId, PosSet};
use fe_materials::State;

/// Largest component a single search explores before treating it as supported.
pub const SUPPORT_SEARCH_LIMIT: usize = 65_536;
/// How many ticks a too-large search is retried before giving up.
pub const SUPPORT_RETRIES: u8 = 3;

/// A connected group of structural cells with no path to an anchor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Detached {
    /// Cells in search order.
    pub cells: Vec<CellPos>,
}

/// A seed waiting for a support check, with the number of retries left.
#[derive(
    Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
pub struct SupportSeed {
    /// Where structure was removed.
    pub pos: CellPos,
    /// Retries left after a search hits the size limit.
    pub retries: u8,
}

enum Search {
    Supported,
    TooLarge,
    Detached(Vec<CellPos>),
}

fn is_structure(world: &World, c: Cell) -> bool {
    if c.has(Cell::FLAG_IN_BODY) || c.is_air() {
        return false;
    }
    let m = world.materials().get(c.material());
    m.state == State::Static && (m.structural || m.anchored)
}

fn is_anchor(world: &World, c: Cell) -> bool {
    c.material() == MaterialId::BOUNDARY || world.materials().get(c.material()).anchored
}

fn search(world: &World, start: CellPos, limit: usize, seen: &mut PosSet) -> Search {
    let mut visited = PosSet::with_capacity(256);
    let mut order = vec![start];
    visited.insert(start);
    let mut head = 0;
    while head < order.len() {
        let p = order[head];
        head += 1;
        for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
            let q = CellPos::new(p.x + dx, p.y + dy);
            if visited.contains(q) {
                continue;
            }
            let c = world.get(q);
            if is_anchor(world, c) || seen.contains(q) {
                // Everything reached so far connects to an anchor too.
                for &v in &order {
                    seen.insert(v);
                }
                return Search::Supported;
            }
            if !is_structure(world, c) {
                continue;
            }
            visited.insert(q);
            order.push(q);
            if order.len() > limit {
                return Search::TooLarge;
            }
        }
    }
    Search::Detached(order)
}

/// How far, in cells, decorations on a falling piece go with it.
const DECORATION_REACH: u16 = 24;

/// Moss, fungus, vines and leaves: static but not structure. They hold
/// nothing up, and they belong to whatever they grow on.
fn is_decoration(world: &World, c: Cell) -> bool {
    if c.has(Cell::FLAG_IN_BODY) || c.is_air() || c.material() == MaterialId::BOUNDARY {
        return false;
    }
    let m = world.materials().get(c.material());
    m.state == State::Static && !m.structural && !m.anchored && !m.indestructible
}

/// Adds the decorations growing on a detached piece, up to
/// [`DECORATION_REACH`] cells out, so they fall with it instead of staying
/// behind as static cells that pin the new body in mid-air.
fn take_decorations(world: &World, cells: &mut Vec<CellPos>, claimed: &mut PosSet) {
    let n = cells.len();
    let mut depth = vec![0u16; n];
    let mut head = 0;
    while head < cells.len() {
        let p = cells[head];
        let d = depth[head];
        head += 1;
        if d >= DECORATION_REACH {
            continue;
        }
        for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
            let q = CellPos::new(p.x + dx, p.y + dy);
            if claimed.contains(q) || !is_decoration(world, world.get(q)) {
                continue;
            }
            claimed.insert(q);
            cells.push(q);
            depth.push(d + 1);
        }
    }
}

/// Checks the neighbors of each seed and returns structure that no longer reaches an anchor.
///
/// `seeds` must be in canonical order. Seeds whose search hit the size limit are
/// returned for a retry on the next tick while they have retries left.
pub fn check_support(
    world: &World,
    seeds: &[SupportSeed],
    limit: usize,
) -> (Vec<Detached>, Vec<SupportSeed>) {
    let mut supported = PosSet::with_capacity(1024);
    let mut claimed = PosSet::with_capacity(256);
    let mut detached = Vec::new();
    let mut retry = Vec::new();
    for seed in seeds {
        for (dx, dy) in [(0, 1), (1, 0), (0, -1), (-1, 0)] {
            let n = CellPos::new(seed.pos.x + dx, seed.pos.y + dy);
            if claimed.contains(n) || supported.contains(n) {
                continue;
            }
            let c = world.get(n);
            if !is_structure(world, c) || is_anchor(world, c) {
                continue;
            }
            match search(world, n, limit, &mut supported) {
                Search::Supported => {
                    supported.insert(n);
                }
                Search::TooLarge => {
                    supported.insert(n);
                    if seed.retries > 0 {
                        retry.push(SupportSeed { pos: seed.pos, retries: seed.retries - 1 });
                    }
                }
                Search::Detached(mut cells) => {
                    for &p in &cells {
                        claimed.insert(p);
                    }
                    take_decorations(world, &mut cells, &mut claimed);
                    detached.push(Detached { cells });
                }
            }
        }
    }
    retry.sort();
    retry.dedup();
    (detached, retry)
}

#[cfg(test)]
mod tests {
    use super::*;
    use fe_core::IRect;
    use std::sync::Arc;

    fn setup() -> (World, MaterialId, MaterialId) {
        let mats = Arc::new(fe_materials::base_set());
        let stone = mats.id("stone").unwrap_or_default();
        let bedrock = mats.id("bedrock").unwrap_or_default();
        let mut w = World::new(mats, 1, 1);
        w.ensure_rect(IRect::new(0, 0, 128, 128), Cell::AIR);
        (w, stone, bedrock)
    }

    #[test]
    fn cutting_a_pillar_detaches_the_top() {
        let (mut w, stone, bedrock) = setup();
        w.fill_rect(IRect::new(0, 120, 128, 128), bedrock);
        w.fill_rect(IRect::new(60, 60, 64, 120), stone);
        w.fill_rect(IRect::new(40, 56, 84, 60), stone);
        let mut seeds = Vec::new();
        for x in 60..64 {
            if let Some(crate::Event::SupportLost(p)) = w.set(CellPos::new(x, 100), Cell::AIR) {
                seeds.push(SupportSeed { pos: p, retries: SUPPORT_RETRIES });
            }
        }
        assert_eq!(seeds.len(), 4);
        let (det, retry) = check_support(&w, &seeds, SUPPORT_SEARCH_LIMIT);
        assert!(retry.is_empty());
        assert_eq!(det.len(), 1);
        // Pillar above the cut (rows 60..100) plus the beam.
        assert_eq!(det[0].cells.len(), 4 * 40 + 44 * 4);
    }

    #[test]
    fn anchored_structure_stays() {
        let (mut w, stone, bedrock) = setup();
        w.fill_rect(IRect::new(0, 120, 128, 128), bedrock);
        w.fill_rect(IRect::new(60, 60, 64, 120), stone);
        let seeds = [SupportSeed { pos: CellPos::new(59, 90), retries: 0 }];
        let (det, _) = check_support(&w, &seeds, SUPPORT_SEARCH_LIMIT);
        assert!(det.is_empty());
    }

    #[test]
    fn oversized_search_is_retried() {
        let (mut w, stone, _) = setup();
        w.fill_rect(IRect::new(10, 10, 110, 110), stone);
        w.set(CellPos::new(50, 50), Cell::AIR);
        let seeds = [SupportSeed { pos: CellPos::new(50, 50), retries: 2 }];
        let (det, retry) = check_support(&w, &seeds, 100);
        assert!(det.is_empty());
        assert_eq!(retry, vec![SupportSeed { pos: CellPos::new(50, 50), retries: 1 }]);
    }
}
