use crate::{CELLS_PER_METER, PixelBody};
use fe_core::{CHUNK_SIZE, Cell, ChunkPos, MaterialId};
use fe_materials::State;
use fe_sim::World;
use rapier2d::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Ticks a terrain collider is kept after no body is near it.
const IDLE_TICKS: u32 = 120;
/// Margin around each body, in cells, inside which terrain gets colliders.
const MARGIN: f32 = 12.0;

/// The collider for one chunk's static cells.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(crate) struct TerrainCollider {
    handle: Option<ColliderHandle>,
    idle: u32,
}

/// Builds, refreshes and drops terrain colliders around bodies.
pub(crate) fn update(
    map: &mut BTreeMap<ChunkPos, TerrainCollider>,
    rapier: &mut PhysicsWorld,
    terrain_body: RigidBodyHandle,
    bodies: &[PixelBody],
    world: &mut World,
) {
    let mut needed = BTreeSet::new();
    for b in bodies {
        let Some(rb) = rapier.bodies.get(b.handle) else { continue };
        let t = rb.translation() * CELLS_PER_METER;
        let (w, h) = b.size();
        let r = f32::from(w.max(h)) + MARGIN;
        let a = fe_core::CellPos::new((t.x - r).floor() as i32, (t.y - r).floor() as i32).chunk();
        let z = fe_core::CellPos::new((t.x + r).floor() as i32, (t.y + r).floor() as i32).chunk();
        for cy in a.y..=z.y {
            for cx in a.x..=z.x {
                needed.insert(ChunkPos::new(cx, cy));
            }
        }
    }

    for &pos in &needed {
        let Some(chunk) = world.grid.chunk_mut(pos) else { continue };
        let changed = chunk.take_solid_changed();
        let entry = map.get_mut(&pos);
        match entry {
            Some(tc) if !changed => tc.idle = 0,
            _ => {
                if let Some(old) = map.remove(&pos).and_then(|t| t.handle) {
                    rapier.colliders.remove(
                        old,
                        &mut rapier.islands,
                        &mut rapier.bodies,
                        &mut rapier.soft_bodies,
                        true,
                    );
                }
                let handle = build(world, pos).map(|c| {
                    rapier.colliders.insert_with_parent(c, terrain_body, &mut rapier.bodies)
                });
                map.insert(pos, TerrainCollider { handle, idle: 0 });
            }
        }
    }

    let stale: Vec<ChunkPos> = map
        .iter_mut()
        .filter(|(p, _)| !needed.contains(*p))
        .filter_map(|(p, tc)| {
            tc.idle += 1;
            (tc.idle > IDLE_TICKS).then_some(*p)
        })
        .collect();
    for p in stale {
        if let Some(h) = map.remove(&p).and_then(|t| t.handle) {
            rapier.colliders.remove(
                h,
                &mut rapier.islands,
                &mut rapier.bodies,
                &mut rapier.soft_bodies,
                true,
            );
        }
    }
}

fn build(world: &World, pos: ChunkPos) -> Option<Collider> {
    let chunk = world.grid.chunk(pos)?;
    let origin = pos.origin();
    let mut keys = Vec::new();
    for i in 0..fe_core::CHUNK_AREA {
        let c = chunk.get(i);
        if c.is_air() || c.has(Cell::FLAG_IN_BODY) {
            continue;
        }
        let static_cell = c.material() == MaterialId::BOUNDARY
            || world.materials().get(c.material()).state == State::Static;
        if static_cell {
            let lx = i as i32 % CHUNK_SIZE;
            let ly = i as i32 / CHUNK_SIZE;
            keys.push(IVector::new(origin.x + lx, origin.y + ly));
        }
    }
    if keys.is_empty() {
        return None;
    }
    Some(
        ColliderBuilder::voxels(Vector::splat(1.0 / CELLS_PER_METER), &keys)
            .friction(0.8)
            .restitution(0.05)
            .build(),
    )
}
