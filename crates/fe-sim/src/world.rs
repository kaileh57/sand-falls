use crate::grid::Grid;
use crate::rules::{self, Event, JobOut, RuleCtx};
use crate::sched::{self, Chunk};
use fe_core::{Cell, CellPos, ChunkPos, IRect, LocalRect, MaterialId, StableHasher};
use fe_materials::{MaterialSet, State};
use std::sync::Arc;

/// Counters from the most recent cell step.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CellStats {
    /// Chunks that had a non-empty dirty rect.
    pub active_chunks: u32,
    /// Loaded chunks.
    pub loaded_chunks: u32,
    /// Cells visited by the rules.
    pub visited_cells: u64,
}

/// The cell world: material set, chunks, seed, tick, and the thread pool.
pub struct World {
    mats: Arc<MaterialSet>,
    fast: Vec<rules::Fast>,
    /// Loaded chunks.
    pub grid: Grid,
    seed: u64,
    tick: u64,
    threads: usize,
    pool: Option<rayon::ThreadPool>,
    stats: CellStats,
    wind: i8,
}

impl World {
    /// Creates an empty world. `threads` of 0 or 1 runs passes on the calling thread.
    pub fn new(mats: Arc<MaterialSet>, seed: u64, threads: usize) -> Self {
        let fast = rules::fast_table(&mats);
        let mut w = Self {
            mats,
            fast,
            grid: Grid::default(),
            seed,
            tick: 0,
            threads: 1,
            pool: None,
            stats: CellStats::default(),
            wind: 0,
        };
        w.set_threads(threads);
        w
    }

    /// Changes the number of worker threads. Results do not depend on this value.
    pub fn set_threads(&mut self, threads: usize) {
        self.threads = threads.max(1);
        self.pool = if self.threads > 1 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(self.threads)
                .thread_name(|i| format!("fe-sim-{i}"))
                .build()
                .ok()
        } else {
            None
        };
    }

    /// Worker thread count.
    pub const fn threads(&self) -> usize {
        self.threads
    }

    /// The material set.
    pub fn materials(&self) -> &Arc<MaterialSet> {
        &self.mats
    }

    /// The world seed.
    pub const fn seed(&self) -> u64 {
        self.seed
    }

    /// The current tick: the number of steps run so far.
    pub const fn tick(&self) -> u64 {
        self.tick
    }

    /// Sets the tick counter, used when loading a snapshot.
    pub const fn set_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Wind from -4 (strong, to the left) to 4 (strong, to the right). Gases and fire drift with it.
    pub const fn wind(&self) -> i8 {
        self.wind
    }

    /// Sets the wind, clamped to -4..=4.
    pub fn set_wind(&mut self, w: i8) {
        self.wind = w.clamp(-4, 4);
    }

    /// Counters from the last [`World::step_cells`].
    pub const fn stats(&self) -> CellStats {
        self.stats
    }

    /// Creates every chunk overlapping `rect` that is not already loaded, filled with `fill`.
    pub fn ensure_rect(&mut self, rect: IRect, fill: Cell) {
        if rect.is_empty() {
            return;
        }
        let (a, b) = rect.chunk_range();
        for cy in a.y..b.y {
            for cx in a.x..b.x {
                let p = ChunkPos::new(cx, cy);
                if !self.grid.contains(p) {
                    self.grid.insert(Chunk::new(p, fill));
                }
            }
        }
    }

    /// Reads a cell. Unloaded cells read as the boundary.
    #[inline]
    pub fn get(&self, p: CellPos) -> Cell {
        self.grid.get(p)
    }

    /// Creates a fresh cell of a material at a position, with the material's defaults.
    pub fn make_cell(&self, id: MaterialId, p: CellPos) -> Cell {
        let r = fe_core::cell_rand(self.seed, self.tick, p.x, p.y, 0x5e7);
        rules::new_cell(&self.mats, id, None, r)
    }

    /// Writes a cell, marks it dirty, and returns any support event the change causes.
    ///
    /// Use this for edits made outside the cell rules: tools, explosions, scene loading.
    pub fn set(&mut self, p: CellPos, cell: Cell) -> Option<Event> {
        let old = self.grid.get(p);
        if old.material() == MaterialId::BOUNDARY
            && old == Cell::BOUNDARY
            && !self.grid.contains(p.chunk())
        {
            return None;
        }
        self.grid.set(p, cell);
        let om = self.mats.get(old.material());
        let nm = self.mats.get(cell.material());
        if (om.state == State::Static || nm.state == State::Static)
            && let Some(c) = self.grid.chunk(p.chunk())
        {
            c.flag_solid_changed();
        }
        let was = om.state == State::Static && om.structural && !old.has(Cell::FLAG_IN_BODY);
        let is = nm.state == State::Static && nm.structural;
        (was && !is).then_some(Event::SupportLost(p))
    }

    /// Fills a rectangle with fresh cells of one material. Returns support events.
    pub fn fill_rect(&mut self, rect: IRect, id: MaterialId) -> Vec<Event> {
        let mut events = Vec::new();
        for y in rect.min.y..rect.max.y {
            for x in rect.min.x..rect.max.x {
                let p = CellPos::new(x, y);
                let c = self.make_cell(id, p);
                events.extend(self.set(p, c));
            }
        }
        events
    }

    /// Runs one tick of the cell rules: four checkerboard passes over dirty chunks.
    ///
    /// Returns the events the rules produced, in canonical order. Does not advance
    /// the tick counter; the caller does that once the whole tick is done.
    pub fn step_cells(&mut self) -> Vec<Event> {
        self.grid.advance();
        let mut groups: [Vec<(ChunkPos, LocalRect)>; 4] = Default::default();
        let mut active = 0u32;
        for c in self.grid.chunks() {
            let d = c.dirty();
            if !d.is_empty() {
                groups[c.pos().group()].push((c.pos(), d));
                active += 1;
            }
        }
        let ctx = RuleCtx::new(&self.mats, &self.fast, self.seed, self.tick).with_wind(self.wind);
        let mut events = Vec::new();
        let mut visited = 0u64;
        let offset = (self.tick % 4) as usize;
        for k in 0..4 {
            let jobs = &groups[(offset + k) % 4];
            if jobs.is_empty() {
                continue;
            }
            let mut outs: Vec<JobOut> = (0..jobs.len()).map(|_| JobOut::default()).collect();
            sched::run_pass(
                &mut self.grid,
                jobs,
                &mut outs,
                self.pool.as_ref(),
                |nb, rect, out| {
                    rules::update_chunk(nb, rect, &ctx, out);
                },
            );
            for o in outs {
                visited += o.visited;
                events.extend(o.events);
            }
        }
        self.stats = CellStats {
            active_chunks: active,
            loaded_chunks: self.grid.len() as u32,
            visited_cells: visited,
        };
        events
    }

    /// Advances the tick counter by one.
    pub const fn finish_tick(&mut self) {
        self.tick += 1;
    }

    /// Hash of the tick, seed and every loaded cell.
    pub fn hash(&self) -> u64 {
        let mut h = StableHasher::new();
        h.write_u64(self.tick);
        h.write_u64(self.seed);
        h.write_i32(i32::from(self.wind));
        h.write_u64(self.mats.hash());
        h.write_u64(self.grid.hash());
        h.finish()
    }

    /// Counts cells of a material inside a rectangle.
    pub fn count(&self, rect: IRect, id: MaterialId) -> u64 {
        let mut n = 0;
        for y in rect.min.y..rect.max.y {
            for x in rect.min.x..rect.max.x {
                if self.grid.get(CellPos::new(x, y)).material() == id {
                    n += 1;
                }
            }
        }
        n
    }

    /// Counts every cell of each material across loaded chunks, indexed by material id.
    pub fn census(&self) -> Vec<u64> {
        let mut counts = vec![0u64; self.mats.len()];
        for c in self.grid.chunks() {
            for i in 0..fe_core::CHUNK_AREA {
                let id = c.get(i).material().index();
                if let Some(n) = counts.get_mut(id) {
                    *n += 1;
                }
            }
        }
        counts
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world(threads: usize) -> World {
        let mats = Arc::new(fe_materials::base_set());
        let mut w = World::new(mats, 7, threads);
        w.ensure_rect(IRect::new(0, 0, 256, 192), Cell::AIR);
        w
    }

    fn id(w: &World, name: &str) -> MaterialId {
        w.materials().id(name).unwrap_or_default()
    }

    fn run(w: &mut World, ticks: u32) {
        for _ in 0..ticks {
            w.step_cells();
            w.finish_tick();
        }
    }

    #[test]
    fn sand_falls_and_piles() {
        let mut w = world(1);
        let sand = id(&w, "sand");
        let stone = id(&w, "stone");
        w.fill_rect(IRect::new(0, 180, 256, 192), stone);
        w.fill_rect(IRect::new(100, 10, 120, 30), sand);
        run(&mut w, 300);
        assert_eq!(w.count(IRect::new(0, 0, 256, 192), sand), 400);
        assert_eq!(w.count(IRect::new(0, 0, 256, 150), sand), 0, "sand should settle on the floor");
        assert!(w.count(IRect::new(80, 170, 140, 180), sand) > 200);
    }

    #[test]
    fn water_conserves_mass_and_levels() {
        let mut w = world(1);
        let water = id(&w, "water");
        let stone = id(&w, "stone");
        w.fill_rect(IRect::new(0, 180, 256, 192), stone);
        w.fill_rect(IRect::new(0, 100, 4, 180), stone);
        w.fill_rect(IRect::new(252, 100, 256, 180), stone);
        w.fill_rect(IRect::new(10, 20, 60, 120), water);
        run(&mut w, 4000);
        assert_eq!(w.count(IRect::new(0, 0, 256, 192), water), 5000);
        // 5000 cells over a 248-wide floor is about 20 rows deep (surface at y 160).
        assert_eq!(w.count(IRect::new(0, 0, 256, 155), water), 0);
        assert_eq!(w.count(IRect::new(4, 165, 252, 180), water), 248 * 15);
        // A partly filled top row keeps shimmering, but only the surface stays awake.
        assert!(w.stats().visited_cells < 3000, "visited {}", w.stats().visited_cells);
    }

    #[test]
    fn world_sleeps_after_settling() {
        let mut w = world(1);
        let sand = id(&w, "sand");
        let stone = id(&w, "stone");
        w.fill_rect(IRect::new(0, 180, 256, 192), stone);
        w.fill_rect(IRect::new(100, 10, 120, 30), sand);
        run(&mut w, 600);
        assert_eq!(w.stats().active_chunks, 0);
    }

    #[test]
    fn thread_count_does_not_change_results() {
        let build = |threads| {
            let mut w = world(threads);
            let (sand, water, lava, wood) =
                (id(&w, "sand"), id(&w, "water"), id(&w, "lava"), id(&w, "wood"));
            w.fill_rect(IRect::new(0, 170, 256, 192), id(&w, "stone"));
            w.fill_rect(IRect::new(20, 10, 90, 60), sand);
            w.fill_rect(IRect::new(100, 20, 180, 90), water);
            w.fill_rect(IRect::new(190, 30, 230, 50), lava);
            w.fill_rect(IRect::new(140, 150, 200, 170), wood);
            run(&mut w, 400);
            w.hash()
        };
        let one = build(1);
        assert_eq!(one, build(2));
        assert_eq!(one, build(4));
    }

    #[test]
    fn water_on_lava_makes_obsidian_and_steam() {
        let mut w = world(1);
        w.fill_rect(IRect::new(0, 180, 256, 192), id(&w, "stone"));
        w.fill_rect(IRect::new(100, 160, 160, 180), id(&w, "lava"));
        w.fill_rect(IRect::new(100, 120, 160, 140), id(&w, "water"));
        run(&mut w, 200);
        let census = w.census();
        assert!(census[id(&w, "obsidian").index()] > 0);
        assert!(census[id(&w, "steam").index()] + census[id(&w, "water").index()] > 0);
    }

    #[test]
    fn wood_burns_to_smoke_and_a_little_ash() {
        let mut w = world(1);
        let wood = id(&w, "wood");
        w.fill_rect(IRect::new(0, 180, 256, 192), id(&w, "stone"));
        w.fill_rect(IRect::new(100, 150, 140, 180), wood);
        w.fill_rect(IRect::new(118, 148, 122, 150), id(&w, "burning_wood"));
        let mut smoke_seen = 0;
        for _ in 0..15 {
            run(&mut w, 100);
            smoke_seen = smoke_seen.max(w.census()[id(&w, "smoke").index()]);
        }
        let census = w.census();
        let left = census[wood.index()] + census[id(&w, "burning_wood").index()];
        let burned = 1200 + 8 - left;
        let ash = census[id(&w, "ash").index()];
        assert!(left < 1000, "wood left {left}");
        assert!(smoke_seen > 50, "smoke {smoke_seen}");
        assert!(ash > 0 && ash * 4 < burned, "ash {ash} of {burned} burned");
    }

    #[test]
    fn wind_tilts_flames_but_keeps_some_jitter() {
        for wind in [3, 4] {
            flames_in_wind(wind);
        }
    }

    fn flames_in_wind(wind: i8) {
        let mut w = world(1);
        w.set_wind(wind);
        w.fill_rect(IRect::new(0, 180, 256, 192), id(&w, "stone"));
        w.fill_rect(IRect::new(126, 176, 130, 180), id(&w, "heater"));
        w.fill_rect(IRect::new(126, 172, 130, 176), id(&w, "burning_coal"));
        let fire = id(&w, "fire");
        let (mut upwind, mut downwind) = (0, 0);
        for _ in 0..120 {
            run(&mut w, 1);
            upwind += w.count(IRect::new(100, 140, 126, 172), fire);
            downwind += w.count(IRect::new(130, 140, 156, 172), fire);
        }
        assert!(downwind > upwind, "wind {wind}: down {downwind} up {upwind}");
        assert!(upwind > 0, "wind {wind}: no flame ever moved upwind");
    }
}
