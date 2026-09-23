//! An act's map, generated from the run's seed the way the game does it, point
//! for point: `Map/StandardActMap.cs` with `Map/MapPathPruning.cs`,
//! `Map/MapPostProcessing.cs`, `Map/MapPoint.cs`, `Map/MapPointTypeCounts.cs`,
//! `Map/ActMap.cs` and each act's `GetMapPointTypes` (`Models/Acts/*.cs`).
//!
//! `MapPoint.Children`, `MapPoint.parents` and `ActMap.startMapPoints` are
//! .NET `HashSet`s, and pruning walks `Children` to list paths, so which
//! duplicate path gets pruned depends on the order a `HashSet` enumerates.
//! `SlotSet` reproduces it. `tools/oracle maps` prints the game's maps for
//! any seed; `examples/mapcheck.rs` diffs them against this port.
//!
//! Not here: multiplayer (one room fewer, `ActModel.GetNumberOfRooms`), and
//! the relic hooks that replace or edit a map after it is generated
//! (`Hook.ModifyGeneratedMap`: `SpoilsMap`'s `SpoilsActMap`, and
//! `GoldenPathActMap`), which `RunManager` applies to the result of
//! `ActMap::generate`. Nor the act 1 start without Neow, which `RunManager`
//! turns from `Ancient` into `Monster` after generation.

use std::collections::{BTreeMap, VecDeque};
use std::fmt::Write;
use std::ops::{Index, IndexMut, Range};

use crate::encounter::Act;
use crate::game_rng::{hash, GameRng};
use crate::types::{Ascension, AscensionLevel};

/// `StandardActMap._mapWidth`.
const COLS: usize = 7;

/// `Map/MapPointType.cs`, in the game's order (segment keys print the
/// ordinal).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PointType {
    Unassigned,
    Unknown,
    Shop,
    Treasure,
    RestSite,
    Monster,
    Elite,
    Boss,
    Ancient,
}

/// A point's slot in `ActMap::points`. Pruned points keep theirs; they only
/// leave the grid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PointId(u32);

impl PointId {
    /// Its slot in the map's points, for tables over them.
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// A .NET `HashSet<MapPoint>` as far as its enumeration order goes. `MapPoint`
/// hashes by identity, and a `HashSet` enumerates its entry array in slot
/// order whatever the hashes: an add takes the most recently freed slot, or
/// the next new one, and a remove frees its slot.
#[derive(Clone, Debug, Default)]
pub struct SlotSet {
    slots: Vec<Option<PointId>>,
    free: Vec<usize>,
}

impl SlotSet {
    fn add(&mut self, id: PointId) {
        if self.contains(id) {
            return;
        }
        match self.free.pop() {
            Some(slot) => self.slots[slot] = Some(id),
            None => self.slots.push(Some(id)),
        }
    }

    fn remove(&mut self, id: PointId) {
        if let Some(slot) = self.slots.iter().position(|&s| s == Some(id)) {
            self.slots[slot] = None;
            self.free.push(slot);
        }
    }

    pub fn contains(&self, id: PointId) -> bool {
        self.slots.contains(&Some(id))
    }

    pub fn len(&self) -> usize {
        self.iter().count()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl Iterator<Item = PointId> + '_ {
        self.slots.iter().flatten().copied()
    }
}

/// `Map/MapPoint.cs`.
#[derive(Clone, Debug)]
pub struct MapPoint {
    pub col: usize,
    pub row: usize,
    pub kind: PointType,
    /// `CanBeModified`: false for the rows `AssignPointTypes` fixes.
    pub can_be_modified: bool,
    pub children: SlotSet,
    pub parents: SlotSet,
}

impl MapPoint {
    fn new(col: usize, row: usize) -> Self {
        Self {
            col,
            row,
            kind: PointType::Unassigned,
            can_be_modified: true,
            children: SlotSet::default(),
            parents: SlotSet::default(),
        }
    }
}

/// `StandardActMap` once built. The grid holds rows 1 to `rows - 1`; the
/// starting point sits below it in row 0 and the boss above it.
#[derive(Clone, Debug)]
pub struct ActMap {
    points: Vec<MapPoint>,
    /// `Grid[col, row]`, row-major here.
    grid: Vec<Option<PointId>>,
    rows: usize,
    pub start: PointId,
    pub boss: PointId,
    /// The Double Boss ascension's second boss, above the first.
    pub second_boss: Option<PointId>,
}

impl Index<PointId> for ActMap {
    type Output = MapPoint;
    fn index(&self, id: PointId) -> &MapPoint {
        &self.points[id.0 as usize]
    }
}

impl IndexMut<PointId> for ActMap {
    fn index_mut(&mut self, id: PointId) -> &mut MapPoint {
        &mut self.points[id.0 as usize]
    }
}

/// `MapPointTypeCounts`. Shops are always 3, and no act fills
/// `PointTypesThatIgnoreRules`, so it is left out.
#[derive(Clone, Copy, Debug)]
struct Counts {
    elites: usize,
    unknowns: usize,
    rests: usize,
}

const SHOPS: usize = 3;

/// `ActModel.GetNumberOfRooms` for one player: each act's
/// `BaseNumberOfRooms`, the boss and the start excluded.
pub(crate) fn rooms(act: Act) -> usize {
    match act {
        Act::Overgrowth | Act::Underdocks => 15,
        Act::Hive => 14,
        Act::Glory => 13,
    }
}

/// Each act's `GetMapPointTypes`, drawn on the map stream before the map.
/// `MapPointTypeCounts.NumOfElites` reads Swarming Elites as it is built.
fn counts(act: Act, rng: &mut GameRng, ascension: Ascension) -> Counts {
    // `MapPointTypeCounts.StandardRandomUnknownCount`.
    let standard_unknowns = |rng: &mut GameRng| rng.next_gaussian_int(12, 1, 10, 14);
    let (rests, unknowns) = match act {
        Act::Overgrowth | Act::Underdocks => {
            let rests = rng.next_gaussian_int(7, 1, 6, 7);
            (rests, standard_unknowns(rng))
        }
        Act::Hive => {
            let rests = rng.next_gaussian_int(6, 1, 6, 7);
            (rests, standard_unknowns(rng) - 1)
        }
        Act::Glory => {
            let rests = rng.next_int_in(5, 7);
            (rests, standard_unknowns(rng) - 1)
        }
    };
    let swarming = if ascension.has(AscensionLevel::SwarmingElites) { 1.6f32 } else { 1.0 };
    Counts { elites: (5.0 * swarming).round() as usize, unknowns: unknowns as usize, rests: rests as usize }
}

impl ActMap {
    /// `StandardActMap.CreateFor` for a single player: the map of `act`, drawn
    /// on the run's `act_{n}_map` stream, `n` being the act's place in the
    /// run. `seed` is the run's (`RunRngs::seed`). The last act gets a second
    /// boss at Double Boss (`RunManager` gives it `SetSecondBossEncounter`).
    pub fn generate(seed: u32, act: Act, ascension: Ascension) -> Self {
        let mut rng = GameRng::named(seed, &format!("act_{}_map", act.index() + 1));
        let counts = counts(act, &mut rng, ascension);
        let second_boss = act.index() == 2 && ascension.has(AscensionLevel::DoubleBoss);
        Generator::new(rng, counts, rooms(act) + 1, second_boss).run()
    }

    /// The points on the grid, column by column (`ActMap.GetAllMapPoints`).
    pub fn grid_points(&self) -> impl Iterator<Item = PointId> + '_ {
        (0..COLS).flat_map(move |col| (0..self.rows).filter_map(move |row| self.cell(col, row)))
    }

    /// Every point, the start and the bosses included.
    pub fn all_points(&self) -> impl Iterator<Item = PointId> + '_ {
        self.grid_points().chain([self.start, self.boss]).chain(self.second_boss)
    }

    fn cell(&self, col: usize, row: usize) -> Option<PointId> {
        self.grid[row * COLS + col]
    }

    fn set_cell(&mut self, col: usize, row: usize, id: Option<PointId>) {
        self.grid[row * COLS + col] = id;
    }

    /// Every point as `tools/oracle maps` prints it: by row, then column,
    /// `col row Type child_col,child_row ...` with children by column.
    pub fn oracle_text(&self) -> String {
        let mut ids: Vec<PointId> = self.all_points().collect();
        ids.sort_by_key(|&p| (self[p].row, self[p].col));
        let mut out = String::new();
        for p in ids {
            let point = &self[p];
            let mut kids: Vec<PointId> = point.children.iter().collect();
            kids.sort_by_key(|&c| self[c].col);
            let kids: Vec<String> = kids.iter().map(|&c| format!("{},{}", self[c].col, self[c].row)).collect();
            writeln!(out, "{} {} {:?} {}", point.col, point.row, point.kind, kids.join(" ")).unwrap();
        }
        out
    }
}

/// The state `StandardActMap`'s constructor works on.
struct Generator {
    map: ActMap,
    rng: GameRng,
    counts: Counts,
    /// `ActMap.startMapPoints`: row 1's path starts.
    start_points: SlotSet,
}

impl Generator {
    fn new(rng: GameRng, counts: Counts, rows: usize, second_boss: bool) -> Self {
        let mut points = vec![MapPoint::new(COLS / 2, rows), MapPoint::new(COLS / 2, 0)];
        if second_boss {
            points.push(MapPoint::new(COLS / 2, rows + 1));
        }
        let map = ActMap {
            points,
            grid: vec![None; COLS * rows],
            rows,
            boss: PointId(0),
            start: PointId(1),
            second_boss: second_boss.then_some(PointId(2)),
        };
        Self { map, rng, counts, start_points: SlotSet::default() }
    }

    /// The constructor's steps, in its order.
    fn run(mut self) -> ActMap {
        self.generate_map();
        self.assign_point_types();
        self.prune_and_repair();
        self.center_grid();
        self.spread_adjacent_map_points();
        self.straighten_paths();
        self.map
    }

    fn rows(&self) -> usize {
        self.map.rows
    }

    fn add_child(&mut self, parent: PointId, child: PointId) {
        self.map[parent].children.add(child);
        self.map[child].parents.add(parent);
    }

    fn remove_child(&mut self, parent: PointId, child: PointId) {
        self.map[parent].children.remove(child);
        self.map[child].parents.remove(parent);
    }

    fn row_points(&self, row: usize) -> Vec<PointId> {
        (0..COLS).filter_map(|col| self.map.cell(col, row)).collect()
    }

    /// `ListExtensions.StableShuffle`: sorted by `MapPoint.CompareTo` (column,
    /// then row), then `UnstableShuffle`.
    fn stable_shuffle(&mut self, list: &mut [PointId]) {
        let map = &self.map;
        list.sort_by_key(|&p| (map[p].col, map[p].row));
        self.rng.shuffle(list);
    }

    // ---- StandardActMap: paths ----

    fn get_or_create_point(&mut self, col: usize, row: usize) -> PointId {
        if let Some(id) = self.map.cell(col, row) {
            return id;
        }
        let id = PointId(self.map.points.len() as u32);
        self.map.points.push(MapPoint::new(col, row));
        self.map.set_cell(col, row, Some(id));
        id
    }

    fn generate_map(&mut self) {
        for i in 0..7 {
            let mut point = self.random_row_one_point();
            if i == 1 {
                while self.start_points.contains(point) {
                    point = self.random_row_one_point();
                }
            }
            self.start_points.add(point);
            self.path_generate(point);
        }
        for p in self.row_points(self.rows() - 1) {
            self.add_child(p, self.map.boss);
        }
        if let Some(second) = self.map.second_boss {
            self.add_child(self.map.boss, second);
        }
        for p in self.row_points(1) {
            self.add_child(self.map.start, p);
        }
    }

    fn random_row_one_point(&mut self) -> PointId {
        let col = self.rng.next_int_in(0, COLS as i32) as usize;
        self.get_or_create_point(col, 1)
    }

    fn path_generate(&mut self, start: PointId) {
        let mut point = start;
        while self.map[point].row < self.rows() - 1 {
            let (col, row) = self.generate_next_coord(point);
            let next = self.get_or_create_point(col, row);
            self.add_child(point, next);
            point = next;
        }
    }

    fn generate_next_coord(&mut self, current: PointId) -> (usize, usize) {
        let MapPoint { col, row, .. } = self.map[current];
        let left = col.saturating_sub(1);
        let right = (col + 1).min(COLS - 1);
        // `StableShuffle` of an already sorted list.
        let mut steps = [-1, 0, 1];
        self.rng.shuffle(&mut steps);
        for step in steps {
            let target = match step {
                -1 => left,
                0 => col,
                _ => right,
            };
            if !self.has_invalid_crossover(current, target) {
                return (target, row + 1);
            }
        }
        panic!("Cannot find next node: seed={}", self.rng.seed);
    }

    /// Whether stepping from `current` to column `target` would cross the
    /// edge out of its neighbour at `target`.
    fn has_invalid_crossover(&self, current: PointId, target: usize) -> bool {
        let MapPoint { col, row, .. } = self.map[current];
        let step = target as i32 - col as i32;
        if step == 0 || step == 7 {
            return false;
        }
        let Some(neighbour) = self.map.cell(target, row) else {
            return false;
        };
        let map = &self.map;
        map[neighbour].children.iter().any(|c| map[c].col as i32 - target as i32 == -step)
    }

    // ---- StandardActMap: point types ----

    fn assign_point_types(&mut self) {
        let rows = self.rows();
        for p in self.row_points(rows - 1) {
            self.map[p].kind = PointType::RestSite;
            self.map[p].can_be_modified = false;
        }
        // `ShouldReplaceTreasureWithElites` would make this row elites; no
        // caller in the game sets it.
        for p in self.row_points(rows - 7) {
            self.map[p].kind = PointType::Treasure;
            self.map[p].can_be_modified = false;
        }
        for p in self.row_points(1) {
            self.map[p].kind = PointType::Monster;
            self.map[p].can_be_modified = false;
        }
        let Counts { elites, unknowns, rests } = self.counts;
        let mut queue: VecDeque<PointType> = [
            (PointType::RestSite, rests),
            (PointType::Shop, SHOPS),
            (PointType::Elite, elites),
            (PointType::Unknown, unknowns),
        ]
        .into_iter()
        .flat_map(|(kind, n)| std::iter::repeat_n(kind, n))
        .collect();
        self.assign_remaining_types_to_random_points(&mut queue);
        let unassigned: Vec<PointId> =
            self.map.grid_points().filter(|&p| self.map[p].kind == PointType::Unassigned).collect();
        for p in unassigned {
            self.map[p].kind = PointType::Monster;
        }
        let (boss, start) = (self.map.boss, self.map.start);
        self.map[boss].kind = PointType::Boss;
        self.map[start].kind = PointType::Ancient;
        if let Some(second) = self.map.second_boss {
            self.map[second].kind = PointType::Boss;
        }
    }

    fn assign_remaining_types_to_random_points(&mut self, queue: &mut VecDeque<PointType>) {
        for _ in 0..3 {
            if queue.is_empty() {
                break;
            }
            let mut open: Vec<PointId> =
                self.map.grid_points().filter(|&p| self.map[p].kind == PointType::Unassigned).collect();
            self.stable_shuffle(&mut open);
            for p in open {
                if queue.is_empty() {
                    break;
                }
                self.map[p].kind = self.next_valid_point_type(queue, p);
            }
        }
    }

    /// The first queued type `point` can take, taken out of the queue; the
    /// ones tried before it go to the back. `Unassigned` if none fits.
    fn next_valid_point_type(&self, queue: &mut VecDeque<PointType>, point: PointId) -> PointType {
        for _ in 0..queue.len() {
            let kind = queue.pop_front().unwrap();
            if self.is_valid_point_type(kind, point) {
                return kind;
            }
            queue.push_back(kind);
        }
        PointType::Unassigned
    }

    fn is_valid_point_type(&self, kind: PointType, point: PointId) -> bool {
        use PointType::*;
        let map = &self.map;
        let p = &map[point];
        let is_kind = |q: PointId| map[q].kind == kind;
        // Top rows: no rest site in the two rows below the rest row.
        if p.row >= self.rows() - 3 && kind == RestSite {
            return false;
        }
        // Lower rows: no rest site or elite.
        if p.row < 6 && matches!(kind, RestSite | Elite) {
            return false;
        }
        // Not twice in a row, above or below (`IsValidWithParents` and
        // `IsValidWithChildren`).
        if matches!(kind, Elite | RestSite | Treasure | Shop)
            && (p.parents.iter().any(is_kind) || p.children.iter().any(is_kind))
        {
            return false;
        }
        // Not twice among siblings, the other children of its parents.
        if matches!(kind, RestSite | Monster | Unknown | Elite | Shop)
            && p.parents.iter().flat_map(|q| map[q].children.iter()).any(|q| q != point && is_kind(q))
        {
            return false;
        }
        true
    }

    // ---- MapPathPruning ----

    /// `PruneAndRepair`: pruning can drop a type below its count, and the
    /// repair can make new duplicates, so up to three rounds.
    fn prune_and_repair(&mut self) {
        for _ in 0..3 {
            self.prune_duplicate_segments();
            if !self.repair_pruned_point_types() {
                break;
            }
        }
    }

    fn repair_pruned_point_types(&mut self) -> bool {
        let Counts { elites, unknowns, rests } = self.counts;
        let shop = self.repair_point_type(PointType::Shop, SHOPS);
        let elite = self.repair_point_type(PointType::Elite, elites);
        let rest = self.repair_point_type(PointType::RestSite, rests);
        let unknown = self.repair_point_type(PointType::Unknown, unknowns);
        shop | elite | rest | unknown
    }

    /// Turns modifiable monsters into `kind` until the map has `target`.
    fn repair_point_type(&mut self, kind: PointType, target: usize) -> bool {
        let have = self.map.grid_points().filter(|&p| self.map[p].kind == kind).count();
        let Some(mut missing) = target.checked_sub(have).filter(|&n| n > 0) else {
            return false;
        };
        let mut monsters: Vec<PointId> = self
            .map
            .grid_points()
            .filter(|&p| self.map[p].kind == PointType::Monster && self.map[p].can_be_modified)
            .collect();
        self.stable_shuffle(&mut monsters);
        let mut repaired = false;
        for p in monsters {
            if missing == 0 {
                break;
            }
            if self.is_valid_point_type(kind, p) {
                self.map[p].kind = kind;
                missing -= 1;
                repaired = true;
            }
        }
        repaired
    }

    fn prune_duplicate_segments(&mut self) {
        let mut iterations = 0;
        let mut matching = self.find_matching_segments();
        while self.prune_paths(&mut matching) {
            iterations += 1;
            assert!(iterations <= 50, "Unable to prune matching segments in {iterations} iterations");
            matching = self.find_matching_segments();
        }
    }

    /// `FindMatchingSegments`: groups of path segments that run between the
    /// same two points (or from the start to the same point) through the
    /// same room types, the groups with more than one in key order
    /// (`SortedDictionary` with `StringComparer.Ordinal`), each group
    /// non-overlapping and in the order the game's paths find them.
    ///
    /// The game lists every start-to-boss path (`FindAllPaths`) and files
    /// every fork-to-merge segment of each (`AddSegmentsToDictionary`), so a
    /// segment is filed once per path through it. This walks each distinct
    /// segment once instead, which gives the same groups:
    /// - A segment is on some listed path exactly when the start reaches its
    ///   first point and its last point reaches the boss.
    /// - A segment filed again overlaps itself, so only its first filing can
    ///   join its group, and one turned away stays out as groups only grow.
    /// - Segments with one key share both ends, so the first listed paths
    ///   through them differ only between the ends, and are listed in the
    ///   order of the children taken there: depth first from the first
    ///   point, children in `HashSet` order, as `walk_segments` goes.
    fn find_matching_segments(&self) -> Vec<Vec<Vec<PointId>>> {
        let mut reaches_boss = vec![None; self.map.points.len()];
        self.reaches_boss(self.map.start, &mut reaches_boss);
        let on_path: Vec<bool> = reaches_boss.iter().map(|r| *r == Some(true)).collect();

        let mut walk = SegmentWalk { on_path, path: Vec::new(), found: Vec::new(), points: Vec::new() };
        for first in self.map.all_points() {
            let p = &self.map[first];
            // `IsValidSegmentStartMapPoint`.
            if walk.on_path[first.0 as usize] && (p.children.len() > 1 || p.row == 0) {
                let from = walk.found.len();
                walk.path.push(first);
                self.walk_segments(&mut walk, 0);
                walk.path.pop();
                // Brings each key's segments together, still in walk order.
                walk.found[from..].sort_by_key(|s| (s.last.0, s.inner_kinds));
            }
        }

        let SegmentWalk { found, points, .. } = walk;
        let mut groups: BTreeMap<String, Vec<Vec<PointId>>> = BTreeMap::new();
        for same_key in found.chunk_by(|a, b| (a.first, a.last, a.inner_kinds) == (b.first, b.last, b.inner_kinds)) {
            if same_key.len() < 2 {
                continue;
            }
            let mut group: Vec<&[PointId]> = Vec::new();
            for found in same_key {
                let segment = &points[found.points.clone()];
                if !group.iter().any(|other| overlapping(other, segment)) {
                    group.push(segment);
                }
            }
            if group.len() > 1 {
                groups.insert(self.segment_key(group[0]), group.iter().map(|s| s.to_vec()).collect());
            }
        }
        groups.into_values().collect()
    }

    /// Whether `point` reaches the boss, filling `memo` for every point the
    /// start reaches (`None` for the rest), so it asks every child. Like
    /// `FindAllPaths` it stops at the boss, whatever follows it.
    fn reaches_boss(&self, point: PointId, memo: &mut [Option<bool>]) -> bool {
        if let Some(reaches) = memo[point.0 as usize] {
            return reaches;
        }
        let p = &self.map[point];
        let reaches = p.kind == PointType::Boss
            || p.children.iter().fold(false, |any, child| self.reaches_boss(child, memo) | any);
        memo[point.0 as usize] = Some(reaches);
        reaches
    }

    /// Walks every path on from `walk.path`, depth first with children in
    /// `HashSet` order, and records each segment that ends on a merge
    /// (`IsValidSegmentEndMapPoint`) at least two steps from its first point.
    /// `inner_kinds` holds the types strictly between the first point and
    /// the last, four bits each; a segment has at most 15 of them.
    fn walk_segments(&self, walk: &mut SegmentWalk, inner_kinds: u64) {
        let (first, last) = (walk.path[0], *walk.path.last().unwrap());
        let p = &self.map[last];
        if walk.path.len() >= 3 && p.parents.len() >= 2 {
            let start = walk.points.len();
            walk.points.extend_from_slice(&walk.path);
            walk.found.push(FoundSegment { first, last, inner_kinds, points: start..walk.points.len() });
        }
        if p.kind == PointType::Boss {
            return;
        }
        // Sixteen 4-bit room types fill the key; a path climbs a row a step,
        // so this only fails for an act taller than any the game has.
        assert!(walk.path.len() <= 16, "segment too long for its key");
        let inner_kinds = if walk.path.len() == 1 { 0 } else { inner_kinds << 4 | p.kind as u64 };
        for child in p.children.iter() {
            if walk.on_path[child.0 as usize] {
                walk.path.push(child);
                self.walk_segments(walk, inner_kinds);
                walk.path.pop();
            }
        }
    }

    /// `GenerateSegmentKey`: the ends' coordinates (just the row for the
    /// start), then every point's type ordinal.
    fn segment_key(&self, segment: &[PointId]) -> String {
        let (first, last) = (&self.map[segment[0]], &self.map[*segment.last().unwrap()]);
        let mut key = if first.row == 0 {
            format!("{}-{},{}-", first.row, last.col, last.row)
        } else {
            format!("{},{}-{},{}-", first.col, first.row, last.col, last.row)
        };
        let kinds: Vec<String> = segment.iter().map(|&p| (self.map[p].kind as u8).to_string()).collect();
        key.push_str(&kinds.join(","));
        key
    }

    /// `PrunePaths`: shuffles each group in turn and prunes all but one of
    /// it, or failing that cuts an edge; true once something changed.
    fn prune_paths(&mut self, matching: &mut [Vec<Vec<PointId>>]) -> bool {
        for group in matching {
            self.rng.shuffle(group);
            if self.prune_all_but_last(group) != 0 {
                return true;
            }
            if group.iter().any(|segment| self.break_parent_child_relationship(segment)) {
                return true;
            }
        }
        false
    }

    fn prune_all_but_last(&mut self, group: &[Vec<PointId>]) -> usize {
        let mut pruned = 0;
        for segment in group {
            if pruned == group.len() - 1 {
                return pruned;
            }
            if self.prune_segment(segment) {
                pruned += 1;
            }
        }
        pruned
    }

    fn prune_segment(&mut self, segment: &[PointId]) -> bool {
        let mut removed = false;
        let last = *segment.last().unwrap();
        for i in 0..segment.len() - 1 {
            let point = segment[i];
            if !self.is_in_map(point) {
                return true;
            }
            let map = &self.map;
            let p = &map[point];
            if p.children.len() > 1
                || p.parents.len() > 1
                || p.parents.iter().any(|n| map[n].children.len() == 1 && !self.is_removed(n))
            {
                continue;
            }
            if segment[i..].iter().any(|&n| map[n].children.len() > 1 && map[n].parents.len() == 1) {
                continue;
            }
            if map[last].parents.len() == 1 {
                return false;
            }
            let strands_a_child =
                p.children.iter().filter(|c| !segment.contains(c)).any(|c| map[c].parents.len() == 1);
            if !strands_a_child {
                self.remove_point(point);
                removed = true;
            }
        }
        removed
    }

    fn remove_point(&mut self, point: PointId) {
        let MapPoint { col, row, .. } = self.map[point];
        self.map.set_cell(col, row, None);
        self.start_points.remove(point);
        for child in self.map[point].children.iter().collect::<Vec<_>>() {
            self.remove_child(point, child);
        }
        for parent in self.map[point].parents.iter().collect::<Vec<_>>() {
            self.remove_child(parent, point);
        }
    }

    /// `MapPathPruning.IsInMap`: on the grid, or the start or a boss.
    fn is_in_map(&self, point: PointId) -> bool {
        let p = &self.map[point];
        self.map.cell(p.col, p.row).is_some() || matches!(p.kind, PointType::Ancient | PointType::Boss)
    }

    /// `MapPathPruning.IsRemoved`: off the grid, which the start always is.
    fn is_removed(&self, point: PointId) -> bool {
        let p = &self.map[point];
        self.map.cell(p.col, p.row).is_none()
    }

    /// `BreakAParentChildRelationshipInSegment`: cuts each edge of the
    /// segment that leaves a fork into a merge. Cutting an edge an earlier
    /// cut already removed still counts.
    fn break_parent_child_relationship(&mut self, segment: &[PointId]) -> bool {
        let mut broke = false;
        for pair in segment.windows(2) {
            let (point, next) = (pair[0], pair[1]);
            if self.map[point].children.len() >= 2 && self.map[next].parents.len() != 1 {
                self.remove_child(point, next);
                broke = true;
            }
        }
        broke
    }

    // ---- MapPostProcessing ----

    /// `CenterGrid`: shifts every point a column toward the middle when both
    /// columns on one side are empty and the other side's are not. The side
    /// it shifts toward is empty, so no point falls off.
    fn center_grid(&mut self) {
        let empty = |col: usize| (0..self.rows()).all(|row| self.map.cell(col, row).is_none());
        let shift: i32 = match (empty(0) && empty(1), empty(COLS - 1) && empty(COLS - 2)) {
            (true, false) => -1,
            (false, true) => 1,
            _ => return,
        };
        let points: Vec<PointId> = self.map.grid_points().collect();
        self.map.grid.fill(None);
        for p in points {
            let col = (self.map[p].col as i32 + shift) as usize;
            self.map[p].col = col;
            let row = self.map[p].row;
            self.map.set_cell(col, row, Some(p));
        }
    }

    /// `SpreadAdjacentMapPoints`: row by row, moves each point to the column
    /// its edges allow that is farthest from its row's other points, until
    /// nothing moves.
    fn spread_adjacent_map_points(&mut self) {
        for row in 0..self.rows() {
            let points = self.row_points(row);
            loop {
                let mut moved = false;
                for &p in &points {
                    let col = self.map[p].col;
                    let mut best = (col, self.gap(col, &points, p));
                    for candidate in self.allowed_columns(p) {
                        if candidate != col && self.map.cell(candidate, row).is_none() {
                            let gap = self.gap(candidate, &points, p);
                            if gap > best.1 {
                                best = (candidate, gap);
                            }
                        }
                    }
                    if best.0 != col {
                        self.map.set_cell(col, row, None);
                        self.map.set_cell(best.0, row, Some(p));
                        self.map[p].col = best.0;
                        moved = true;
                    }
                }
                if !moved {
                    break;
                }
            }
        }
    }

    /// `GetAllowedPositions`: the columns within one of every parent and
    /// child. The game builds a `HashSet<int>` of 0 to 6 and only removes
    /// from it, so it enumerates in ascending order.
    fn allowed_columns(&self, point: PointId) -> Vec<usize> {
        let p = &self.map[point];
        let near: Vec<usize> = p.parents.iter().chain(p.children.iter()).map(|q| self.map[q].col).collect();
        (0..COLS).filter(|&c| near.iter().all(|&n| c.abs_diff(n) <= 1)).collect()
    }

    /// `ComputeGap`: the distance from `col` to the nearest other point in
    /// the row.
    fn gap(&self, col: usize, row: &[PointId], point: PointId) -> usize {
        row.iter().filter(|&&q| q != point).map(|&q| col.abs_diff(self.map[q].col)).min().unwrap_or(usize::MAX)
    }

    /// `StraightenPaths`: a point with one parent and one child that both
    /// sit to one side moves a column toward them, if that cell is free.
    /// Reads the grid as it changes, like the game.
    fn straighten_paths(&mut self) {
        for row in 0..self.rows() {
            for col in 0..COLS {
                let Some(p) = self.map.cell(col, row) else { continue };
                let (parents, children) = (&self.map[p].parents, &self.map[p].children);
                if parents.len() != 1 || children.len() != 1 {
                    continue;
                }
                let parent_col = self.map[parents.iter().next().unwrap()].col;
                let child_col = self.map[children.iter().next().unwrap()].col;
                let target = if col < child_col && col < parent_col && col < COLS - 1 {
                    col + 1
                } else if col > child_col && col > parent_col && col > 0 {
                    col - 1
                } else {
                    continue;
                };
                if self.map.cell(target, row).is_none() {
                    self.map[p].col = target;
                    self.map.set_cell(col, row, None);
                    self.map.set_cell(target, row, Some(p));
                }
            }
        }
    }
}

/// `find_matching_segments`' state while it walks the map from each fork.
struct SegmentWalk {
    /// By `PointId`: whether some start-to-boss path goes through it.
    on_path: Vec<bool>,
    /// The points from the segment's first to where the walk is.
    path: Vec<PointId>,
    found: Vec<FoundSegment>,
    /// Every found segment's points, end to end.
    points: Vec<PointId>,
}

/// A fork-to-merge segment: its ends and inner types, which make its
/// `GenerateSegmentKey`, and where its points sit in `SegmentWalk::points`.
struct FoundSegment {
    first: PointId,
    last: PointId,
    inner_kinds: u64,
    points: Range<usize>,
}

/// `MapPathPruning.OverlappingSegment`: two segments with the same key share
/// an inner point at the same step.
fn overlapping(a: &[PointId], b: &[PointId]) -> bool {
    a.len() >= 3 && b.len() >= 3 && (1..=a.len() - 2).any(|i| a[i] == b[i])
}

/// The act named as the game names it.
pub fn act_named(name: &str) -> Option<Act> {
    match name {
        "Overgrowth" => Some(Act::Overgrowth),
        "Underdocks" => Some(Act::Underdocks),
        "Hive" => Some(Act::Hive),
        "Glory" => Some(Act::Glory),
        _ => None,
    }
}

/// Checks `tools/oracle maps` output against `ActMap::generate`: returns how
/// many maps it held and the header of each one that differs, with the first
/// line that does.
pub fn diff_oracle(text: &str) -> (usize, Vec<String>) {
    let mut blocks: Vec<(&str, String)> = Vec::new();
    for line in text.lines() {
        match (line.strip_prefix("map "), blocks.last_mut()) {
            (Some(header), _) => blocks.push((header, String::new())),
            (None, Some((_, points))) => {
                points.push_str(line);
                points.push('\n');
            }
            (None, None) => panic!("points before the first header: {line:?}"),
        }
    }
    let mut mismatches = Vec::new();
    for (header, want) in &blocks {
        let [seed, act, ascension] = header.split(' ').collect::<Vec<_>>()[..] else {
            panic!("bad header {header:?}");
        };
        let act = act_named(act).unwrap_or_else(|| panic!("unknown act {act}"));
        let got = ActMap::generate(hash(seed) as u32, act, Ascension(ascension.parse().unwrap())).oracle_text();
        if got != *want {
            let line = got
                .lines()
                .zip(want.lines())
                .find(|(g, w)| g != w)
                .map_or("different length".to_string(), |(g, w)| format!("got {g:?}, game {w:?}"));
            mismatches.push(format!("{header}: {line}"));
        }
    }
    (blocks.len(), mismatches)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `tools/oracle maps` for 25 seeds of every act at A0 and A10, after
    /// TBL5VNYN4M's Hive at A10, which the game saved from a real run.
    #[test]
    fn matches_the_game() {
        let (maps, mismatches) = diff_oracle(include_str!("../testdata/oracle-maps.txt"));
        assert!(maps >= 100, "fixture holds {maps} maps");
        assert!(mismatches.is_empty(), "{} of {maps} differ:\n{}", mismatches.len(), mismatches.join("\n"));
    }

    /// The generator never adds to a set after removing from it, so the maps
    /// do not reach this; the order is what a .NET `HashSet<MapPoint>` gives.
    #[test]
    fn slot_set_reuses_the_last_freed_slot() {
        let mut set = SlotSet::default();
        for i in 0..4 {
            set.add(PointId(i));
        }
        set.remove(PointId(1));
        set.remove(PointId(2));
        set.add(PointId(7));
        set.add(PointId(8));
        set.add(PointId(0));
        assert_eq!(set.iter().collect::<Vec<_>>(), [PointId(0), PointId(8), PointId(7), PointId(3)]);
    }
}
