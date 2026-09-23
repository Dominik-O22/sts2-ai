//! The game's own random numbers, value for value, for the run layer: with
//! them a run's seed reproduces its map, rewards and shops, and a save's
//! stream counters pick up where the game left off. Combat keeps
//! `rng::Rng`, which does not try to match (DESIGN.md, Simulator).
//!
//! `Random/MegaRandom.cs` (xoshiro256** seeded through splitmix64),
//! `Random/Rng.cs` (the counted wrapper every system draws from),
//! `Runs/RunRngSet.cs` and `Random/PlayerRngSet.cs` (the named streams),
//! `Helpers/StringHelper.cs` (the seed hash). `tools/oracle` prints what the
//! game's code gives for a seed; the tests pin it.

/// `MegaRandom`.
#[derive(Clone, Debug)]
struct MegaRandom {
    s: [u64; 4],
}

impl MegaRandom {
    fn new(mut seed: u64) -> Self {
        let mut splitmix = || {
            seed = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = seed;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self { s: [splitmix(), splitmix(), splitmix(), splitmix()] }
    }

    fn next_u64(&mut self) -> u64 {
        let [s0, s1, s2, s3] = &mut self.s;
        let result = s1.wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = *s1 << 17;
        *s2 ^= *s0;
        *s3 ^= *s1;
        *s1 ^= *s2;
        *s0 ^= *s3;
        *s2 ^= t;
        *s3 = s3.rotate_left(45);
        result
    }

    fn next_double(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * 1.110_223_024_625_156_5E-16
    }

    /// `Next(maxValue)`: scaled from a double, not a modulo.
    fn next(&mut self, max: i32) -> i32 {
        (self.next_double() * max as f64) as i32
    }
}

/// `Rng`: a stream that counts its draws, so a save can fast-forward a fresh
/// one to where the game was.
#[derive(Clone, Debug)]
pub struct GameRng {
    random: MegaRandom,
    pub seed: u32,
    pub counter: u32,
}

impl GameRng {
    pub fn new(seed: u32) -> Self {
        Self { random: MegaRandom::new(seed as u64), seed, counter: 0 }
    }

    /// `new Rng(seed, name)`: the stream named `name` of a run seeded `seed`.
    pub fn named(seed: u32, name: &str) -> Self {
        Self::new(seed.wrapping_add(hash(name) as u32))
    }

    /// `FastForwardCounter`: skip to draw `target`.
    pub fn fast_forward(&mut self, target: u32) {
        assert!(self.counter <= target, "cannot rewind an Rng ({} to {target})", self.counter);
        while self.counter < target {
            self.counter += 1;
            self.random.next_u64();
        }
    }

    pub fn next_bool(&mut self) -> bool {
        self.counter += 1;
        self.random.next(2) == 0
    }

    /// `NextInt(maxExclusive)`.
    pub fn next_int(&mut self, max: i32) -> i32 {
        assert!(max >= 1, "NextInt({max})");
        self.counter += 1;
        self.random.next(max)
    }

    /// `NextInt(minInclusive, maxExclusive)`.
    pub fn next_int_in(&mut self, min: i32, max: i32) -> i32 {
        assert!(min < max, "NextInt({min}, {max})");
        self.counter += 1;
        let span = max as i64 - min as i64;
        (self.random.next_double() * span as f64) as i64 as i32 + min
    }

    /// `NextUnsignedInt(minInclusive, maxExclusive)`.
    pub fn next_unsigned_in(&mut self, min: u32, max: u32) -> u32 {
        assert!(min < max);
        self.counter += 1;
        min + (self.random.next_double() * (max - min) as f64) as u32
    }

    /// `NextFloat(max)`, which is `NextFloat(0, max)`.
    pub fn next_float(&mut self, max: f32) -> f32 {
        self.next_float_in(0.0, max)
    }

    pub fn next_float_in(&mut self, min: f32, max: f32) -> f32 {
        self.counter += 1;
        (self.random.next_double() * (max - min) as f64 + min as f64) as f32
    }

    pub fn next_double(&mut self) -> f64 {
        self.counter += 1;
        self.random.next_double()
    }

    /// `NextGaussianInt`: Box-Muller, rounded half to even like
    /// `Math.Round`, redrawn until it lands in `min..=max`.
    pub fn next_gaussian_int(&mut self, mean: i32, std_dev: i32, min: i32, max: i32) -> i32 {
        loop {
            let d = 1.0 - self.next_double();
            let u = 1.0 - self.next_double();
            let z = (-2.0 * d.ln()).sqrt() * (std::f64::consts::TAU * u).sin();
            let n = (mean as f64 + std_dev as f64 * z).round_ties_even() as i32;
            if (min..=max).contains(&n) {
                return n;
            }
        }
    }

    /// `NextItem`: a uniform pick, `None` from an empty list (which draws
    /// nothing).
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            return None;
        }
        Some(&items[self.next_int_in(0, items.len() as i32) as usize])
    }

    /// `WeightedNextItem`: one draw, then the first item whose running
    /// weight reaches it; `None` if rounding leaves it past the end.
    pub fn weighted_pick<'a, T>(&mut self, items: &'a [T], weight: impl Fn(&T) -> f32) -> Option<&'a T> {
        let total: f32 = items.iter().map(&weight).sum();
        let mut left = self.next_float(1.0) * total;
        items.iter().find(|item| {
            left -= weight(item);
            left <= 0.0
        })
    }

    /// `Shuffle`: Fisher-Yates from the back.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.next_int(i as i32 + 1) as usize;
            items.swap(i, j);
        }
    }
}

/// `StringHelper.GetDeterministicHashCode`: two interleaved djb2-style
/// hashes over the UTF-16 code units, with C#'s wrapping `int` arithmetic.
pub fn hash(s: &str) -> i32 {
    let (mut a, mut b) = (352_654_597i32, 352_654_597i32);
    let units: Vec<u16> = s.encode_utf16().collect();
    for pair in units.chunks(2) {
        a = (a << 5).wrapping_add(a) ^ pair[0] as i32;
        if let Some(&u) = pair.get(1) {
            b = (b << 5).wrapping_add(b) ^ u as i32;
        }
    }
    a.wrapping_add(b.wrapping_mul(1_566_083_941))
}

/// `StringHelper.SnakeCase` on an identifier: `UpFront` to `up_front`.
pub fn snake_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len() + 4);
    for (i, ch) in name.chars().enumerate() {
        if ch.is_ascii_uppercase() && i > 0 {
            out.push('_');
        }
        out.push(ch.to_ascii_lowercase());
    }
    out
}

/// `RunRngType`, in the game's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunStream {
    UpFront,
    Shuffle,
    UnknownMapPoint,
    CombatCardGeneration,
    CombatPotionGeneration,
    CombatCardSelection,
    CombatEnergyCosts,
    CombatTargets,
    MonsterAi,
    Niche,
    CombatOrbs,
    TreasureRoomRelics,
}

impl RunStream {
    pub const ALL: [RunStream; 12] = [
        Self::UpFront,
        Self::Shuffle,
        Self::UnknownMapPoint,
        Self::CombatCardGeneration,
        Self::CombatPotionGeneration,
        Self::CombatCardSelection,
        Self::CombatEnergyCosts,
        Self::CombatTargets,
        Self::MonsterAi,
        Self::Niche,
        Self::CombatOrbs,
        Self::TreasureRoomRelics,
    ];
}

/// `PlayerRngType`, in the game's order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlayerStream {
    Rewards,
    Shops,
    Transformations,
}

impl PlayerStream {
    pub const ALL: [PlayerStream; 3] = [Self::Rewards, Self::Shops, Self::Transformations];
}

/// `RunRngSet` with the single player's `PlayerRngSet`: every stream a run
/// draws from, seeded from the run's seed string (`Player.InitializeSeed`
/// adds the player's slot, 0 alone).
#[derive(Clone, Debug)]
pub struct RunRngs {
    pub seed: u32,
    run: Vec<GameRng>,
    player: Vec<GameRng>,
}

impl RunRngs {
    pub fn new(seed: &str) -> Self {
        let seed = hash(seed) as u32;
        let stream = |name: String| GameRng::named(seed, &name);
        Self {
            seed,
            run: RunStream::ALL.iter().map(|s| stream(snake_case(&format!("{s:?}")))).collect(),
            player: PlayerStream::ALL.iter().map(|s| stream(snake_case(&format!("{s:?}")))).collect(),
        }
    }

    pub fn run(&mut self, stream: RunStream) -> &mut GameRng {
        &mut self.run[stream as usize]
    }

    pub fn player(&mut self, stream: PlayerStream) -> &mut GameRng {
        &mut self.player[stream as usize]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// What `tools/oracle rng 207KM9GFWS` printed from the game's code.
    #[test]
    fn matches_the_game() {
        let mut rngs = RunRngs::new("207KM9GFWS");
        assert_eq!(rngs.seed, 3_320_802_071);
        assert_eq!(snake_case("CombatCardGeneration"), "combat_card_generation");
        let first_five = |r: &mut GameRng| (0..5).map(|_| r.next_int(100)).collect::<Vec<_>>();
        let want: [[i32; 5]; 12] = [
            [46, 99, 3, 24, 66],
            [61, 98, 58, 42, 63],
            [83, 58, 39, 13, 2],
            [4, 4, 7, 31, 73],
            [42, 83, 21, 19, 40],
            [71, 59, 25, 25, 23],
            [78, 18, 68, 25, 24],
            [6, 41, 53, 65, 13],
            [43, 70, 18, 62, 49],
            [55, 79, 89, 76, 7],
            [71, 27, 65, 46, 14],
            [62, 51, 21, 10, 48],
        ];
        for (stream, want) in RunStream::ALL.iter().zip(want) {
            assert_eq!(first_five(rngs.run(*stream)), want, "{stream:?}");
        }
        assert_eq!(first_five(rngs.player(PlayerStream::Rewards)), [66, 23, 78, 13, 18]);

        let mut r = GameRng::named(rngs.seed, "up_front");
        assert_eq!((0..5).map(|_| r.next_int_in(-3, 7)).collect::<Vec<_>>(), [1, 6, -3, -1, 3]);
        assert_eq!((0..5).map(|_| r.next_bool() as i32).collect::<Vec<_>>(), [1, 1, 0, 1, 0]);
        assert_eq!((0..3).map(|_| r.next_float(1.0)).collect::<Vec<_>>(), [0.6641236, 0.41022843, 0.8160737]);
        assert_eq!((0..3).map(|_| r.next_double()).collect::<Vec<_>>(), [0.16261445306831468, 0.017817093146488494, 0.2086316264418071]);
        assert_eq!((0..3).map(|_| r.next_unsigned_in(5, 50)).collect::<Vec<_>>(), [26, 26, 13]);
        assert_eq!((0..3).map(|_| r.next_gaussian_int(10, 3, 5, 15)).collect::<Vec<_>>(), [10, 9, 14]);
        let mut list: Vec<i32> = (0..10).collect();
        r.shuffle(&mut list);
        assert_eq!(list, [7, 5, 0, 3, 2, 1, 9, 4, 8, 6]);
        assert_eq!(r.pick(&[10, 20, 30, 40]), Some(&20));
        assert_eq!(r.counter, 35);

        let mut ff = GameRng::named(rngs.seed, "up_front");
        ff.fast_forward(7);
        assert_eq!(ff.next_int(100), 86);
    }
}
