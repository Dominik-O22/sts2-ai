//! Xoshiro256** seeded through splitmix64, mirroring `Random/MegaRandom.cs`.
//! We do not try to reproduce the game's exact stream values (see DESIGN.md,
//! Simulator). What we keep is the split into named streams, so replay
//! injection can override one stream without disturbing the others.

#[derive(Clone, Debug)]
pub struct Rng {
    s: [u64; 4],
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut x = seed;
        let mut next = || {
            x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = x;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        };
        Self { s: [next(), next(), next(), next()] }
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform integer in `0..n`. `n` must be > 0.
    pub fn next_int(&mut self, n: usize) -> usize {
        debug_assert!(n > 0);
        (self.next_u64() % n as u64) as usize
    }

    /// Uniform float in `0..max`.
    pub fn next_float(&mut self, max: f32) -> f32 {
        let unit = (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32;
        unit * max
    }

    /// `rng.NextItem(list)`: a uniform pick, or `None` when empty.
    pub fn pick<'a, T>(&mut self, items: &'a [T]) -> Option<&'a T> {
        if items.is_empty() {
            None
        } else {
            Some(&items[self.next_int(items.len())])
        }
    }

    /// Fisher-Yates, descending index, matching `ListExtensions.UnstableShuffle`.
    pub fn shuffle<T>(&mut self, v: &mut [T]) {
        let mut i = v.len();
        while i > 1 {
            i -= 1;
            let j = self.next_int(i + 1);
            v.swap(i, j);
        }
    }
}

/// The subset of `RunRngType` streams that combat consumes.
#[derive(Clone, Debug)]
pub struct CombatRngs {
    pub shuffle: Rng,
    pub monster_ai: Rng,
    pub targets: Rng,
    /// Monster HP rolls. The game calls this stream `Niche`.
    pub niche: Rng,
    /// `CombatCardGeneration`: random cards created in combat.
    pub card_generation: Rng,
    /// `CombatCardSelection`: random picks among existing cards.
    pub card_selection: Rng,
}

impl CombatRngs {
    pub fn new(seed: u64) -> Self {
        Self {
            shuffle: Rng::new(seed ^ 0x01),
            monster_ai: Rng::new(seed ^ 0x02),
            targets: Rng::new(seed ^ 0x03),
            niche: Rng::new(seed ^ 0x04),
            card_generation: Rng::new(seed ^ 0x05),
            card_selection: Rng::new(seed ^ 0x06),
        }
    }
}
