use serde::{Deserialize, Serialize};

/// Index of a material in the loaded material set.
///
/// Only the low 12 bits are used, so a set holds at most 4,096 materials.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub struct MaterialId(pub u16);

impl MaterialId {
    /// Empty space. Always id 0.
    pub const AIR: Self = Self(0);
    /// The edge of the world: static, anchored, indestructible. Always id 1.
    pub const BOUNDARY: Self = Self(1);
    /// Largest number of materials a set can hold.
    pub const MAX_COUNT: usize = 1 << Cell::MATERIAL_BITS;

    /// Returns the id as an index into material tables.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }
}

/// One simulated pixel, packed into 4 bytes.
///
/// Layout: 12-bit material id, 4 flag bits, an 8-bit temperature, and an 8-bit
/// material-specific value. The value holds remaining lifetime for materials that
/// expire, hit points for static solids, fall speed for powders, and a velocity
/// code for liquids: the flow speed of its column on the top cell of resting
/// liquid, or a horizontal and a fall speed code on liquid in the air. The liquid
/// flags [`Cell::FLAG_LIQ_AIR`] and [`Cell::FLAG_LIQ_LEFT`] complete it; they stay
/// clear on every other kind of cell.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Cell {
    bits: u16,
    /// Temperature on the scale described by [`crate::celsius_to_temp`].
    pub temp: u8,
    /// Material-specific value: lifetime, hit points, or fall speed.
    pub value: u8,
}

impl Cell {
    /// Number of bits used for the material id.
    pub const MATERIAL_BITS: u32 = 12;
    const MATERIAL_MASK: u16 = (1 << Self::MATERIAL_BITS) - 1;

    /// Set when the cell was visited on a tick with the same parity as the flag.
    pub const FLAG_PARITY: u16 = 1 << 12;
    /// Set while the cell is part of a stamped rigid body.
    pub const FLAG_IN_BODY: u16 = 1 << 13;
    /// Liquids only: the cell is in the air (falling or thrown) and its value
    /// holds a horizontal and a fall speed code.
    pub const FLAG_LIQ_AIR: u16 = 1 << 14;
    /// Liquids only: the horizontal speed points left. On a resting liquid cell
    /// with a zero speed code it means the pressure pass has checked the cell
    /// and found nothing to do, so it does not ask again until something moves it.
    pub const FLAG_LIQ_LEFT: u16 = 1 << 15;

    /// An air cell at ambient temperature.
    pub const AIR: Self = Self { bits: 0, temp: crate::AMBIENT_TEMP, value: 0 };
    /// The world-edge cell returned for reads outside loaded chunks.
    pub const BOUNDARY: Self =
        Self { bits: MaterialId::BOUNDARY.0, temp: crate::AMBIENT_TEMP, value: 255 };

    /// Creates a cell with no flags set.
    #[inline]
    pub const fn new(material: MaterialId, temp: u8, value: u8) -> Self {
        Self { bits: material.0 & Self::MATERIAL_MASK, temp, value }
    }

    /// Returns the material id.
    #[inline]
    pub const fn material(self) -> MaterialId {
        MaterialId(self.bits & Self::MATERIAL_MASK)
    }

    /// Returns a copy with a different material, keeping flags, temperature and value.
    #[inline]
    #[must_use]
    pub const fn with_material(self, material: MaterialId) -> Self {
        Self {
            bits: (self.bits & !Self::MATERIAL_MASK) | (material.0 & Self::MATERIAL_MASK),
            ..self
        }
    }

    /// Returns true if the given flag bit is set.
    #[inline]
    pub const fn has(self, flag: u16) -> bool {
        self.bits & flag != 0
    }

    /// Returns a copy with the flag bit set or cleared.
    #[inline]
    #[must_use]
    pub const fn with_flag(self, flag: u16, on: bool) -> Self {
        let bits = if on { self.bits | flag } else { self.bits & !flag };
        Self { bits, ..self }
    }

    /// Returns the parity bit as 0 or 1.
    #[inline]
    pub const fn parity(self) -> u32 {
        ((self.bits & Self::FLAG_PARITY) != 0) as u32
    }

    /// Returns a copy with the parity bit set to the low bit of `tick`.
    #[inline]
    #[must_use]
    pub const fn with_parity(self, tick: u64) -> Self {
        self.with_flag(Self::FLAG_PARITY, tick & 1 == 1)
    }

    /// Returns true for air.
    #[inline]
    pub const fn is_air(self) -> bool {
        self.bits & Self::MATERIAL_MASK == 0
    }

    /// Packs the cell into a `u32` (little-endian field order) for hashing and storage.
    #[inline]
    pub const fn to_bits(self) -> u32 {
        self.bits as u32 | (self.temp as u32) << 16 | (self.value as u32) << 24
    }

    /// Unpacks a cell from [`Cell::to_bits`] output.
    #[inline]
    pub const fn from_bits(bits: u32) -> Self {
        Self { bits: bits as u16, temp: (bits >> 16) as u8, value: (bits >> 24) as u8 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_is_four_bytes() {
        assert_eq!(size_of::<Cell>(), 4);
    }

    #[test]
    fn material_and_flags_are_independent() {
        let c = Cell::new(MaterialId(4095), 9, 200).with_flag(Cell::FLAG_IN_BODY, true);
        assert_eq!(c.material(), MaterialId(4095));
        assert!(c.has(Cell::FLAG_IN_BODY));
        let c = c.with_material(MaterialId(3));
        assert_eq!(c.material(), MaterialId(3));
        assert!(c.has(Cell::FLAG_IN_BODY));
        assert_eq!(c.temp, 9);
        assert_eq!(c.value, 200);
    }

    #[test]
    fn bits_round_trip() {
        let c = Cell::new(MaterialId(77), 31, 5).with_parity(1);
        assert_eq!(Cell::from_bits(c.to_bits()), c);
    }
}
