use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainType {
    Empty,
    Settlement,
    Port,
    Ruin,
    Forest,
    Mountain,
    Ocean,
    Plains,
}

impl TerrainType {
    pub fn from_code(code: i64) -> Option<Self> {
        match code {
            0 => Some(TerrainType::Empty),
            1 => Some(TerrainType::Settlement),
            2 => Some(TerrainType::Port),
            3 => Some(TerrainType::Ruin),
            4 => Some(TerrainType::Forest),
            5 => Some(TerrainType::Mountain),
            10 => Some(TerrainType::Ocean),
            11 => Some(TerrainType::Plains),
            _ => None,
        }
    }

    /// Maps terrain to prediction class index.
    /// Ocean, Plains, Empty -> 0
    /// Settlement -> 1
    /// Port -> 2
    /// Ruin -> 3
    /// Forest -> 4
    /// Mountain -> 5
    pub fn prediction_class(&self) -> usize {
        match self {
            TerrainType::Ocean | TerrainType::Plains | TerrainType::Empty => 0,
            TerrainType::Settlement => 1,
            TerrainType::Port => 2,
            TerrainType::Ruin => 3,
            TerrainType::Forest => 4,
            TerrainType::Mountain => 5,
        }
    }

    /// Plains, Ruin, and Forest are buildable terrain types.
    pub fn is_buildable(&self) -> bool {
        matches!(self, TerrainType::Plains | TerrainType::Ruin | TerrainType::Forest)
    }

    /// Ocean and Mountain are static (cannot change).
    pub fn is_static(&self) -> bool {
        matches!(self, TerrainType::Ocean | TerrainType::Mountain)
    }

    pub fn is_ocean(&self) -> bool {
        matches!(self, TerrainType::Ocean)
    }
}

struct TerrainTypeVisitor;

impl<'de> Visitor<'de> for TerrainTypeVisitor {
    type Value = TerrainType;

    fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        formatter.write_str("an integer terrain code (i64 or u64)")
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
        TerrainType::from_code(v)
            .ok_or_else(|| E::custom(format!("unknown terrain code: {}", v)))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
        // JSON integers may deserialize as u64; cast safely.
        self.visit_i64(v as i64)
    }
}

impl<'de> Deserialize<'de> for TerrainType {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_any(TerrainTypeVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prediction_class() {
        assert_eq!(TerrainType::Ocean.prediction_class(), 0);
        assert_eq!(TerrainType::Plains.prediction_class(), 0);
        assert_eq!(TerrainType::Empty.prediction_class(), 0);
        assert_eq!(TerrainType::Settlement.prediction_class(), 1);
        assert_eq!(TerrainType::Port.prediction_class(), 2);
        assert_eq!(TerrainType::Ruin.prediction_class(), 3);
        assert_eq!(TerrainType::Forest.prediction_class(), 4);
        assert_eq!(TerrainType::Mountain.prediction_class(), 5);
    }

    #[test]
    fn test_is_buildable() {
        assert!(TerrainType::Plains.is_buildable());
        assert!(TerrainType::Ruin.is_buildable());
        assert!(TerrainType::Forest.is_buildable());

        assert!(!TerrainType::Ocean.is_buildable());
        assert!(!TerrainType::Mountain.is_buildable());
        assert!(!TerrainType::Settlement.is_buildable());
        assert!(!TerrainType::Port.is_buildable());
        assert!(!TerrainType::Empty.is_buildable());
    }

    #[test]
    fn test_from_code() {
        assert_eq!(TerrainType::from_code(0), Some(TerrainType::Empty));
        assert_eq!(TerrainType::from_code(1), Some(TerrainType::Settlement));
        assert_eq!(TerrainType::from_code(2), Some(TerrainType::Port));
        assert_eq!(TerrainType::from_code(3), Some(TerrainType::Ruin));
        assert_eq!(TerrainType::from_code(4), Some(TerrainType::Forest));
        assert_eq!(TerrainType::from_code(5), Some(TerrainType::Mountain));
        assert_eq!(TerrainType::from_code(10), Some(TerrainType::Ocean));
        assert_eq!(TerrainType::from_code(11), Some(TerrainType::Plains));
        assert_eq!(TerrainType::from_code(99), None);
    }

    #[test]
    fn test_serde_deserialize_i64() {
        let t: TerrainType = serde_json::from_str("10").unwrap();
        assert_eq!(t, TerrainType::Ocean);
    }

    #[test]
    fn test_serde_deserialize_u64() {
        // Positive integers in JSON are deserialized as u64 by serde_json.
        let t: TerrainType = serde_json::from_str("11").unwrap();
        assert_eq!(t, TerrainType::Plains);
    }

    #[test]
    fn test_is_static() {
        assert!(TerrainType::Ocean.is_static());
        assert!(TerrainType::Mountain.is_static());
        assert!(!TerrainType::Plains.is_static());
    }

    #[test]
    fn test_is_ocean() {
        assert!(TerrainType::Ocean.is_ocean());
        assert!(!TerrainType::Plains.is_ocean());
    }
}
