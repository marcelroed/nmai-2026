use std::fmt;

use serde::de::{self, Visitor};
use serde::{Deserialize, Deserializer};

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
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
    fn from_code(code: i64) -> Result<Self, String> {
        match code {
            0 => Ok(Self::Empty),
            1 => Ok(Self::Settlement),
            2 => Ok(Self::Port),
            3 => Ok(Self::Ruin),
            4 => Ok(Self::Forest),
            5 => Ok(Self::Mountain),
            10 => Ok(Self::Ocean),
            11 => Ok(Self::Plains),
            _ => Err(format!("unknown terrain code: {code}")),
        }
    }
}

impl<'de> Deserialize<'de> for TerrainType {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct TerrainTypeVisitor;

        impl Visitor<'_> for TerrainTypeVisitor {
            type Value = TerrainType;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a valid Astar Island terrain code")
            }

            fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                TerrainType::from_code(value).map_err(E::custom)
            }

            fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
            where
                E: de::Error,
            {
                let value = i64::try_from(value)
                    .map_err(|_| E::custom(format!("terrain code out of range: {value}")))?;
                self.visit_i64(value)
            }
        }

        deserializer.deserialize_i64(TerrainTypeVisitor)
    }
}
