use serde::{Serialize, Deserialize};

/// Declare all simulation parameters in one place.
/// Each entry: (field_name, default, lower_bound, upper_bound)
///
/// The macro generates:
///   - The `Params` struct with `pub` f32 fields
///   - `default_prior()` from the default column
///   - `to_vec()` / `from_vec()` in declaration order
///   - `lower_bounds()` / `upper_bounds()` vectors
///   - `field_names()` for debug/serialization
macro_rules! define_params {
    ( $( $field:ident : $default:expr , $lo:expr , $hi:expr ; )* ) => {
        /// Simulation hyperparameters.
        ///
        /// Serializes to/from JSON as a named struct. The CMA-ES optimiser
        /// uses `to_vec` / `from_vec` for flat-vector access in declaration order.
        #[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
        pub struct Params {
            $( pub $field: f32, )*
        }

        impl Params {
            /// Number of parameters.
            pub const N: usize = { let mut _n = 0usize; $( let _ = stringify!($field); _n += 1; )* _n };

            /// Cross-round average values used as the prior / default parameter set.
            pub fn default_prior() -> Self {
                Params { $( $field: $default, )* }
            }

            /// Pack parameters into a `Vec<f32>` in declaration order.
            pub fn to_vec(&self) -> Vec<f32> {
                vec![ $( self.$field, )* ]
            }

            /// Unpack a `Vec<f32>` (declaration order) into `Params`.
            pub fn from_vec(v: &[f32]) -> Self {
                assert_eq!(v.len(), Self::N,
                    "Params::from_vec expects exactly {} elements, got {}", Self::N, v.len());
                let mut _i = 0usize;
                $( let $field = v[_i]; _i += 1; )*
                Params { $( $field, )* }
            }

            /// Lower bounds in declaration order.
            pub fn lower_bounds() -> Vec<f32> {
                vec![ $( $lo, )* ]
            }

            /// Upper bounds in declaration order.
            pub fn upper_bounds() -> Vec<f32> {
                vec![ $( $hi, )* ]
            }

            /// Field names in declaration order (useful for diagnostics).
            pub fn field_names() -> Vec<&'static str> {
                vec![ $( stringify!($field), )* ]
            }
        }
    };
}

define_params! {
    // ── Food production ─────────────────────────────────────────────────
    //                          default    lo      hi
    food_weather_std:           0.0,       0.0,    0.08;
    food_pop_coeff:            -0.113,    -0.20,   0.0;
    food_feedback:              0.91,      0.5,    1.2;
    food_plains:                0.043,     0.02,   0.07;
    food_forest:                0.074,     0.03,   0.12;
    food_mountain:              0.0,      -0.02,   0.02;
    food_settlement:           -0.017,    -0.04,   0.01;

    // ── Population dynamics ─────────────────────────────────────────────
    pop_growth_rate:            0.06,      0.0,    0.20;
    pop_max:                    5.0,       2.0,    8.0;
    pop_wealth_coeff:           0.01,      0.0,    0.05;
    port_upgrade_prob:          0.10,      0.0,    0.20;

    // ── Defense ─────────────────────────────────────────────────────────
    defense_recovery_rate:      0.02,      0.0,    0.10;

    // ── Growth ──────────────────────────────────────────────────────────
    growth_prob:                0.10,      0.01,   0.50;
    spawn_distance_max:         3.0,       1.5,   15.0;
    parent_cost:                0.3,       0.1,    0.8;

    // ── Conflict ────────────────────────────────────────────────────────
    raid_prob:                  0.25,      0.01,   0.40;
    raid_range:                 3.0,       1.5,   15.0;
    longship_range_bonus:       2.0,       0.0,    5.0;
    aggression:                 1.5,       0.0,    3.0;
    raid_defense_mult:          0.80,      0.50,   1.0;
    raid_success_prob:          0.50,      0.0,    1.0;
    raid_steal_frac:            0.30,      0.0,    0.80;
    raid_damage_inc:            0.20,      0.02,   0.50;
    raid_kill_scale:            0.40,      0.0,    1.0;
    raid_takeover_scale:        0.20,      0.0,    1.0;

    // ── Longships ───────────────────────────────────────────────────────
    longship_build_prob:        0.10,      0.0,    0.30;
    longship_cost:              0.10,      0.02,   0.30;

    // ── Trade ───────────────────────────────────────────────────────────
    trade_range:                5.0,       2.0,   10.0;
    trade_food_bonus:           0.02,      0.0,    0.10;
    trade_wealth_bonus:         0.01,      0.0,    0.05;
    tech_diffusion_rate:        0.1,       0.0,    0.30;

    // ── Winter ──────────────────────────────────────────────────────────
    winter_severity:            0.045,     0.0,    0.10;
    collapse_threshold:         0.12,      0.0,    0.20;
    catastrophe_freq:           0.40,      0.0,    1.50;
    catastrophe_death_rate:     0.25,      0.0,    0.60;

    // ── Environment ─────────────────────────────────────────────────────
    ruin_to_settlement:         0.70,      0.30,   1.50;
    ruin_to_forest:             0.18,      0.03,   0.30;
    reclaim_range:              3.0,       1.0,    5.0;

    // ── Initial stats ───────────────────────────────────────────────────
    init_pop_mean:              1.0,       0.5,    1.5;
    init_food_mean:             0.55,      0.2,    0.9;
    init_wealth_mean:           0.30,      0.05,   0.6;
    init_defense_mean:          0.40,      0.1,    0.7;

    // ── Extended dynamics ───────────────────────────────────────────────
    starvation_rate:            0.10,      0.0,    0.50;
    food_ocean:                 0.01,     -0.02,   0.05;
    winter_dispersal_range:     3.0,       1.5,    6.0;
    child_pop:                  0.5,       0.2,    1.0;

    // ── Spawn & raid distance + terrain ─────────────────────────────────
    spawn_weight_forest:        1.2,       0.1,    2.0;
    spawn_distance_decay:       0.3,       0.0,    1.0;
    spawn_terrain_bonus:        0.1,       0.0,    0.5;
    raid_distance_decay:        0.2,       0.0,    1.0;
    tech_food_coeff:            0.05,      0.0,    0.3;
    tech_growth_coeff:          0.05,      0.0,    0.3;
    port_ocean_bonus:           0.3,       0.0,    1.0;
    wealth_decay:               0.014,     0.0,    0.10;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_to_from_vec() {
        let p = Params::default_prior();
        let v = p.to_vec();
        let p2 = Params::from_vec(&v);
        assert_eq!(p, p2);
    }

    #[test]
    fn test_vec_length() {
        let p = Params::default_prior();
        assert_eq!(p.to_vec().len(), Params::N);
    }

    #[test]
    fn test_bounds_length() {
        assert_eq!(Params::lower_bounds().len(), Params::N);
        assert_eq!(Params::upper_bounds().len(), Params::N);
    }

    #[test]
    fn test_bounds_ordering() {
        let lb = Params::lower_bounds();
        let ub = Params::upper_bounds();
        for (i, (lo, hi)) in lb.iter().zip(ub.iter()).enumerate() {
            assert!(lo <= hi, "lower_bound[{}]={} > upper_bound[{}]={}", i, lo, i, hi);
        }
    }

    #[test]
    fn test_default_within_bounds() {
        let p = Params::default_prior();
        let v = p.to_vec();
        let lb = Params::lower_bounds();
        let ub = Params::upper_bounds();
        let names = Params::field_names();
        for i in 0..Params::N {
            assert!(v[i] >= lb[i] && v[i] <= ub[i],
                "{}: default {} not in [{}, {}]", names[i], v[i], lb[i], ub[i]);
        }
    }

    #[test]
    fn test_json_roundtrip() {
        let p = Params::default_prior();
        let json = serde_json::to_string(&p).unwrap();
        let p2: Params = serde_json::from_str(&json).unwrap();
        assert_eq!(p, p2);
    }

    #[test]
    #[should_panic(expected = "expects exactly")]
    fn test_from_vec_wrong_length_panics() {
        Params::from_vec(&[0.0; 3]);
    }
}
