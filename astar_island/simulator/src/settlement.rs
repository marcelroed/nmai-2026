use serde::Deserialize;

/// Runtime settlement state (uses f32 for memory efficiency).
#[derive(Debug, Clone)]
pub struct Settlement {
    pub x: usize,
    pub y: usize,
    pub population: f32,
    pub food: f32,
    pub wealth: f32,
    pub defense: f32,
    pub has_port: bool,
    pub alive: bool,
    pub owner_id: u16,
    pub tech_level: f32,
    pub longships: u32,
    pub total_damage: f32,
}

/// Replay settlement as parsed from JSON (uses f64 to match JSON format).
#[derive(Debug, Clone, Deserialize)]
pub struct ReplaySettlement {
    pub x: usize,
    pub y: usize,
    pub population: f64,
    pub food: f64,
    pub wealth: f64,
    pub defense: f64,
    pub has_port: bool,
    pub alive: bool,
    pub owner_id: u16,
}

/// Initial settlement as parsed from JSON (only spatial/structural fields).
#[derive(Debug, Clone, Deserialize)]
pub struct InitialSettlement {
    pub x: usize,
    pub y: usize,
    pub has_port: bool,
    pub alive: bool,
}

impl Settlement {
    /// Convert a replay settlement (f64 fields) to a runtime settlement (f32 fields).
    pub fn from_replay(r: &ReplaySettlement) -> Self {
        Settlement {
            x: r.x,
            y: r.y,
            population: r.population as f32,
            food: r.food as f32,
            wealth: r.wealth as f32,
            defense: r.defense as f32,
            has_port: r.has_port,
            alive: r.alive,
            owner_id: r.owner_id,
            tech_level: 0.0,
            longships: 0,
            total_damage: 0.0,
        }
    }

    /// Construct a settlement from initial structural data plus explicit stat values.
    pub fn from_initial(
        init: &InitialSettlement,
        pop: f32,
        food: f32,
        wealth: f32,
        defense: f32,
        owner_id: u16,
    ) -> Self {
        Settlement {
            x: init.x,
            y: init.y,
            population: pop,
            food,
            wealth,
            defense,
            has_port: init.has_port,
            alive: init.alive,
            owner_id,
            tech_level: 0.0,
            longships: 0,
            total_damage: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_replay() {
        let r = ReplaySettlement {
            x: 3,
            y: 7,
            population: 100.5,
            food: 200.75,
            wealth: 50.0,
            defense: 10.25,
            has_port: true,
            alive: true,
            owner_id: 42,
        };
        let s = Settlement::from_replay(&r);
        assert_eq!(s.x, 3);
        assert_eq!(s.y, 7);
        assert!((s.population - 100.5_f32).abs() < 1e-3);
        assert!((s.food - 200.75_f32).abs() < 1e-3);
        assert_eq!(s.has_port, true);
        assert_eq!(s.alive, true);
        assert_eq!(s.owner_id, 42);
    }

    #[test]
    fn test_from_initial() {
        let init = InitialSettlement {
            x: 5,
            y: 9,
            has_port: false,
            alive: true,
        };
        let s = Settlement::from_initial(&init, 50.0, 100.0, 25.0, 5.0, 1);
        assert_eq!(s.x, 5);
        assert_eq!(s.y, 9);
        assert_eq!(s.population, 50.0);
        assert_eq!(s.food, 100.0);
        assert_eq!(s.wealth, 25.0);
        assert_eq!(s.defense, 5.0);
        assert_eq!(s.has_port, false);
        assert_eq!(s.alive, true);
        assert_eq!(s.owner_id, 1);
    }

    #[test]
    fn test_replay_settlement_deserialize() {
        let json = r#"{"x":1,"y":2,"population":10.0,"food":20.0,"wealth":5.0,"defense":3.0,"has_port":false,"alive":true,"owner_id":7}"#;
        let r: ReplaySettlement = serde_json::from_str(json).unwrap();
        assert_eq!(r.x, 1);
        assert_eq!(r.y, 2);
        assert_eq!(r.owner_id, 7);
    }

    #[test]
    fn test_initial_settlement_deserialize() {
        let json = r#"{"x":4,"y":6,"has_port":true,"alive":false}"#;
        let init: InitialSettlement = serde_json::from_str(json).unwrap();
        assert_eq!(init.x, 4);
        assert_eq!(init.y, 6);
        assert_eq!(init.has_port, true);
        assert_eq!(init.alive, false);
    }
}
