use rand::Rng;
use rand_distr::{Distribution as DistR, Exp};

#[derive(Copy, Clone, Debug)]
pub enum Distribution {
    Exponential(f64),
}

impl Distribution {
    pub fn name(&self) -> &'static str {
        match *self {
            Distribution::Exponential(_) => "exponential",
        }
    }
    pub fn sample<R: Rng>(&self, rng: &mut R) -> u64 {
        match *self {
            Distribution::Exponential(m) => Exp::new(1.0 / m).unwrap().sample(rng) as u64,
        }
    }

    pub fn create(spec: &str) -> Result<Self, &str> {
        let tokens: Vec<&str> = spec.split(":").collect();
        assert!(tokens.len() > 0);
        match tokens[0] {
            "exponential" => {
                assert!(tokens.len() == 2);
                let val: f64 = tokens[1].parse().unwrap();
                Ok(Distribution::Exponential(val))
            }
            _ => Err("bad distribution spec"),
        }
    }
}
