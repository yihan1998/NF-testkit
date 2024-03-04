use std::collections::HashMap;
use std::fs::File;
use std::io::{self, BufRead, BufReader};

pub fn read_config_file(file_path: &str) -> Result<HashMap<String, String>, io::Error> {
    let file = File::open(file_path)?;
    let reader = BufReader::new(file);

    let mut config = HashMap::new();

    for line in reader.lines() {
        let line = line?;
        let parts: Vec<&str> = line.splitn(2, ':').map(|s| s.trim()).collect();

        if parts.len() == 2 {
            config.insert(parts[0].to_string(), parts[1].to_string());
        }
    }

    Ok(config)
}