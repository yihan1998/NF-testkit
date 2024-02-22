use regex::Regex;

use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

pub fn load_regex_rule_from_file (path: &Path) -> Vec<Regex> {
    // Open the file
    let file = File::open(path).unwrap();
    let reader = io::BufReader::new(file);

    let regexes = reader.lines()
        .filter_map(|line| line.ok()) // Filter out any lines that can't be read
        .filter_map(|line| Regex::new(&line).ok()) // Try to compile each line as a regex, ignoring errors
        .collect::<Vec<_>>();

    regexes.iter().for_each(|regex| println!("\t{}", regex.as_str()));

    regexes
}