use std::thread;
use std::path::Path;

use regex::Regex;

use core_affinity;

use net;

use clap::{App, Arg};
use clap::value_t_or_exit;

mod config;
use config::load_regex_rule_from_file;

fn main() {
    let matches = App::new("DNS filter")
        .version("0.1")
        .arg(
            Arg::with_name("cores")
                .short("c")
                .long("cores")
                .default_value("1")
                .help("Number of cores"),
        )
        .arg(
            Arg::with_name("rule")
                .short("r")
                .long("rule")
                .value_name("FILE")
                .default_value("1")
                .help("Number of client threads"),
        )
        .get_matches();

    let nr_core = value_t_or_exit!(matches, "cores", usize);
    let net_nr_core = nr_core.try_into().unwrap();
    let net_nr_rxq = net_nr_core;
    let net_nr_txq = net_nr_core;

    let rule = value_t_or_exit!(matches, "rule", String);
    let rule_path = Path::new(&rule);

    println!(" > Load regex config...");
    let regexes =  load_regex_rule_from_file(rule_path);
    println!(" > Regex loaded!");

    println!(" > Init DPDK...");
    match net::net_init(net_nr_core, net_nr_rxq, net_nr_txq) {
        Ok(_) => println!(" > DPDK init done!"),
        Err(e) => println!("An error occurred: {}", e),
    }

    let core_ids = core_affinity::get_core_ids().expect("Failed to get core IDs");

    let mut handles = vec![];

    for i in 0..nr_core {
        let core_id = core_ids[i];
        let handle = thread::spawn(move || {
            core_affinity::set_for_current(core_id);
            // Replace the following line with your thread's workload
            println!("Running thread {} on core {:?}", i, core_id.id);
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
