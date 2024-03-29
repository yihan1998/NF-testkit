use std::thread;
use std::path::Path;
use std::time::Instant;
use std::time::Duration;

use core_affinity;

use net;

mod function;
use function::*;

mod stack;
use stack::*;

mod network;

use clap::{App, Arg};
use clap::value_t_or_exit;

mod config;
use config::load_regex_rule_from_file;

#[derive(Copy, Clone)]
pub enum Transport {
    Udp,
    Tcp,
}

fn run_empty_stack(
    id: usize,
    stack: Stack,
    tport: Transport,
    function: Function,
) {
    match net::net_setup(id as i32) {
        Ok(_) => println!(" > DPDK setup on core {}!", id),
        Err(e) => println!("An error occurred: {}", e),
    }
    let mut log_time = Instant::now();
    let mut sec_recv = 0;
    let mut sec_send = 0;
    loop {
        if Instant::now() >= log_time {
            println!(" CPU {} | RX => {}, TX => {}", id, sec_recv, sec_send);
            // Schedule the next call
            log_time = Instant::now() + Duration::from_secs(1);
            sec_recv = 0;
            sec_send = 0;
        }

        let nb_recv = net::net_rx();
        if nb_recv > 0 {
            println!(" CPU {}| Receive {} packets!", id as i32, nb_recv);
            sec_recv += nb_recv;
            for i in 0..nb_recv {
                let mut pkt_len: i32 = 0;
                let pkt = net::net_get_rxpkt(&mut pkt_len as *mut i32);
                match stack.run(pkt, pkt_len as usize) {
                    Ok(_) => println!(" Receive packet!"),
                    Err(e) => println!("An error occurred: {}", e),
                }
            }
        }
        let nb_send = net::net_tx();
        if nb_send > 0 {
            sec_send += nb_send;
        }
    }
}

#[allow(unused_variables)]
fn run_full_stack(
    id: usize,
    stack: Stack,
    tport: Transport,
    function: Function,
) {
    match net::net_setup(id as i32) {
        Ok(_) => println!(" > DPDK setup on core {}!", id),
        Err(e) => println!("An error occurred: {}", e),
    }
    loop {
        let nb_recv = net::net_rx();
        if nb_recv > 0 {
            for i in 0..nb_recv {
                let mut pkt_len: i32 = 0;
                let pkt = net::net_get_rxpkt(&mut pkt_len as *mut i32);
                stack.run(pkt, pkt_len as usize);
            }
        }
        let nb_send = net::net_tx();
    }
}

fn main() {
    let matches = App::new("Network Functions")
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
                .help("Rule file"),
        )
        .arg(
            Arg::with_name("function")
                .short("f")
                .long("function")
                .value_name("FUNCTION")
                .possible_values(&[
                    "dns-filter",
                    "flow-monitor",
                ])
                .required(true)
                .help("Which function to run"),
        )
        .arg(
            Arg::with_name("stack")
                .short("s")
                .long("stack")
                .value_name("STACK")
                .possible_values(&[
                    "empty",
                    "full",
                ])
                .required(true)
                .help("What stack to run with"),
        )
        .arg(
            Arg::with_name("transport")
                .short("t")
                .takes_value(true)
                .default_value("udp")
                .help("udp or tcp"),
        )
        .get_matches();

    let nb_core = value_t_or_exit!(matches, "cores", usize);
    let nb_rxq = nb_core.try_into().unwrap();
    let nb_txq = nb_core.try_into().unwrap();

    // Retrieve the IDs of all active CPU cores.
    let core_ids = core_affinity::get_core_ids().unwrap();

    let rule = value_t_or_exit!(matches, "rule", String);
    let rule_path = Path::new(&rule);

    println!(" > Load regex config...");
    let _regexes =  load_regex_rule_from_file(rule_path);
    println!(" > Regex loaded!");

    println!(" > Init DPDK...");
    match net::net_init(nb_core.try_into().unwrap(), nb_rxq, nb_txq) {
        Ok(_) => println!(" > DPDK init done!"),
        Err(e) => println!("An error occurred: {}", e),
    }

    let func_type = matches.value_of("function").unwrap();
    let func = match func_type {
        "dns-filter" => Function::DnsFilter,
        "flow-monitor" => Function::FlowMonitor,
        _ => unreachable!(),
    };

    let stack_type = matches.value_of("stack").unwrap();
    let stack = match stack_type {
        "empty" => Stack::Empty,
        "full" => Stack::FullStack,
        _ => unreachable!(),
    };

    let tport_type = matches.value_of("transport").unwrap();
    let tport = match tport_type {
        "udp" => Transport::Udp,
        "tcp" => Transport::Tcp,
        _ => unreachable!(),
    };

    let mut handles = vec![];

    for i in 0..nb_core {
        let core_id = core_ids[i];
        let handle = thread::spawn(move || {
            core_affinity::set_for_current(core_id);
            // Replace the following line with your thread's workload
            println!("Running thread {} on core {:?}", i, core_id.id);
            match stack {
                Stack::Empty => run_empty_stack(core_id.id, Stack::Empty, tport, func),
                Stack::FullStack => run_full_stack(core_id.id, Stack::FullStack, tport, func),
                _ => unreachable!(),
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
