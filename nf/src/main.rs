use std::thread;
use std::path::Path;

use core_affinity;

use net;

mod function;
use function::*;

mod stack;
use stack::*;

use clap::{App, Arg};
use clap::value_t_or_exit;

mod config;
use config::load_regex_rule_from_file;

#[allow(unused_variables)]
fn run_linux_empty(
    id: usize,
    stack: Stack,
    function: Function,
) {
    
}

#[allow(unused_variables)]
fn run_linux_dataplane_tcpip(
    id: usize,
    stack: Stack,
    function: Function,
) {
    
}

#[allow(unused_variables)]
fn run_linux_tcpip(
    id: usize,
    stack: Stack,
    function: Function,
) {
    
}

#[allow(unused_variables)]
fn run_linux_dataplane_grpc(
    id: usize,
    stack: Stack,
    function: Function,
) {
    
}

#[allow(unused_variables)]
fn run_linux_linux_grpc(
    id: usize,
    stack: Stack,
    function: Function,
) {
    
}

#[allow(unused_variables)]
fn run_empty(
    id: usize,
    stack: Stack,
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
                stack.run(pkt, pkt_len);
            }
        }
        let nb_send = net::net_tx();
    }
}

#[allow(unused_variables)]
fn run_dataplane_tcpip(
    id: usize,
    stack: Stack,
    function: Function,
) {
    
}

#[allow(unused_variables)]
fn run_dataplane_grpc(
    id: usize,
    stack: Stack,
    function: Function,
) {
    
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
                .help("Number of client threads"),
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
                    "null",
                    "tcp/ip",
                    "grpc",
                ])
                .required(true)
                .help("What stack to run with"),
        )
        .arg(
            Arg::with_name("mode")
                .short("mode")
                .long("mode")
                .value_name("MODE")
                .possible_values(&[
                    "linux",
                    "runtime",
                ])
                .required(true)
                .help("Which mode to run in"),
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
        "dataplane-tcp/ip" => Stack::DataPlaneTcpIpStack,
        "linux-tcp/ip" => Stack::LinuxTcpIp,
        "dataplane-grpc" => Stack::DataPlaneGrpc,
        "linux-grpc" => Stack::LinuxGrpc,
        _ => unreachable!(),
    };

    let mut handles = vec![];

    for i in 0..nb_core {
        let matches = matches.clone();
        let core_id = core_ids[i];
        let handle = thread::spawn(move || {
            core_affinity::set_for_current(core_id);
            // Replace the following line with your thread's workload
            println!("Running thread {} on core {:?}", i, core_id.id);
            match matches.value_of("mode").unwrap() {
                "linux" => match stack {
                    Stack::Empty => {
                        run_linux_empty(core_id.id, stack, func);
                    },
                    Stack::DataPlaneTcpIpStack => {
                        run_linux_dataplane_tcpip(core_id.id, stack, func);
                    },
                    Stack::LinuxTcpIp => {
                        run_linux_tcpip(core_id.id, stack, func);
                    },
                    Stack::DataPlaneGrpc => {
                        run_linux_dataplane_grpc(core_id.id, stack, func);
                    },
                    Stack::LinuxGrpc => {
                        run_linux_linux_grpc(core_id.id, stack, func);
                    },
                },
                "runtime" => match stack {
                    Stack::Empty => {
                        run_empty(core_id.id, stack, func);
                    },
                    Stack::DataPlaneTcpIpStack => {
                        run_dataplane_tcpip(core_id.id, stack, func);
                    },
                    Stack::LinuxTcpIp => {
                        println!("Shouldn't use Linux TCP/IP stack with runtime!");
                    },
                    Stack::DataPlaneGrpc => {
                        run_dataplane_grpc(core_id.id, stack, func);
                    },
                    Stack::LinuxGrpc => {
                        println!("Shouldn't use Linux gRPC stack with runtime!");
                    },
                },
                _ => unreachable!(),
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }
}
