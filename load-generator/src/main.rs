use std::io;
use std::env;
use std::thread;
use std::time::Instant;
use std::time::Duration;
use std::sync::atomic::AtomicU64;
use std::collections::{BTreeMap};
use serde::{Serialize, Deserialize};

use core_affinity;

use rand::Rng;
use rand_mt::Mt64;

use net;

mod function;
use function::*;

mod stack;
use stack::*;

mod pktinfo;
use pktinfo::{Transport, PktInfo, InitPktInfo};

use clap::{App, Arg};
use clap::value_t_or_exit;

mod distribution;
use distribution::Distribution;

mod config;
use config::read_config_file;

#[derive(Copy, Clone)]
struct RequestSchedule {
    arrival: Distribution,
    runtime: Duration,
    discard_pct: f32,
}

struct ScheduleResult {
    packet_count: usize,
    drop_count: usize,
    never_sent_count: usize,
    first_send: Option<Duration>,
    last_send: Option<Duration>,
    latencies: BTreeMap<u64, usize>,
    first_tsc: Option<u64>,
}

#[derive(Default, Serialize, Deserialize)]
pub struct Packet {
    magic: u32,
    target_start: Duration,
    actual_start: Option<Duration>,
    completion_time: Option<Duration>,
}

fn gen_packets_for_schedule(sched: RequestSchedule) -> Vec<Packet> {
    let mut packets: Vec<Packet> = Vec::new();
    let mut rng: Mt64 = Mt64::new(rand::thread_rng().gen::<u64>());
    let mut last = 0;
    let end = sched.runtime.as_secs() * 1000_000_000;
    loop {
        // println!("Pkt target start @{}", last);
        packets.push(Packet {
            magic: 0x1234ABCD,
            target_start: Duration::from_nanos(last),
            ..Default::default()
        });

        let nxt = last + sched.arrival.sample(&mut rng);
        if nxt >= end {
            break;
        }
        last = nxt;
    }

    packets
}

fn process_result_final(sched: RequestSchedule) -> bool {
    return true
}

fn run_empty_stack(
    id: usize,
    packets_per_second: usize,
    stack: Stack,
    tport: Transport,
    function: Function,
    info: PktInfo
)  -> (u32, u32) {
    let ns_per_packet = 1000_000_000 / packets_per_second;

    match net::net_setup(id as i32) {
        Ok(_) => {},
        Err(e) => println!("An error occurred: {}", e),
    }

    let sched = RequestSchedule {
        arrival: Distribution::Exponential(ns_per_packet as f64),
        runtime: Duration::from_secs(20),
        discard_pct: 15.0,
    };

    let mut packets = gen_packets_for_schedule(sched);

    let mut log_time = Instant::now();
    let start_time = Instant::now();

    let mut sec_recv = 0u32;
    let mut sec_send = 0u32;
    let mut tot_recv = 0u32;
    let mut tot_send = 0u32;

    let mut last_sent = 0;

    loop {
        let now = Instant::now();
        if now >= log_time {
            let sec_recv_rate = sec_recv as f32 / Duration::from_secs(1).as_millis() as f32;
            let sec_send_rate = sec_send as f32 / Duration::from_secs(1).as_millis() as f32;
            println!(" CPU {} | RX => {}/{} Kpps, TX => {}/{} Kpps", id, sec_recv, sec_recv_rate, sec_send, sec_send_rate);
            // Schedule the next call
            log_time = now + Duration::from_secs(1);
            tot_recv += sec_recv;
            tot_send += sec_send;
            sec_recv = 0;
            sec_send = 0;
        }

        if start_time.elapsed() >= sched.runtime {
            let elapsed = start_time.elapsed().as_micros() as f32;
            let tot_recv_rate = tot_recv as f32 / elapsed;
            let tot_send_rate = tot_send as f32 / elapsed;
            println!(" CPU {} | Avg. RX => {:.4} Mpps, Avg. TX => {:.4} Mpps", id, tot_recv_rate, tot_send_rate);
            break;
        }

        let nb_recv = net::net_rx() as u32;
        if nb_recv > 0 {
            println!(" CPU {}| Receive {} packets!", id as i32, nb_recv);
            sec_recv += nb_recv;
            for i in 0..nb_recv {
                let mut pkt_len: i32 = 0;
                let pkt = net::net_get_rxpkt(&mut pkt_len as *mut i32);
                match stack.rx(pkt, pkt_len as usize) {
                    Ok(_) => println!(" Receive packet!"),
                    Err(e) => println!("An error occurred: {}", e),
                }
            }
        }

        loop {
            if last_sent >= packets.len() {
                break;
            }

            let mut packet_to_send = &mut packets[last_sent];

            // println!("Next pkt target start: {:?}, elapsed {:?}", packet_to_send.target_start, start_time.elapsed());

            /* Set up all packets, time to send */
            if packet_to_send.target_start > start_time.elapsed() {
                break;
            }

            let payload = function.gen_request(packet_to_send, start_time.elapsed());
            match stack.tx(tport, info, &payload, payload.len()) {
                Ok(_) => last_sent += 1,
                Err(e) => println!("An error occurred during TX: {}", e),
            }
        }

        let nb_send = net::net_tx() as u32;
        if nb_send > 0 {
            sec_send += nb_send;
        }
    }

    // process_result_final(sched);

    (tot_recv, tot_send)
}

fn main() -> io::Result<()>  {
    env::set_var("RUST_BACKTRACE", "1");
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
        .arg(
            Arg::with_name("mpps")
                .short("m")
                .takes_value(true)
                .default_value("1.0")
                .help("How many *million* packets should be sent per second"),
        )
        .get_matches();

    let nb_cores = value_t_or_exit!(matches, "cores", usize);
    let nb_rxq = nb_cores.try_into().unwrap();
    let nb_txq = nb_cores.try_into().unwrap();

    let packets_per_second = (1.0e6 * value_t_or_exit!(matches, "mpps", f32)) as usize;
    let per_core_pps = packets_per_second / nb_cores;

    // Retrieve the IDs of all active CPU cores.
    let core_ids = core_affinity::get_core_ids().unwrap();

    println!(" > Init DPDK...");
    match net::net_init(nb_cores.try_into().unwrap(), nb_rxq, nb_txq) {
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

    let config = read_config_file("pktgen.spec")?;
    let mut info = InitPktInfo(config);
    /* Make some adjustments */
    match func {
        Function::DnsFilter => info.set_dst_port(53),
        Function::FlowMonitor => (),
    };

    let mut handles = vec![];
    let mut tot_recv = 0;
    let mut tot_send = 0;

    let start_time = Instant::now();

    for i in 0..nb_cores {
        let core_id = core_ids[i];
        let handle = thread::spawn(move || {
            core_affinity::set_for_current(core_id);
            // Replace the following line with your thread's workload
            match stack {
                Stack::Empty => run_empty_stack(core_id.id, per_core_pps, Stack::Empty, tport, func, info),
                // Stack::FullStack => run_full_stack(core_id.id, Stack::FullStack, tport, func),
                _ => unreachable!(),
            }
        });
        handles.push(handle);
    }

    for handle in handles {
        let (per_core_recv, per_core_send) =  handle.join().unwrap();
        tot_recv += per_core_recv;
        tot_send += per_core_send;
    }

    let elapsed = start_time.elapsed().as_micros() as f32;
    let tot_recv_rate = tot_recv as f32 / elapsed;
    let tot_send_rate = tot_send as f32 / elapsed;

    println!(" Avg. RX => {:.5} Mpps, Avg. TX => {:.5} Mpps", tot_recv_rate, tot_send_rate);

    Ok(())
}
