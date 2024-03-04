use std::net::Ipv4Addr;
use std::str::FromStr;
use std::collections::HashMap;
use pnet::datalink::MacAddr;

#[derive(Copy, Clone)]
pub enum Transport {
    Udp,
    Tcp,
}

#[derive(Copy, Clone)]
pub struct PktInfo {
    /* Ethernet header */
    src_mac: MacAddr,
    dst_mac: MacAddr,

    /* IP header */
    src_ip: Ipv4Addr,
    dst_ip: Ipv4Addr,

    /* TCP/UDP header */
    src_port: u16,
    dst_port: u16,

    payload_len: usize,    
}

impl PktInfo {
    pub fn get_src_mac(&self) -> MacAddr { self.src_mac }

    pub fn get_dst_mac(&self) -> MacAddr { self.dst_mac }

    pub fn get_src_ip(&self) -> Ipv4Addr { self.src_ip }

    pub fn get_dst_ip(&self) -> Ipv4Addr { self.dst_ip }

    pub fn get_src_port(&self) -> u16 { self.src_port }

    pub fn set_dst_port(&mut self, dst_port: u16) { self.dst_port = dst_port }
    pub fn get_dst_port(&self) -> u16 { self.dst_port }

    pub fn get_payload_len(&self) -> usize { self.payload_len }
}

pub fn parse_mac_address(mac_str: &str) -> Result<[u8; 6], &'static str> {
    let parts: Vec<&str> = mac_str.split(|c| c == ':' || c == '-').collect();

    if parts.len() != 6 {
        return Err("Invalid MAC address format");
    }

    let mut mac_bytes = [0u8; 6];
    for (i, part) in parts.iter().enumerate() {
        mac_bytes[i] = u8::from_str_radix(part, 16).map_err(|_| "Invalid hex value")?;
    }

    Ok(mac_bytes)
}

pub fn InitPktInfo(config: HashMap<String, String>) -> PktInfo {
    // Accessing values from the config HashMap
    let mut info = PktInfo {
        src_mac: MacAddr::zero(),
        dst_mac: MacAddr::zero(),
        src_ip: Ipv4Addr::new(0, 0, 0, 0),
        dst_ip: Ipv4Addr::new(0, 0, 0, 0),
        src_port: 1234,
        dst_port: 4321,
        payload_len: 64,
    };

    if let Some(src_mac_str) = config.get("src_mac") {
        println!("Src MAC: {}", src_mac_str);
        match MacAddr::from_str(src_mac_str) {
            Ok(src_mac) => info.src_mac = src_mac,
            Err(e) => println!("Error: {}", e),
        }
    }
    if let Some(src_ip_str) = config.get("src_ip") {
        println!("Src IP: {}", src_ip_str);
        match Ipv4Addr::from_str(src_ip_str) {
            Ok(ip) => info.src_ip = ip,
            Err(e) => println!("Failed to parse IP address: {}", e),
        }
    }
    if let Some(dst_mac_str) = config.get("dst_mac") {
        println!("Dst MAC: {}", dst_mac_str);
        match MacAddr::from_str(dst_mac_str) {
            Ok(dst_mac) => info.dst_mac = dst_mac,
            Err(e) => println!("Error: {}", e),
        }
    }
    if let Some(dst_ip_str) = config.get("dst_ip") {
        println!("Dst IP: {}", dst_ip_str);
        match Ipv4Addr::from_str(dst_ip_str) {
            Ok(ip) => info.dst_ip = ip,
            Err(e) => println!("Failed to parse IP address: {}", e),
        }
    }

    // let cfg_src_mac: String = info.src_mac
    //     .iter()
    //     .map(|byte| format!("{:02X}", byte))
    //     .collect::<Vec<String>>().join(":");
    // let cfg_dst_mac: String = info.dst_mac
    //     .iter()
    //     .map(|byte| format!("{:02X}", byte))
    //     .collect::<Vec<String>>().join(":");
    println!("Src MAC: {}, IP: {}, port: {} => Dst MAC: {}, IP: {}, port: {}", 
        info.src_mac.to_string(), info.src_ip.to_string(), info.src_port.to_string(),
        info.dst_mac.to_string(), info.dst_ip.to_string(), info.dst_port.to_string());
    
    info
}