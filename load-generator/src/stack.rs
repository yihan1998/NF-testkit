use crate::Transport;
use crate::PktInfo;

use std::slice;

use etherparse::*;
use etherparse::{LinkSlice::*, NetSlice::*, TransportSlice::*, VlanSlice::*};

use pnet::packet::ethernet::{EtherTypes, EthernetPacket, MutableEthernetPacket};
use pnet::packet::ipv4::{Ipv4Packet, MutableIpv4Packet, Ipv4Flags};
use pnet::packet::udp::{MutableUdpPacket, UdpPacket};
use pnet::packet::{Packet, PacketSize, ip::IpNextHeaderProtocols};
use pnet::util::MacAddr;

use std::net::{Ipv4Addr};

#[derive(Copy, Clone)]
pub enum Stack {
    Empty,  // Used for packet-based test
    FullStack,
}

unsafe fn parse_packet(data: Vec<u8>) -> Result<(), i32> {
    let packet = SlicedPacket::from_ethernet(&data);

    match packet {
        Err(value) => println!("Err {:?}", value),
        Ok(value) => {
            match value.link {
                Some(Ethernet2(value)) => println!(
                    "  Ethernet2 {:?} => {:?}",
                    value.source(),
                    value.destination()
                ),
                Some(EtherPayload(payload)) => {
                    println!("  EtherPayload (ether type {:?})", payload.ether_type)
                }
                None => {}
            }

            match value.vlan {
                Some(SingleVlan(value)) => println!("  SingleVlan {:?}", value.vlan_identifier()),
                Some(DoubleVlan(value)) => println!(
                    "  DoubleVlan {:?}, {:?}",
                    value.outer().vlan_identifier(),
                    value.inner().vlan_identifier()
                ),
                None => {}
            }

            match value.net {
                Some(Ipv4(ipv4)) => {
                    println!(
                        "  Ipv4 {:?} => {:?}",
                        ipv4.header().source_addr(),
                        ipv4.header().destination_addr()
                    );
                    if false == ipv4.extensions().is_empty() {
                        println!("    {:?}", ipv4.extensions());
                    }
                }
                Some(Ipv6(ipv6)) => {
                    println!(
                        "  Ipv6 {:?} => {:?}",
                        ipv6.header().source_addr(),
                        ipv6.header().destination_addr()
                    );
                    if false == ipv6.extensions().is_empty() {
                        println!("    {:?}", ipv6.extensions());
                    }
                }
                None => {}
            }

            match value.transport {
                Some(Icmpv4(value)) => println!(" Icmpv4 {:?}", value),
                Some(Icmpv6(value)) => println!(" Icmpv6 {:?}", value),
                Some(Udp(value)) => println!(
                    "  UDP {:?} -> {:?}",
                    value.source_port(),
                    value.destination_port()
                ),
                Some(Tcp(value)) => {
                    println!(
                        "  TCP {:?} -> {:?}",
                        value.source_port(),
                        value.destination_port()
                    );
                    let options: Vec<Result<TcpOptionElement, TcpOptionReadError>> =
                        value.options_iterator().collect();
                    println!("    {:?}", options);
                }
                None => {}
            }
        }
    }

    return Ok(());
}

impl Stack {
    pub fn rx(
        &self,
        pkt: *const u8,
        len: usize,
    ) -> Result<(), i32> {
        match *self {
            Stack::Empty => unsafe {
                let data: Vec<u8> = slice::from_raw_parts(pkt, len).to_vec();
                return parse_packet(data); 
            },
            Stack::FullStack => {
                return Ok(());
            },
        }
    }

    pub fn tx(
        &self,
        tport: Transport,
        info: PktInfo,
        payload: &[u8],
        payload_len: usize
    ) -> Result<(), &str> {
        match *self {
            Stack::Empty => {
                match tport {
                    Transport::Tcp => {
                        return Err("Unimplemented TCP!");
                    },
                    Transport::Udp => {
                        let src_mac = info.get_src_mac();
                        let dst_mac = info.get_dst_mac();
                        let src_ip = info.get_src_ip();
                        let dst_ip = info.get_dst_ip();
                        let src_port = info.get_src_port();
                        let dst_port = info.get_dst_port();

                        // Ethernet frame buffer
                        let mut ethernet_buffer = [0u8; 1500]; // Max Ethernet frame size for most networks
                        let mut ethernet_packet = MutableEthernetPacket::new(&mut ethernet_buffer).unwrap();

                        // Set Ethernet fields
                        ethernet_packet.set_source(src_mac);
                        ethernet_packet.set_destination(dst_mac);
                        ethernet_packet.set_ethertype(EtherTypes::Ipv4);

                        // IPv4 packet buffer
                        let mut ipv4_buffer = [0u8; 1500]; // Temporary buffer
                        let mut ipv4_packet = MutableIpv4Packet::new(&mut ipv4_buffer).unwrap();

                        // Set IPv4 fields
                        ipv4_packet.set_version(4);
                        ipv4_packet.set_header_length(5);
                        ipv4_packet.set_total_length(20 + 8 + payload_len as u16); // IP header + UDP header + payload
                        ipv4_packet.set_ttl(64);
                        ipv4_packet.set_next_level_protocol(IpNextHeaderProtocols::Udp);
                        ipv4_packet.set_source(src_ip);
                        ipv4_packet.set_destination(dst_ip);

                        // UDP packet buffer and setup
                        let mut udp_buffer = [0u8; 1500]; // Temporary buffer for the UDP packet
                        let mut udp_packet = MutableUdpPacket::new(&mut udp_buffer).unwrap();

                        // Set UDP fields
                        udp_packet.set_source(src_port); // Example source port
                        udp_packet.set_destination(dst_port); // Example destination port
                        udp_packet.set_length(8 + payload_len as u16); // UDP header + payload
                        udp_packet.set_payload(payload);

                        // Calculate sizes and set payloads correctly
                        let udp_packet_size = udp_packet.packet_size() + payload_len;
                        ipv4_packet.set_payload(&udp_packet.packet()[..udp_packet_size]);

                        let ipv4_packet_size = ipv4_packet.packet_size();
                        ethernet_packet.set_payload(&ipv4_packet.packet()[..ipv4_packet_size]);

                        let total_length = ethernet_packet.packet_size() + ipv4_packet_size;

                        let pkt = net::net_get_txpkt(total_length as i32);
                        if pkt.is_null() {
                            return Err("Fail to get new TX buffer!");
                        }

                        unsafe {
                            std::ptr::copy_nonoverlapping(ethernet_packet.packet().to_vec().as_ptr(), pkt, total_length);
                        }

                        return Ok(())
                    },
                }
            },
            Stack::FullStack => {
                return Ok(())
            },
        }
    }
}