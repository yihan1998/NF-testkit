extern crate pnet;

use pnet::datalink::MacAddr;
use pnet::packet::ethernet::EtherTypes;

use etherparse::*;
use etherparse::{LinkSlice::*, NetSlice::*, TransportSlice::*, VlanSlice::*};

pub unsafe fn handle_packet(data: Vec<u8>) -> Result<(), i32> {
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