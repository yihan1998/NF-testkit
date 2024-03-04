use crate::Packet;

use std::error::Error;
use std::time::Duration;

use bincode;
use serde::{Serialize, Deserialize};

use trust_dns_proto::{
    op::{Message, Query},
    rr::{Name, RecordType},
    serialize::binary::{BinEncodable, BinEncoder}
};

#[derive(Copy, Clone)]
pub enum Function {
    DnsFilter,
    FlowMonitor,
}

fn serialize_message_and_packet(message: &Message, packet: &mut Packet) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Serialize the DNS message using trust-dns-proto built-in methods
    let mut buffer_msg = Vec::new();
    {
        let mut encoder = trust_dns_proto::serialize::binary::BinEncoder::new(&mut buffer_msg);
        message.emit(&mut encoder)?;
    }

    // Serialize the Packet struct using Bincode
    let buffer_pkt = bincode::serialize(packet)?;

    // Concatenate both buffers
    let mut combined_buffer = buffer_msg;
    combined_buffer.extend_from_slice(&buffer_pkt);

    Ok(combined_buffer)
}

impl Function {
    pub fn gen_request(&self, packet: &mut Packet, now: Duration) -> Vec<u8> {
        match *self {
            Function::DnsFilter => {
                let mut message = Message::new();
                let name = Name::from_ascii("google.com.").unwrap(); // Domain name for the query
                let query = Query::query(name, RecordType::A); // Looking for A records
                
                message.add_query(query);
                message.set_id(1234); // Set a transaction ID
                message.set_recursion_desired(true); // Set the RD (Recursion Desired) flag

                packet.actual_start = Some(now);

                match serialize_message_and_packet(&message, packet) {
                    Ok(data) => { return data },
                    Err(_) => { println!("Failed to serialize data!") },
                }
            },
            Function::FlowMonitor => {

            },
        }

        vec![]
    }
}