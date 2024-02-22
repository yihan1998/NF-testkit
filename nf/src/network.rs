#[repr(C, packed)]
#[derive(Copy, Clone)]
struct EthernetFrame {
    destination: [u8; 6],
    source: [u8; 6],
    ethertype: u16,
    payload: NetworkLayerProtocol, // Nested network layer protocol
}

pub enum NetworkLayerProtocol {
    ARP(ARPPacket),
    IPv4(IPv4Packet),
    IPv6(IPv6Packet),
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct ARPPacket {
    hrd_type: u16,
    pro_type: u16,
    hrd_len: u8,
    pro_len: u8,
    op: u16,
    sha: [u8; 6],
    sip: [u8; 4],
    tha: [u8; 6],
    tip: [u8; 4],
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct IPv4Packet {
    version: u8,
    ihl: u8,
    total_length: u16,
    protocol: u8,
    source: [u8; 4],
    destination: [u8; 4],
    payload: TransportLayerProtocol, // Nested transport layer protocol
}

#[repr(C, packed)]
#[derive(Copy, Clone)]
struct IPv6Packet {
    version: u8,
    traffic_class: u8,
    flow_label: u32,
    payload_length: u16,
    next_header: u8,
    hop_limit: u8,
    source: [u8; 16],
    destination: [u8; 16],
    payload: TransportLayerProtocol, // Nested transport layer protocol
}

pub enum TransportLayerProtocol {
    TCP(TcpSegment),
    UDP(UdpDatagram),
}

struct TcpSegment {
    source_port: u16,
    destination_port: u16,
    sequence_number: u32,
    acknowledgment_number: u32,
    data_offset: u8,
    flags: u16,
    window_size: u16,
    checksum: u16,
    urgent_pointer: u16,
    // TCP payload (e.g., application data) could be here
}

struct UdpDatagram {
    source_port: u16,
    destination_port: u16,
    length: u16,
    checksum: u16,
    // UDP payload (e.g., application data) could be here
}

impl NetworkLayerProtocol {
    pub fn handle_network_layer(&self) {
        Ok(match *self {
            NetworkLayerProtocol::ARPPacket => {

            }
            NetworkLayerProtocol::IPv4Packet => {

            }
            NetworkLayerProtocol::IPv6Packet => {
                
            }
        })
    }
}