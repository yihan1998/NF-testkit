struct PacketContext {
    header: PacketHeader,
    payload: Vec<u8>, // Assuming payload as bytes for simplicity
}

struct PacketHeader {
    source_ip: String,
    destination_ip: String,
    protocol: String,
    // Add other relevant header fields
}
