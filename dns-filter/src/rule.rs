trait MatchRule {
    fn matches(&self, context: &PacketContext) -> bool;
}

// Matches based on source IP in the packet header
struct SourceIPMatch {
    ip: String,
}

impl MatchRule for SourceIPMatch {
    fn matches(&self, context: &PacketContext) -> bool {
        self.ip == context.header.source_ip
    }
}

// Matches based on a specific byte pattern in the packet payload
struct PayloadMatch {
    pattern: Vec<u8>,
}

impl MatchRule for PayloadMatch {
    fn matches(&self, context: &PacketContext) -> bool {
        // Example implementation; real-world might use more efficient searching
        context.payload.windows(self.pattern.len()).any(|window| window == self.pattern.as_slice())
    }
}