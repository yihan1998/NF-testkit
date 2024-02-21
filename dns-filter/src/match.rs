trait Matchable {
    fn matches(&self, node: &TreeNode) -> bool;
}

impl Matchable for PacketContext {
    fn matches(&self, node: &TreeNode) -> bool {
        match node {
            TreeNode::And(left, right) => self.matches(left) && self.matches(right),
            TreeNode::Or(left, right) => self.matches(left) || self.matches(right),
            TreeNode::Not(child) => !self.matches(child),
            TreeNode::SourceIPMatch(ip) => &self.header.source_ip == ip,
            TreeNode::PayloadMatch(pattern) => self.payload.windows(pattern.len()).any(|window| window == pattern.as_slice()),
            TreeNode::PortRangeMatch(start_port, end_port) => {
                (self.header.source_port >= *start_port && self.header.source_port <= *end_port) ||
                (self.header.destination_port >= *start_port && self.header.destination_port <= *end_port)
            },
        }
    }
}