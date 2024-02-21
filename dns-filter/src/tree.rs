// Define the tree node structure with the abstracted MatchType
enum TreeNode {
    And(Box<TreeNode>, Box<TreeNode>),
    Or(Box<TreeNode>, Box<TreeNode>),
    Not(Box<TreeNode>),
    SourceIPMatch(String),
    PayloadMatch(Vec<u8>),
    PortRangeMatch(u16, u16), // Stores (start_port, end_port)
}

impl TreeNode {
    fn evaluate(&self, ip_address: &str, tcp_port: u16) -> bool {
        match self {
            TreeNode::IPAddress(pattern) => pattern == ip_address,
            TreeNode::TCPPort(port) => *port == tcp_port,
            TreeNode::And(left, right) => left.evaluate(ip_address, tcp_port) && right.evaluate(ip_address, tcp_port),
            TreeNode::Or(left, right) => left.evaluate(ip_address, tcp_port) || right.evaluate(ip_address, tcp_port),
        }
    }

    // Function to build an AND relation tree from a set of rules
    fn and_relation(rules: Vec<TreeNode>) -> Option<TreeNode> {
        rules.into_iter().reduce(|acc, rule| TreeNode::And(Box::new(acc), Box::new(rule)))
    }
}

// Function to parse a single rule into a TreeNode
fn parse_rule(rule: &str) -> Option<TreeNode> {
    if let Ok(port) = rule.parse::<u16>() {
        Some(TreeNode::TCPPort(port))
    } else {
        Some(TreeNode::IPAddress(rule.to_string()))
    }
}

// Function to read and parse the config file, building the expression tree
fn build_tree_from_config(file_path: &Path) -> io::Result<Option<TreeNode>> {
    let file = File::open(file_path)?;
    let mut trees: Vec<TreeNode> = Vec::new();

    for line in io::BufReader::new(file).lines() {
        let line = line?;
        let rules: Vec<_> = line.split_whitespace().map(parse_rule).collect::<Option<Vec<_>>>().unwrap_or_default();

        if let Some(tree) = TreeNode::and_relation(rules) {
            trees.push(tree);
        }
    }

    // Combine all line trees with OR relation
    Ok(trees.into_iter().reduce(|acc, tree| TreeNode::Or(Box::new(acc), Box::new(tree))))
}