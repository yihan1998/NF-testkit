use std::fs::File;
use std::io::{self, BufRead};
use std::path::Path;

// Define the tree node structure with the abstracted MatchType
pub enum TreeNode {
    And(Box<TreeNode>, Box<TreeNode>),
    Or(Box<TreeNode>, Box<TreeNode>),
    Not(Box<TreeNode>),
    // SourceIPMatch(String),
    // PayloadMatch(Vec<u8>),
    // PortRangeMatch(u16, u16), // Stores (start_port, end_port)
}

impl TreeNode {
    fn matches(&self, context: &rule::PacketContext) -> bool {
        match self {
            // TreeNode::IPAddress(pattern) => pattern == ip_address,
            // TreeNode::TCPPort(port) => *port == tcp_port,
            TreeNode::And(left, right) => left.matches(context) && right.matches(context),
            TreeNode::Or(left, right) => left.matches(context) || right.matches(context),
            TreeNode::Not(child) => !self.matches(child),
        }
    }

    // Function to build an AND relation tree from a set of rules
    fn and_relation(rules: Vec<TreeNode>) -> Option<TreeNode> {
        rules.into_iter().reduce(|acc, rule| TreeNode::And(Box::new(acc), Box::new(rule)))
    }

    fn not_relation(rules: Vec<TreeNode>) -> Option<TreeNode> {
        rules.into_iter().reduce(|acc, rule| TreeNode::Not(Box::new(acc), Box::new(rule)))
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
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.is_empty() {
            continue;
        }

        let action = parts[0];
        let mut rules = Vec::new();

        for &rule in &parts[1..] {
            if rule.starts_with("!") {
                let rule = rule.trim_start_matches("NOT");
                rules.push(TreeNode::Not(Box::new(TreeNode::MatchRule(rule.to_string()))));
            } else {
                rules.push(TreeNode::MatchRule(rule.to_string()));
            }
        }

        // Combine rules with AND relation
        let and_node = TreeNode::And(rules);

        // Combine action
        let action_node = TreeNode::Action(action.to_string());

        // Combine action and rule with AND
        trees.push(TreeNode::And(vec![action_node, and_node]));
    }

    // Combine all line trees with OR relation
    Ok(trees.into_iter().reduce(|acc, tree| TreeNode::Or(Box::new(acc), Box::new(tree))))
}