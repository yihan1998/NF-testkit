use clap::{App, Arg};

mod tree;
use tree::TreeNode;

fn main() {
    let matches = App::new("DNS filter")
        .version("0.1")
        .arg(
            Arg::with_name("config")
                .short("c")
                .long("config")
                .takes_value(true),
        )
        .get_matches();
    let config_path = matches.value_of("config");
    let tree = build_tree_from_config(config_path)?;
}
