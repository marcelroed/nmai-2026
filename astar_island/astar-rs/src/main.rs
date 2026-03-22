use anyhow::Result;

use crate::data::RoundDetails;

mod data;
mod visualize;

fn main() -> Result<()> {
    let results = RoundDetails::from_round_id_and_map("71451d74-be9f-471f-aacd-a41f3b68a9cd", 0);
    println!("{:?}", results);
    // let args = tui::parse_args()?;
    // tui::run(args)
    Ok(())
}
