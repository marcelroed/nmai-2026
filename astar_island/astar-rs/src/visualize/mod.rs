use std::io::{self, IsTerminal};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

mod app;
mod cached;
mod render;

use app::App;
use cached::{LoadedRound, list_cached_rounds};
use crate::data::{DEFAULT_DATA_DIR, RoundDetails};

pub struct CliArgs {
    pub round_id: Option<String>,
    pub details_path: Option<PathBuf>,
    pub once: bool,
}

pub fn parse_args() -> Result<CliArgs> {
    let mut round_id = None;
    let mut details_path = None;
    let mut once = false;

    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--round-id" => {
                let value = args.next().context("--round-id requires a value")?;
                round_id = Some(value);
            }
            "--details-path" => {
                let value = args.next().context("--details-path requires a value")?;
                details_path = Some(PathBuf::from(value));
            }
            "--once" => once = true,
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            _ if arg.starts_with("--") => anyhow::bail!("unknown flag: {arg}"),
            _ if round_id.is_none() => round_id = Some(arg),
            _ => anyhow::bail!("unexpected positional argument: {arg}"),
        }
    }

    Ok(CliArgs {
        round_id,
        details_path,
        once,
    })
}

fn print_help() {
    println!("astar-rs visualize");
    println!();
    println!("Usage:");
    println!("  cargo run -- [--round-id <id>] [--details-path <path>] [--once]");
    println!("  cargo run -- [round-id] [--once]");
    println!();
    println!("Options:");
    println!("  --round-id <id>       Load ../data/<id>/details.json and ../data/<id>/query/");
    println!("  --details-path <path> Load a specific details.json and sibling query/ directory");
    println!("  --once                Print a single static frame and exit");
}

pub fn run(args: CliArgs) -> Result<()> {
    let initial_round = if let Some(details_path) = args.details_path {
        LoadedRound::from_details_path(details_path)?
    } else if let Some(round_id) = args.round_id {
        LoadedRound::from_round_id(&round_id)?
    } else {
        LoadedRound::from_round_id("71451d74-be9f-471f-aacd-a41f3b68a9cd")?
    };

    let cached_rounds = list_cached_rounds(Path::new(DEFAULT_DATA_DIR))?;
    let mut app = App::new(initial_round, cached_rounds);

    if args.once || !io::stdout().is_terminal() || !io::stdin().is_terminal() {
        print!("{}", render::render_text_snapshot(&app));
        return Ok(());
    }

    app.run_interactive()
}

pub fn round_details_debug_snapshot(details: &RoundDetails) -> Result<String> {
    Ok(render::render_round_details_debug_snapshot(details))
}
