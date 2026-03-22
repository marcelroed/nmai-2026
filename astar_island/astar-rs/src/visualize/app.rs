use std::io;
use std::time::Duration;

use anyhow::Result;
use crossterm::event::{self, Event, KeyCode, KeyEventKind};
use crossterm::execute;
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};
use ratatui::Terminal;
use ratatui::backend::CrosstermBackend;

use crate::visualize::cached::{CachedRoundSummary, LoadedRound, MapView};
use crate::visualize::render;

pub struct App {
    pub loaded_round: LoadedRound,
    pub cached_rounds: Vec<CachedRoundSummary>,
    pub selected_map: usize,
    pub message: String,
}

impl App {
    pub fn new(loaded_round: LoadedRound, cached_rounds: Vec<CachedRoundSummary>) -> Self {
        Self {
            loaded_round,
            cached_rounds,
            selected_map: 0,
            message: String::new(),
        }
    }

    pub fn run_interactive(&mut self) -> Result<()> {
        enable_raw_mode()?;
        let mut stdout = io::stdout();
        execute!(stdout, EnterAlternateScreen)?;

        let backend = CrosstermBackend::new(stdout);
        let mut terminal = Terminal::new(backend)?;
        let result = self.event_loop(&mut terminal);

        disable_raw_mode()?;
        execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
        terminal.show_cursor()?;

        result
    }

    pub fn current_map(&self) -> &MapView {
        &self.loaded_round.map_views[self.selected_map]
    }

    pub fn set_round_from_loaded(
        &mut self,
        loaded_round: LoadedRound,
        message: impl Into<String>,
    ) {
        self.loaded_round = loaded_round;
        self.selected_map = self
            .selected_map
            .min(self.loaded_round.map_views.len().saturating_sub(1));
        self.message = message.into();
    }

    pub fn move_map(&mut self, delta: isize) {
        let len = self.loaded_round.map_views.len();
        if len == 0 {
            return;
        }
        let next = (self.selected_map as isize + delta).clamp(0, len.saturating_sub(1) as isize)
            as usize;
        self.selected_map = next;
        self.message = format!("selected map {}", next);
    }

    pub fn move_round(&mut self, delta: isize) -> Result<()> {
        if self.cached_rounds.is_empty() {
            self.message = "no cached rounds found".to_owned();
            return Ok(());
        }

        let current_id = &self.loaded_round.details.id;
        let current_idx = self
            .cached_rounds
            .iter()
            .position(|round| &round.id == current_id)
            .unwrap_or(0);
        let next_idx = (current_idx as isize + delta)
            .clamp(0, self.cached_rounds.len().saturating_sub(1) as isize)
            as usize;

        if next_idx == current_idx {
            self.message = "no more cached rounds in that direction".to_owned();
            return Ok(());
        }

        let next_round_id = self.cached_rounds[next_idx].id.clone();
        let loaded_round = LoadedRound::from_round_id(&next_round_id)?;
        self.set_round_from_loaded(loaded_round, format!("loaded round {next_round_id}"));
        Ok(())
    }

    fn event_loop(
        &mut self,
        terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    ) -> Result<()> {
        loop {
            terminal.draw(|frame| render::draw(frame, self))?;

            if !event::poll(Duration::from_millis(200))? {
                continue;
            }

            let Event::Key(key) = event::read()? else {
                continue;
            };
            if key.kind != KeyEventKind::Press {
                continue;
            }

            if !self.handle_key(key.code)? {
                break;
            }
        }

        Ok(())
    }

    fn handle_key(&mut self, key: KeyCode) -> Result<bool> {
        match key {
            KeyCode::Char('q') => return Ok(false),
            KeyCode::Char('h') => self.move_map(-1),
            KeyCode::Char('l') => self.move_map(1),
            KeyCode::Char('j') => self.move_round(-1)?,
            KeyCode::Char('k') => self.move_round(1)?,
            KeyCode::Char('r') => {
                let loaded_round = LoadedRound::from_paths(
                    self.loaded_round.details_path.clone(),
                    self.loaded_round.query_dir.clone(),
                )?;
                self.set_round_from_loaded(loaded_round, "reloaded query data");
            }
            _ => {}
        }

        Ok(true)
    }
}
