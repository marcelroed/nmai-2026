use std::fmt::Write as _;

use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span, Text};
use ratatui::widgets::{Block, Borders, Paragraph, Tabs, Wrap};

use crate::data::{RoundDetails, TerrainType};
use crate::visualize::app::App;
use crate::visualize::cached::{LoadedRound, MapView};

pub fn draw(frame: &mut ratatui::Frame<'_>, app: &App) {
    let layout = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(2),
            Constraint::Length(3),
            Constraint::Min(10),
            Constraint::Length(2),
            Constraint::Length(1),
        ])
        .split(frame.area());

    let header = Paragraph::new(render_header(&app.loaded_round));
    frame.render_widget(header, layout[0]);

    let tabs = Tabs::new(render_map_tabs(app))
        .select(app.selected_map)
        .highlight_style(
            Style::default()
                .fg(Color::Black)
                .bg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .style(Style::default().fg(Color::White))
        .divider(" ");
    frame.render_widget(tabs, layout[1]);

    let body = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Min(50), Constraint::Length(34)])
        .split(layout[2]);

    let grid = Paragraph::new(render_grid_text(app)).wrap(Wrap { trim: false });
    frame.render_widget(grid, body[0]);

    let sidebar = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Length(12), Constraint::Min(10)])
        .split(body[1]);

    frame.render_widget(
        Paragraph::new(render_legend()).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Legend")
                .border_style(Style::default().fg(Color::Yellow)),
        ),
        sidebar[0],
    );
    frame.render_widget(
        Paragraph::new(render_summary(app)).block(
            Block::default()
                .borders(Borders::ALL)
                .title("Round")
                .border_style(Style::default().fg(Color::LightRed)),
        ),
        sidebar[1],
    );

    let footer = Paragraph::new(Line::from(vec![Span::styled(
        "h/l map  j/k round  r refresh  q quit",
        Style::default().fg(Color::DarkGray),
    )]));
    frame.render_widget(footer, layout[3]);

    let message = Paragraph::new(app.message.as_str())
        .style(Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC));
    frame.render_widget(message, layout[4]);
}

pub fn render_text_snapshot(app: &App) -> String {
    let mut out = String::new();
    let header = render_header(&app.loaded_round);
    let _ = writeln!(&mut out, "{}", line_to_plain(&header));
    for tab in render_map_tabs(app) {
        let _ = write!(&mut out, "{}", line_to_plain(&tab));
    }
    out.push('\n');
    out.push_str(&text_to_plain(&render_grid_text(app)));
    out.push('\n');
    out.push_str("Legend\n");
    out.push_str(&text_to_plain(&render_legend()));
    out.push('\n');
    out.push_str("Summary\n");
    out.push_str(&text_to_plain(&render_summary(app)));
    out.push('\n');
    out.push_str("h/l map  j/k round  r refresh  q quit\n");
    if !app.message.is_empty() {
        let _ = writeln!(&mut out, "{}", app.message);
    }
    out
}

pub fn render_debug_snapshot(loaded_round: &LoadedRound) -> String {
    let mut out = String::new();
    let header = render_header(loaded_round);
    let _ = writeln!(&mut out, "{}", line_to_plain(&header));
    let _ = writeln!(
        &mut out,
        "Event date: {}  Window: {}m  Maps: {}  Weight: {:.2}",
        loaded_round.details.event_date,
        loaded_round.details.prediction_window_minutes,
        loaded_round.details.map_count,
        loaded_round.details.round_weight
    );
    let _ = writeln!(
        &mut out,
        "Started: {}  Closes: {}",
        loaded_round.details.started_at.format("%Y-%m-%d %H:%M:%S%:z"),
        loaded_round.details.closes_at.format("%Y-%m-%d %H:%M:%S%:z")
    );
    out.push('\n');
    out.push_str("Legend\n");
    out.push_str(&text_to_plain(&render_legend()));

    for map in &loaded_round.map_views {
        let _ = writeln!(&mut out, "\nMap {}\n", map.map_idx);
        out.push_str(&text_to_plain(&render_grid(map)));
        out.push('\n');
        out.push_str("Summary\n");
        out.push_str(&text_to_plain(&render_map_summary(loaded_round, map)));
    }

    out
}

pub fn render_round_details_debug_snapshot(details: &RoundDetails) -> String {
    let mut out = String::new();
    let _ = writeln!(
        &mut out,
        "Round #{}  ID: {}  Map {}",
        details.round_number, details.id, details.map_index
    );
    out.push('\n');
    out.push_str("Legend\n");
    out.push_str(&text_to_plain(&render_legend()));
    out.push('\n');
    let grid_text = render_grid_from_grid(&details.initial_state.grid);
    out.push_str(&text_to_plain(&grid_text));
    out.push('\n');

    let settlement_count = details.initial_state.settlements.len();
    let alive_settlement_count = details
        .initial_state
        .settlements
        .iter()
        .filter(|settlement| settlement.alive)
        .count();
    let port_count = details
        .initial_state
        .settlements
        .iter()
        .filter(|settlement| settlement.has_port)
        .count();

    let _ = writeln!(&mut out, "Summary");
    let _ = writeln!(&mut out, "Settlements: {}", settlement_count);
    let _ = writeln!(&mut out, "Alive settlements: {}", alive_settlement_count);
    let _ = writeln!(&mut out, "Ports: {}", port_count);

    out
}

fn render_header(loaded_round: &LoadedRound) -> Line<'static> {
    Line::from(vec![
        Span::styled(
            format!(
                "Round #{} ({})",
                loaded_round.details.round_number, loaded_round.details.status
            ),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        ),
        Span::raw("  "),
        Span::styled(
            format!("ID: {}", loaded_round.details.id),
            Style::default().fg(Color::Gray),
        ),
    ])
}

fn render_map_tabs(app: &App) -> Vec<Line<'static>> {
    app.loaded_round
        .map_views
        .iter()
        .map(|map| Line::from(format!(" Map {} ", map.map_idx)))
        .collect()
}

fn render_grid_text(app: &App) -> Text<'static> {
    render_grid(app.current_map())
}

fn render_grid(map: &MapView) -> Text<'static> {
    render_grid_from_grid(&map.stitched_grid)
}

fn render_grid_from_grid(grid: &[Vec<TerrainType>]) -> Text<'static> {
    let row_label_width = grid.len().saturating_sub(1).to_string().len().max(1);

    let mut lines = Vec::with_capacity(grid.len() + 1);
    let mut header_spans = vec![Span::raw(format!("{:width$}  ", "", width = row_label_width))];
    for col in 0..grid[0].len() {
        if col % 5 == 0 {
            header_spans.push(Span::styled(
                format!("{col:<5}"),
                Style::default().fg(Color::DarkGray),
            ));
        }
    }
    lines.push(Line::from(header_spans));

    for (row_idx, row) in grid.iter().enumerate() {
        let mut spans = vec![Span::styled(
            format!("{row_idx:>width$} ", width = row_label_width),
            Style::default().fg(Color::DarkGray),
        )];

        for cell in row {
            spans.push(Span::styled(terrain_glyph(*cell).to_owned(), terrain_style(*cell)));
        }

        spans.push(Span::styled(
            format!(" {row_idx}"),
            Style::default().fg(Color::DarkGray),
        ));
        lines.push(Line::from(spans));
    }

    Text::from(lines)
}

fn render_legend() -> Text<'static> {
    let mut lines = Vec::new();
    for terrain in [
        TerrainType::Ocean,
        TerrainType::Plains,
        TerrainType::Settlement,
        TerrainType::Port,
        TerrainType::Ruin,
        TerrainType::Forest,
        TerrainType::Mountain,
    ] {
        lines.push(Line::from(vec![
            Span::styled(terrain_glyph(terrain).to_owned(), terrain_style(terrain)),
            Span::raw(" "),
            Span::raw(terrain_name(terrain)),
        ]));
    }
    lines.push(Line::from(vec![
        Span::styled("?", Style::default().fg(Color::DarkGray)),
        Span::raw(" Unknown"),
    ]));
    Text::from(lines)
}

fn render_summary(app: &App) -> Text<'static> {
    render_map_summary(&app.loaded_round, app.current_map())
}

fn render_map_summary(loaded_round: &LoadedRound, map: &MapView) -> Text<'static> {
    let (observed, total) = observed_counts(map);
    let changed_tiles = changed_tile_count(&map.base_grid, &map.stitched_grid);
    let coverage = if total == 0 {
        0.0
    } else {
        100.0 * observed as f64 / total as f64
    };
    let query_budget = loaded_round
        .queries
        .iter()
        .find_map(|query| query.result.queries_max)
        .unwrap_or(0);
    let query_usage = loaded_round
        .queries
        .iter()
        .filter_map(|query| query.result.queries_used)
        .max()
        .unwrap_or(0);

    Text::from(vec![
        Line::from(vec![Span::styled(
            format!("Coverage: {coverage:.0}%"),
            Style::default()
                .fg(Color::Yellow)
                .add_modifier(Modifier::BOLD),
        )]),
        Line::raw(format!("Rows: {}", loaded_round.details.n_rows)),
        Line::raw(format!("Cols: {}", loaded_round.details.n_cols)),
        Line::raw(format!("Queries: {}", map.query_count)),
        Line::raw(format!("Query files: {}", loaded_round.queries.len())),
        Line::raw(format!("Query used: {}", query_usage)),
        Line::raw(format!("Query max: {}", query_budget)),
        Line::raw(format!("Settlement: {}", map.settlement_count)),
        Line::raw(format!("Alive settlements: {}", map.alive_settlement_count)),
        Line::raw(format!("Port: {}", map.port_count)),
        Line::raw(format!("Forest: {}", map.counts.get(TerrainType::Forest))),
        Line::raw(format!("Mountain: {}", map.counts.get(TerrainType::Mountain))),
        Line::raw(format!("Ocean: {}", map.counts.get(TerrainType::Ocean))),
        Line::raw(format!("Changed tiles: {}", changed_tiles)),
        Line::raw(format!(
            "Window: {}m",
            loaded_round.details.prediction_window_minutes
        )),
        Line::raw(format!("Maps: {}", loaded_round.details.map_count)),
        Line::raw(format!("Weight: {:.2}", loaded_round.details.round_weight)),
        Line::raw(format!(
            "Started: {}",
            loaded_round.details.started_at.format("%H:%M:%S%:z")
        )),
        Line::raw(format!(
            "Closes: {}",
            loaded_round.details.closes_at.format("%H:%M:%S%:z")
        )),
    ])
}

fn text_to_plain(text: &Text<'_>) -> String {
    let mut out = String::new();
    for line in &text.lines {
        let _ = writeln!(&mut out, "{}", line_to_plain(line));
    }
    out
}

fn line_to_plain(line: &Line<'_>) -> String {
    let mut out = String::new();
    for span in &line.spans {
        out.push_str(span.content.as_ref());
    }
    out
}

fn terrain_name(terrain: TerrainType) -> &'static str {
    match terrain {
        TerrainType::Empty => "Empty",
        TerrainType::Settlement => "Settlement",
        TerrainType::Port => "Port",
        TerrainType::Ruin => "Ruin",
        TerrainType::Forest => "Forest",
        TerrainType::Mountain => "Mountain",
        TerrainType::Ocean => "Ocean",
        TerrainType::Plains => "Plains",
    }
}

fn terrain_glyph(terrain: TerrainType) -> &'static str {
    match terrain {
        TerrainType::Empty => ".",
        TerrainType::Settlement => "S",
        TerrainType::Port => "P",
        TerrainType::Ruin => "R",
        TerrainType::Forest => "f",
        TerrainType::Mountain => "^",
        TerrainType::Ocean => "~",
        TerrainType::Plains => ".",
    }
}

fn terrain_style(terrain: TerrainType) -> Style {
    match terrain {
        TerrainType::Empty => Style::default().fg(Color::Gray),
        TerrainType::Settlement => Style::default().fg(Color::Yellow),
        TerrainType::Port => Style::default().fg(Color::Cyan),
        TerrainType::Ruin => Style::default().fg(Color::Red),
        TerrainType::Forest => Style::default().fg(Color::Green),
        TerrainType::Mountain => Style::default().fg(Color::White),
        TerrainType::Ocean => Style::default().fg(Color::Blue),
        TerrainType::Plains => Style::default().fg(Color::Gray),
    }
}

fn observed_counts(map: &MapView) -> (usize, usize) {
    let observed = map
        .observed_mask
        .iter()
        .flat_map(|row| row.iter())
        .filter(|cell| **cell)
        .count();
    let total = map
        .observed_mask
        .first()
        .map(|row| map.observed_mask.len() * row.len())
        .unwrap_or(0);
    (observed, total)
}

fn changed_tile_count(base_grid: &[Vec<TerrainType>], stitched_grid: &[Vec<TerrainType>]) -> usize {
    let mut changed = 0usize;
    for (row_idx, row) in stitched_grid.iter().enumerate() {
        for (col_idx, cell) in row.iter().enumerate() {
            if base_grid[row_idx][col_idx] != *cell {
                changed += 1;
            }
        }
    }
    changed
}
