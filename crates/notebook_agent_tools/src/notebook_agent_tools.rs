//! Agent-facing helpers for reading and editing Jupyter notebook cells.
//!
//! The pure JSON operations preserve the notebook document around the edited
//! cell: metadata, outputs, ids and every key the agent does not touch are left
//! exactly as they were. The live registry lets the native agent tools apply the
//! same operations to an already-open notebook editor instead of rewriting the
//! file underneath it.

use collections::HashMap;
use gpui::{AnyWeakEntity, App, EntityId, Global};
use serde::Serialize;
use serde_json::Value;
use std::{
    path::{Path, PathBuf},
    rc::Rc,
};
use uuid::Uuid;

pub const NOTEBOOK_EXTENSION: &str = "ipynb";

/// The cell kinds that can be read and created through the agent tools.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotebookCellType {
    Code,
    Markdown,
    Raw,
}

impl NotebookCellType {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Code => "code",
            Self::Markdown => "markdown",
            Self::Raw => "raw",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "code" => Some(Self::Code),
            "markdown" => Some(Self::Markdown),
            "raw" => Some(Self::Raw),
            _ => None,
        }
    }
}

/// A 1-based, inclusive range of lines inside a single cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LineRange {
    pub start: u32,
    pub end: u32,
}

impl LineRange {
    pub fn new(start: u32, end: u32) -> Result<Self, String> {
        if start == 0 || end == 0 {
            return Err("Line numbers are 1-based, so they start at 1.".to_string());
        }
        if end < start {
            return Err(format!(
                "The end line ({end}) must be greater than or equal to the start line ({start})."
            ));
        }
        Ok(Self { start, end })
    }
}

/// A single cell as reported to the agent.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotebookCellSnapshot {
    pub cell_number: usize,
    pub id: String,
    pub cell_type: NotebookCellType,
    pub source: String,
    pub output_summary: Option<String>,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct NotebookSnapshot {
    pub cells: Vec<NotebookCellSnapshot>,
}

/// Reads the cells of a parsed notebook without mutating it.
pub fn read_cells(notebook: &Value) -> Result<NotebookSnapshot, String> {
    let cells = notebook_cells(notebook)?;
    let mut snapshot = Vec::with_capacity(cells.len());

    for (index, cell) in cells.iter().enumerate() {
        let cell_type = cell
            .get("cell_type")
            .and_then(Value::as_str)
            .and_then(NotebookCellType::parse)
            .ok_or_else(|| {
                format!(
                    "Cell {} has an unsupported or missing cell_type.",
                    index + 1
                )
            })?;
        snapshot.push(NotebookCellSnapshot {
            cell_number: index + 1,
            id: cell_id(cell),
            cell_type,
            source: cell_source(cell),
            output_summary: output_summary(cell),
        });
    }

    Ok(NotebookSnapshot { cells: snapshot })
}

/// Replaces the source of a cell, or only the requested line range of it.
pub fn edit_cell(
    notebook: &mut Value,
    cell_number: usize,
    new_source: &str,
    line_range: Option<LineRange>,
) -> Result<String, String> {
    let index = cell_index(cell_number, notebook_cell_count(notebook)?)?;
    let cell = cell_at_index_mut(notebook, index)?;
    let source = match line_range {
        Some(line_range) => apply_line_edit(&cell_source(cell), line_range, new_source)?,
        None => new_source.to_string(),
    };
    write_source(cell, &source);
    Ok(format!(
        "Replaced the source of cell {cell_number} ({} lines).",
        source.split('\n').count()
    ))
}

/// Inserts a new cell so that it becomes the cell at `position` (1-based).
pub fn insert_cell(
    notebook: &mut Value,
    position: usize,
    cell_type: NotebookCellType,
    source: &str,
) -> Result<String, String> {
    let cell_count = notebook_cell_count(notebook)?;
    if position == 0 || position > cell_count + 1 {
        return Err(format!(
            "Cannot insert a cell at position {position}: the notebook has {cell_count} cells, so the position must be between 1 and {}.",
            cell_count + 1
        ));
    }

    let cell = new_cell(cell_type, source);
    notebook_cells_mut(notebook)?.insert(position - 1, cell);
    Ok(format!(
        "Inserted a {} cell at position {position}.",
        cell_type.as_str()
    ))
}

/// Deletes the cell at `cell_number` (1-based).
pub fn delete_cell(notebook: &mut Value, cell_number: usize) -> Result<String, String> {
    let index = cell_index(cell_number, notebook_cell_count(notebook)?)?;
    notebook_cells_mut(notebook)?.remove(index);
    Ok(format!("Deleted cell {cell_number}."))
}

/// Moves the cell at `from_cell` so that it ends up at `to_cell` (both 1-based).
pub fn move_cell(notebook: &mut Value, from_cell: usize, to_cell: usize) -> Result<String, String> {
    let cell_count = notebook_cell_count(notebook)?;
    let from_index = cell_index(from_cell, cell_count)?;
    let to_index = cell_index(to_cell, cell_count)?;
    if from_index == to_index {
        return Ok(format!(
            "Cell {from_cell} is already at position {to_cell}."
        ));
    }

    let cell = notebook_cells_mut(notebook)?.remove(from_index);
    notebook_cells_mut(notebook)?.insert(to_index, cell);
    Ok(format!("Moved cell {from_cell} to position {to_cell}."))
}

/// Replaces a 1-based inclusive line range inside `source` with `replacement`.
pub fn apply_line_edit(
    source: &str,
    line_range: LineRange,
    replacement: &str,
) -> Result<String, String> {
    let line_starts = line_starts(source);
    let line_count = line_starts.len();
    if line_range.end as usize > line_count {
        return Err(format!(
            "The cell has {line_count} lines, so line {} does not exist.",
            line_range.end
        ));
    }

    let start_offset = line_starts[line_range.start as usize - 1];
    let end_offset = line_starts
        .get(line_range.end as usize)
        .copied()
        .unwrap_or(source.len());

    let mut edited = String::with_capacity(source.len() + replacement.len());
    edited.push_str(&source[..start_offset]);
    edited.push_str(replacement);
    if !replacement.is_empty()
        && !replacement.ends_with('\n')
        && (end_offset < source.len() || source.ends_with('\n'))
    {
        edited.push('\n');
    }
    edited.push_str(&source[end_offset..]);
    Ok(edited)
}

/// Formats a snapshot for the agent's `notebook_read` output.
pub fn format_read_report(path: &Path, snapshot: &NotebookSnapshot) -> String {
    let mut report = format!(
        "Notebook `{}` ({} cells, 1-based positions)\n",
        path.display(),
        snapshot.cells.len()
    );
    for cell in &snapshot.cells {
        report.push_str(&format!(
            "\n[{}] {} (id: {})\n",
            cell.cell_number,
            cell.cell_type.as_str(),
            cell.id
        ));
        if !cell.source.is_empty() {
            report.push_str("source:\n");
            report.push_str(&cell.source);
            if !cell.source.ends_with('\n') {
                report.push('\n');
            }
        }
        if let Some(output_summary) = &cell.output_summary {
            report.push_str(output_summary);
            report.push('\n');
        }
    }
    report
}

/// Serializes a notebook the way nbformat/Jupyter writes it: keys keep the
/// order in which they appear in the file, every level is indented with a single
/// space and the document ends with a newline. Re-serializing a notebook the
/// agent has not changed therefore reproduces the original bytes, so an edit
/// only rewrites the cell it touched instead of reformatting the whole file.
pub fn to_notebook_json(notebook: &Value) -> Result<String, String> {
    let formatter = serde_json::ser::PrettyFormatter::with_indent(b" ");
    let mut bytes = Vec::new();
    let mut serializer = serde_json::Serializer::with_formatter(&mut bytes, formatter);
    notebook
        .serialize(&mut serializer)
        .map_err(|error| format!("Failed to serialize the notebook: {error}"))?;
    bytes.push(b'\n');
    String::from_utf8(bytes).map_err(|error| format!("Failed to encode the notebook: {error}"))
}

fn notebook_cells(notebook: &Value) -> Result<&Vec<Value>, String> {
    notebook
        .get("cells")
        .and_then(Value::as_array)
        .ok_or_else(|| "The notebook JSON has no `cells` array.".to_string())
}

fn notebook_cells_mut(notebook: &mut Value) -> Result<&mut Vec<Value>, String> {
    notebook
        .get_mut("cells")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| "The notebook JSON has no `cells` array.".to_string())
}

fn notebook_cell_count(notebook: &Value) -> Result<usize, String> {
    Ok(notebook_cells(notebook)?.len())
}

fn cell_index(cell_number: usize, cell_count: usize) -> Result<usize, String> {
    if cell_number == 0 || cell_number > cell_count {
        return Err(format!(
            "Cell positions are 1-based and this notebook has {cell_count} cells, so {cell_number} is not a valid cell number."
        ));
    }
    Ok(cell_number - 1)
}

fn cell_at_index_mut(notebook: &mut Value, index: usize) -> Result<&mut Value, String> {
    notebook_cells_mut(notebook)?
        .get_mut(index)
        .ok_or_else(|| format!("Cell {} does not exist.", index + 1))
}

fn cell_id(cell: &Value) -> String {
    cell.get("id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_string()
}

fn cell_source(cell: &Value) -> String {
    match cell.get("source") {
        Some(Value::String(source)) => source.clone(),
        Some(Value::Array(lines)) => lines
            .iter()
            .filter_map(Value::as_str)
            .collect::<Vec<_>>()
            .concat(),
        _ => String::new(),
    }
}

fn write_source(cell: &mut Value, source: &str) {
    let uses_lines = cell.get("source").is_some_and(Value::is_array);
    if uses_lines {
        cell["source"] = Value::Array(source_lines(source));
    } else {
        cell["source"] = Value::String(source.to_string());
    }
}

fn source_lines(source: &str) -> Vec<Value> {
    source
        .split_inclusive('\n')
        .map(|line| Value::String(line.to_string()))
        .collect()
}

fn new_cell(cell_type: NotebookCellType, source: &str) -> Value {
    let mut cell = serde_json::Map::new();
    cell.insert(
        "cell_type".to_string(),
        Value::String(cell_type.as_str().to_string()),
    );
    cell.insert("id".to_string(), Value::String(Uuid::new_v4().to_string()));
    cell.insert(
        "metadata".to_string(),
        Value::Object(serde_json::Map::new()),
    );
    cell.insert("source".to_string(), Value::Array(source_lines(source)));
    if cell_type == NotebookCellType::Code {
        cell.insert("execution_count".to_string(), Value::Null);
        cell.insert("outputs".to_string(), Value::Array(Vec::new()));
    }
    Value::Object(cell)
}

fn output_summary(cell: &Value) -> Option<String> {
    let outputs = cell.get("outputs").and_then(Value::as_array)?;
    if outputs.is_empty() {
        return None;
    }

    let mut kinds: Vec<&str> = Vec::new();
    for output in outputs {
        let kind = output
            .get("output_type")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        if !kinds.contains(&kind) {
            kinds.push(kind);
        }
    }
    Some(format!("outputs: {} ({})", outputs.len(), kinds.join(", ")))
}

fn line_starts(source: &str) -> Vec<usize> {
    let mut starts = vec![0];
    for (index, byte) in source.bytes().enumerate() {
        if byte == b'\n' {
            starts.push(index + 1);
        }
    }
    starts
}

/// A live edit requested against an open notebook editor.
#[derive(Clone, Debug)]
pub enum NotebookLiveCommand {
    Read,
    EditCell {
        cell_number: usize,
        new_source: String,
        line_range: Option<LineRange>,
    },
    InsertCell {
        position: usize,
        cell_type: NotebookCellType,
        source: String,
    },
    DeleteCell {
        cell_number: usize,
    },
    MoveCell {
        from_cell: usize,
        to_cell: usize,
    },
}

#[derive(Clone, Debug)]
pub struct NotebookLiveResult {
    pub snapshot: NotebookSnapshot,
    pub message: String,
}

pub type LiveHandler =
    Rc<dyn Fn(NotebookLiveCommand, &mut App) -> Result<NotebookLiveResult, String>>;

struct LiveNotebook {
    entity: AnyWeakEntity,
    handler: LiveHandler,
}

/// Maps a notebook's absolute path to the editors currently presenting it, so
/// agent tools can edit live cells instead of rewriting the file behind the UI.
/// Several editors may present the same path (for example two windows); the
/// most recently registered live one answers and the rest keep their place.
#[derive(Default)]
pub struct NotebookLiveRegistry {
    notebooks: HashMap<PathBuf, Vec<LiveNotebook>>,
}

impl Global for NotebookLiveRegistry {}

impl NotebookLiveRegistry {
    pub fn register(&mut self, path: PathBuf, entity: AnyWeakEntity, handler: LiveHandler) {
        let notebooks = self.notebooks.entry(path).or_default();
        let entity_id = entity.entity_id();
        notebooks.retain(|notebook| notebook.entity.entity_id() != entity_id);
        notebooks.push(LiveNotebook { entity, handler });
    }

    pub fn unregister(&mut self, entity_id: EntityId) {
        self.notebooks.retain(|_, notebooks| {
            notebooks.retain(|notebook| notebook.entity.entity_id() != entity_id);
            !notebooks.is_empty()
        });
    }

    pub fn handler(&mut self, path: &Path) -> Option<LiveHandler> {
        let notebooks = self.notebooks.get_mut(path)?;
        notebooks.retain(|notebook| notebook.entity.is_upgradable());
        let handler = notebooks.last().map(|notebook| notebook.handler.clone());
        let is_empty = notebooks.is_empty();
        if is_empty {
            self.notebooks.remove(path);
        }
        handler
    }
}

pub fn registry(cx: &mut App) -> &mut NotebookLiveRegistry {
    cx.default_global::<NotebookLiveRegistry>()
}

pub fn register_live(cx: &mut App, path: PathBuf, entity: AnyWeakEntity, handler: LiveHandler) {
    registry(cx).register(path, entity, handler);
}

pub fn unregister_live(cx: &mut App, entity_id: EntityId) {
    registry(cx).unregister(entity_id);
}

pub fn run_live(
    cx: &mut App,
    path: &Path,
    command: NotebookLiveCommand,
) -> Option<Result<NotebookLiveResult, String>> {
    let handler = registry(cx).handler(path)?;
    Some(handler(command, cx))
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::AppContext as _;
    use serde_json::json;

    fn notebook() -> Value {
        json!({
            "nbformat": 4,
            "nbformat_minor": 5,
            "metadata": { "kernelspec": { "name": "python3" } },
            "cells": [
                {
                    "cell_type": "code",
                    "id": "first",
                    "metadata": { "tags": ["keep"] },
                    "execution_count": 7,
                    "outputs": [{ "output_type": "stream", "name": "stdout", "text": "hi\n" }],
                    "source": ["print('one')\n"]
                },
                {
                    "cell_type": "markdown",
                    "id": "second",
                    "metadata": {},
                    "source": "Some *text*\n"
                },
                {
                    "cell_type": "code",
                    "id": "third",
                    "metadata": {},
                    "execution_count": null,
                    "outputs": [],
                    "source": ["print('three')\n"]
                }
            ]
        })
    }

    #[test]
    fn read_cells_reports_positions_types_sources_and_outputs() {
        let snapshot = read_cells(&notebook()).unwrap();
        assert_eq!(snapshot.cells.len(), 3);
        assert_eq!(snapshot.cells[0].cell_number, 1);
        assert_eq!(snapshot.cells[0].id, "first");
        assert_eq!(snapshot.cells[0].cell_type, NotebookCellType::Code);
        assert_eq!(snapshot.cells[0].source, "print('one')\n");
        assert_eq!(
            snapshot.cells[0].output_summary.as_deref(),
            Some("outputs: 1 (stream)")
        );
        assert_eq!(snapshot.cells[1].cell_type, NotebookCellType::Markdown);
        assert_eq!(snapshot.cells[2].output_summary, None);
    }

    #[test]
    fn edit_cell_preserves_metadata_outputs_and_other_cells() {
        let mut notebook = notebook();
        edit_cell(&mut notebook, 1, "print('changed')\n", None).unwrap();

        let cells = notebook["cells"].as_array().unwrap();
        assert_eq!(cells[0]["id"], "first");
        assert_eq!(cells[0]["metadata"], json!({ "tags": ["keep"] }));
        assert_eq!(cells[0]["execution_count"], 7);
        assert_eq!(cells[0]["outputs"][0]["output_type"], "stream");
        assert_eq!(cells[0]["source"], json!(["print('changed')\n"]));
        assert_eq!(cells[1]["id"], "second");
        assert_eq!(cells[2]["id"], "third");
        assert_eq!(notebook["metadata"]["kernelspec"]["name"], "python3");
    }

    #[test]
    fn edit_cell_can_replace_a_line_range() {
        let mut notebook = notebook();
        edit_cell(
            &mut notebook,
            2,
            "replacement",
            Some(LineRange::new(1, 1).unwrap()),
        )
        .unwrap();
        assert_eq!(notebook["cells"][1]["source"], "replacement\n");
    }

    #[test]
    fn insert_cell_places_cell_at_requested_position() {
        let mut notebook = notebook();
        insert_cell(&mut notebook, 2, NotebookCellType::Code, "print('new')\n").unwrap();

        let cells = notebook["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 4);
        assert_eq!(cells[1]["cell_type"], "code");
        assert_eq!(cells[1]["source"], json!(["print('new')\n"]));
        assert!(cells[1]["id"].as_str().is_some_and(|id| !id.is_empty()));
        assert_eq!(cells[2]["id"], "second");
    }

    #[test]
    fn insert_cell_at_end_is_allowed() {
        let mut notebook = notebook();
        insert_cell(&mut notebook, 4, NotebookCellType::Markdown, "last").unwrap();
        assert_eq!(notebook["cells"][3]["cell_type"], "markdown");
    }

    #[test]
    fn delete_cell_removes_only_the_requested_cell() {
        let mut notebook = notebook();
        delete_cell(&mut notebook, 2).unwrap();

        let cells = notebook["cells"].as_array().unwrap();
        assert_eq!(cells.len(), 2);
        assert_eq!(cells[0]["id"], "first");
        assert_eq!(cells[1]["id"], "third");
    }

    #[test]
    fn move_cell_reorders_without_losing_data() {
        let mut notebook = notebook();
        move_cell(&mut notebook, 1, 3).unwrap();

        let cells = notebook["cells"].as_array().unwrap();
        assert_eq!(cells[0]["id"], "second");
        assert_eq!(cells[1]["id"], "third");
        assert_eq!(cells[2]["id"], "first");
        assert_eq!(cells[2]["outputs"][0]["output_type"], "stream");
    }

    #[test]
    fn invalid_positions_are_rejected() {
        let mut notebook = notebook();
        assert!(edit_cell(&mut notebook, 0, "x", None).is_err());
        assert!(edit_cell(&mut notebook, 4, "x", None).is_err());
        assert!(delete_cell(&mut notebook, 4).is_err());
        assert!(move_cell(&mut notebook, 1, 4).is_err());
        assert!(insert_cell(&mut notebook, 5, NotebookCellType::Code, "x").is_err());
        assert!(LineRange::new(0, 1).is_err());
        assert!(LineRange::new(2, 1).is_err());
    }

    #[test]
    fn out_of_range_line_edit_is_rejected() {
        let mut notebook = notebook();
        assert!(edit_cell(&mut notebook, 2, "x", Some(LineRange::new(1, 5).unwrap())).is_err());
    }

    #[test]
    fn line_edit_replaces_multiple_lines() {
        let source = "one\ntwo\nthree\n";
        let edited = apply_line_edit(source, LineRange::new(2, 3).unwrap(), "changed").unwrap();
        assert_eq!(edited, "one\nchanged\n");
    }

    #[test]
    fn line_edit_keeps_a_missing_trailing_newline() {
        let source = "one\ntwo";
        let edited = apply_line_edit(source, LineRange::new(2, 2).unwrap(), "changed").unwrap();
        assert_eq!(edited, "one\nchanged");
    }

    const NBFORMAT_NOTEBOOK: &str = r#"{
 "cells": [
  {
   "cell_type": "code",
   "id": "first",
   "metadata": {},
   "outputs": [],
   "execution_count": null,
   "source": [
    "print('one')\n"
   ]
  },
  {
   "cell_type": "code",
   "id": "second",
   "metadata": {},
   "outputs": [],
   "execution_count": null,
   "source": [
    "print('two')\n"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}
"#;

    #[test]
    fn serializing_an_unchanged_notebook_reproduces_the_original_bytes() {
        let notebook: Value = serde_json::from_str(NBFORMAT_NOTEBOOK).unwrap();
        assert_eq!(to_notebook_json(&notebook).unwrap(), NBFORMAT_NOTEBOOK);
    }

    #[test]
    fn editing_a_cell_only_rewrites_that_cell_in_the_serialized_output() {
        let mut notebook: Value = serde_json::from_str(NBFORMAT_NOTEBOOK).unwrap();
        edit_cell(&mut notebook, 2, "print('changed')\n", None).unwrap();
        let edited = to_notebook_json(&notebook).unwrap();
        let expected = NBFORMAT_NOTEBOOK.replace("print('two')\\n", "print('changed')\\n");
        assert_eq!(edited, expected);
    }

    fn live_handler(message: &'static str) -> LiveHandler {
        Rc::new(move |_command, _cx| {
            Ok(NotebookLiveResult {
                snapshot: NotebookSnapshot::default(),
                message: message.to_string(),
            })
        })
    }

    #[gpui::test]
    async fn unregistering_the_last_handler_falls_back_to_the_previous_one(
        cx: &mut gpui::TestAppContext,
    ) {
        let path = PathBuf::from("/notebooks/agent.ipynb");

        let first = cx.new(|_| ());
        let first_entity: AnyWeakEntity = first.downgrade().into();
        let second = cx.new(|_| ());
        let second_id = second.entity_id();
        let second_entity: AnyWeakEntity = second.downgrade().into();

        cx.update(|cx| {
            register_live(cx, path.clone(), first_entity, live_handler("first"));
            register_live(cx, path.clone(), second_entity, live_handler("second"));
        });

        let handler = cx
            .update(|cx| registry(cx).handler(&path))
            .expect("the last registered handler must answer");
        assert_eq!(
            cx.update(|cx| handler(NotebookLiveCommand::Read, cx)
                .expect("handler runs")
                .message),
            "second"
        );

        cx.update(|cx| unregister_live(cx, second_id));

        let handler = cx
            .update(|cx| registry(cx).handler(&path))
            .expect("the first handler must take over once the last is unregistered");
        assert_eq!(
            cx.update(|cx| handler(NotebookLiveCommand::Read, cx)
                .expect("handler runs")
                .message),
            "first"
        );

        drop(first);
        assert!(
            cx.update(|cx| registry(cx).handler(&path)).is_none(),
            "no handler should answer once every editor is gone"
        );
    }
}
