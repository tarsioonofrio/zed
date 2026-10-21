use collections::HashMap;
use gpui::{App, Global};
use std::{
    ops::RangeInclusive,
    path::PathBuf,
    time::{Duration, Instant},
};
use text::BufferId;

/// How long a recorded copy stays eligible to be turned back into a notebook
/// cell mention on paste. Long enough for a normal copy/paste, short enough that
/// an unrelated later paste of identical text is unlikely to match.
const COPY_EXPIRY: Duration = Duration::from_secs(60);

/// Identifies a cell within a notebook, using the cell's current 1-based
/// position so labels stay stable while the notebook is being edited.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NotebookCellReference {
    pub notebook_path: PathBuf,
    /// 1-based position of the cell in the notebook.
    pub cell_number: usize,
    pub cell_line_count: u32,
}

struct RecordedCopy {
    text: String,
    reference: NotebookCellReference,
    line_range: RangeInclusive<u32>,
    recorded_at: Instant,
}

/// Maps each notebook cell's buffer to its notebook reference so that
/// selections and copies can be labelled with the notebook path and cell
/// number rather than the cell's synthetic buffer. Also remembers the most
/// recent copy so a paste can restore the cell crease.
#[derive(Default)]
pub struct NotebookCellRegistry {
    cells: HashMap<BufferId, NotebookCellReference>,
    last_copy: Option<RecordedCopy>,
}

impl Global for NotebookCellRegistry {}

impl NotebookCellRegistry {
    pub fn register(&mut self, buffer_id: BufferId, reference: NotebookCellReference) {
        self.cells.insert(buffer_id, reference);
    }

    pub fn unregister(&mut self, buffer_id: BufferId) {
        self.cells.remove(&buffer_id);
    }

    pub fn update_cell_number(&mut self, buffer_id: BufferId, cell_number: usize) {
        if let Some(reference) = self.cells.get_mut(&buffer_id) {
            reference.cell_number = cell_number;
        }
    }

    pub fn lookup(&self, buffer_id: BufferId) -> Option<&NotebookCellReference> {
        self.cells.get(&buffer_id)
    }

    pub fn record_copy(
        &mut self,
        text: String,
        reference: NotebookCellReference,
        line_range: RangeInclusive<u32>,
    ) {
        self.last_copy = Some(RecordedCopy {
            text,
            reference,
            line_range,
            recorded_at: Instant::now(),
        });
    }

    /// Returns the recorded cell for `text` and clears it, so a single copy can
    /// only produce one crease. Expired or non-matching copies are dropped.
    pub fn take_matching_copy(
        &mut self,
        text: &str,
    ) -> Option<(NotebookCellReference, RangeInclusive<u32>)> {
        let recorded = self.last_copy.as_ref()?;
        if recorded.recorded_at.elapsed() > COPY_EXPIRY {
            self.last_copy = None;
            return None;
        }
        if recorded.text != text {
            return None;
        }
        let recorded = self.last_copy.take()?;
        Some((recorded.reference, recorded.line_range))
    }
}

pub fn init(cx: &mut App) {
    cx.set_global(NotebookCellRegistry::default());
}

pub fn registry(cx: &mut App) -> &mut NotebookCellRegistry {
    cx.default_global::<NotebookCellRegistry>()
}

pub fn lookup(cx: &App, buffer_id: BufferId) -> Option<NotebookCellReference> {
    cx.try_global::<NotebookCellRegistry>()?
        .lookup(buffer_id)
        .cloned()
}
