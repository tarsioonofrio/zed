//! Language-server features for notebook code cells.
//!
//! Each notebook normally has no on-disk artefact that a language server can
//! analyse, because every code cell lives in its own anonymous local buffer.
//! This crate mirrors all code cells of a notebook into a single virtual file
//! that is opened as a first-class project buffer and registered with the
//! notebook's language servers. Completions, hover and diagnostics requested
//! from a cell are translated into the virtual document, serviced there, and
//! translated back into the cell's own coordinates.

use std::{
    cell::RefCell,
    ops::Range,
    path::{Path, PathBuf},
    rc::Rc,
    sync::Arc,
    time::Duration,
};

use anyhow::{Context as _, Result};
use clock::Global;
use collections::{HashMap, HashSet};
use editor::{
    CodeActionProvider, CompletionContext, CompletionProvider, Editor, GotoDefinitionKind,
    RenameTarget, SemanticsProvider,
};
use fs::Fs;
use futures::{StreamExt as _, future::Shared};
use gpui::{
    App, AppContext as _, AsyncApp, BackgroundExecutor, Context, Entity, FutureExt, Subscription, Task, TaskExt, WeakEntity,
    Window,
};
use language::{
    Anchor, Buffer, BufferEvent, BufferRow, BufferSnapshot, CharScopeContext, CodeLabel, Diagnostic,
    DiagnosticEntry, DiagnosticSet, Language, LanguageName, ToOffset,
};
use lsp::LanguageServerId;
use nbformat::v4::CellId;
use project::{
    CodeAction, Completion, CompletionDisplayOptions, CompletionResponse, CompletionSource,
    DocumentHighlight, Hover, HoverBlock, HoverBlockKind, InlayHint, InvalidationStrategy,
    Location, LocationLink, PrepareRenameResponse, Project, ProjectPath, ProjectTransaction,
    lsp_store::{
        BufferSemanticTokens, CacheInlayHints, FormatTrigger, LspFormatTarget, LspStoreEvent,
        OpenLspBufferHandle,
    },
};
use util::{ResultExt, rel_path::RelPath};

/// How long to wait after a cell edit before rewriting the virtual document,
/// so that a burst of keystrokes results in a single document change.
const REWRITE_DEBOUNCE: Duration = Duration::from_millis(120);

/// How long to wait for the kernel to answer a completion or inspection
/// request before falling back to the language server's results alone. The
/// kernel answers from live state and is usually much faster than this.
const KERNEL_REQUEST_TIMEOUT: Duration = Duration::from_millis(300);

/// A completion suggested by the running kernel for one cell.
///
/// `cursor_start`/`cursor_end` are cursor positions within the cell's code
/// measured in Unicode code points, exactly as the Jupyter protocol defines
/// them, and delimit the text the completion replaces.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelCompletion {
    pub matches: Vec<String>,
    pub cursor_start: usize,
    pub cursor_end: usize,
}

/// The markup kind of an inspection returned by the kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KernelInspectionKind {
    PlainText,
    Markdown,
}

/// Documentation for the symbol under the cursor, as returned by the kernel.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct KernelInspection {
    pub kind: KernelInspectionKind,
    pub text: String,
}

/// The notebook's connection to its running kernel, used to complement the
/// language server with live-state completions and inspection.
///
/// The trait lives here so that [`NotebookLsp`] can drive it while `repl`
/// provides the implementation over the actual kernel; tests substitute a
/// fake. Implementations must return `Ok(None)` rather than block when the
/// kernel is busy or unavailable.
pub trait KernelCompleter: 'static {
    fn complete(&self, code: String, cursor_pos: usize) -> Task<Result<Option<KernelCompletion>>>;
    fn inspect(&self, code: String, cursor_pos: usize) -> Task<Result<Option<KernelInspection>>>;
}

/// The rows of a single code cell within the virtual notebook document.
///
/// `start_offset..end_offset` is the byte range of the cell's text inside the
/// document and is the source of truth for position translation; the row fields
/// are informational.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CellSpan {
    pub cell_id: CellId,
    pub start_row: u32,
    pub row_count: u32,
    pub start_offset: usize,
    pub end_offset: usize,
}

/// Builds the virtual document text from the ordered code cells and records
/// where each cell lives inside it. Each cell's text is neutralized for IPython
/// magics (see [`neutralize_magics`]), but the rewrite preserves byte length, so
/// a cell-local byte offset still maps to a document offset by addition.
pub fn build_document(cells: &[(CellId, String)]) -> (String, Vec<CellSpan>) {
    let mut text = String::new();
    let mut spans = Vec::with_capacity(cells.len());
    let mut row: u32 = 0;

    for (cell_id, cell_text) in cells {
        let start_row = row;
        let start_offset = text.len();
        let neutralized = neutralize_magics(cell_text);
        text.push_str(&neutralized);
        let end_offset = text.len();
        let row_count = neutralized.bytes().filter(|byte| *byte == b'\n').count() as u32 + 1;
        row += row_count;

        spans.push(CellSpan {
            cell_id: cell_id.clone(),
            start_row,
            row_count,
            start_offset,
            end_offset,
        });

        // Always leave one blank line between cells so that adjacent cells can
        // never merge into a single statement, without disturbing the verbatim
        // cell text that precedes the separator.
        if cell_text.ends_with('\n') {
            text.push('\n');
            row += 1;
        } else {
            text.push_str("\n\n");
            row += 2;
        }
    }

    (text, spans)
}

/// Rewrites IPython magics and shell escapes into syntactically valid Python so
/// the language server does not report false syntax errors for syntax it cannot
/// parse.
///
/// The rewrite is byte-length preserving: a magic or shell line is replaced by
/// a comment of the same length, a trailing `?`/`??` by spaces, and a cell that
/// opens with `%%` becomes a block of comments. Keeping the byte length (and
/// every newline) identical is what lets a cell-local byte offset still map to
/// a document byte offset by addition, so no separate position mapping is
/// needed. Every replacement is ASCII, so any cell offset remains a valid UTF-8
/// boundary in the document.
fn neutralize_magics(cell_text: &str) -> String {
    if !cell_text.contains(['%', '!', '?']) {
        return cell_text.to_string();
    }

    let is_cell_magic = cell_text
        .lines()
        .next()
        .is_some_and(|line| line.trim_start().starts_with("%%"));

    let mut result = String::with_capacity(cell_text.len());
    for (index, line) in cell_text.split('\n').enumerate() {
        if index > 0 {
            result.push('\n');
        }
        if is_cell_magic {
            result.push_str(&comment_line(line));
        } else if is_magic_or_shell_line(line) {
            result.push_str(&comment_line(line));
        } else {
            result.push_str(&strip_trailing_help(line));
        }
    }
    result
}

/// True for a line that IPython would treat as a line magic (`%time ...`) or a
/// shell escape (`!ls`).
fn is_magic_or_shell_line(line: &str) -> bool {
    let trimmed = line.trim_start();
    match trimmed.as_bytes().first() {
        Some(b'%') => !trimmed.starts_with("%%"),
        Some(b'!') => true,
        _ => false,
    }
}

/// Turns a whole line into a comment of the same byte length, so line and
/// column counts are preserved.
fn comment_line(line: &str) -> String {
    if line.is_empty() {
        return String::new();
    }
    let mut comment = String::with_capacity(line.len());
    comment.push('#');
    comment.push_str(&" ".repeat(line.len() - 1));
    comment
}

/// Replaces a trailing run of `?` characters (IPython's `obj?` / `obj??`) with
/// spaces, preserving the line's length.
fn strip_trailing_help(line: &str) -> String {
    let without = line.trim_end_matches('?');
    if without.len() == line.len() {
        return line.to_string();
    }
    let mut result = String::with_capacity(line.len());
    result.push_str(without);
    result.push_str(&" ".repeat(line.len() - without.len()));
    result
}

/// Maps a byte offset within a cell to the corresponding offset in the virtual
/// document, returning `None` when the offset lies outside the cell.
pub fn document_offset_for_cell(span: &CellSpan, cell_offset: usize) -> Option<usize> {
    if cell_offset > span.end_offset - span.start_offset {
        return None;
    }
    Some(span.start_offset + cell_offset)
}

/// Maps a byte offset within the virtual document to the corresponding offset
/// within its cell, returning `None` when the offset lies outside the cell
/// (for example on a separator line).
pub fn cell_offset_for_document(span: &CellSpan, document_offset: usize) -> Option<usize> {
    if document_offset < span.start_offset || document_offset > span.end_offset {
        return None;
    }
    Some(document_offset - span.start_offset)
}

/// Finds the cell that owns the given document offset, together with the offset
/// relative to the start of that cell.
pub fn cell_at_document_offset(
    spans: &[CellSpan],
    document_offset: usize,
) -> Option<(&CellSpan, usize)> {
    spans.iter().find_map(|span| {
        cell_offset_for_document(span, document_offset).map(|cell_offset| (span, cell_offset))
    })
}

/// A code cell participating in the virtual document.
pub struct NotebookCell {
    pub id: CellId,
    pub editor: Entity<Editor>,
    pub buffer: Entity<Buffer>,
}

struct CellState {
    id: CellId,
    editor: Entity<Editor>,
    buffer: Entity<Buffer>,
    span: Option<CellSpan>,
    providers_installed: bool,
    _subscription: Subscription,
}

impl CellState {
    fn span(&self) -> Option<&CellSpan> {
        self.span.as_ref()
    }
}

struct DocumentState {
    path: ProjectPath,
    buffer: Entity<Buffer>,
    _lsp_handle: OpenLspBufferHandle,
    text: String,
}

/// Owns the virtual document for one notebook and installs the per-cell
/// completion and semantics providers.
pub struct NotebookLsp {
    project: Entity<Project>,
    notebook_path: PathBuf,
    language: Option<Arc<Language>>,
    document: Option<DocumentState>,
    cells: Vec<CellState>,
    python_interpreter: Option<PathBuf>,
    kernel_completer: Option<Rc<dyn KernelCompleter>>,
    _rewrite_task: Option<Task<()>>,
    _setup_task: Option<Task<()>>,
    _toolchain_task: Option<Task<()>>,
    _lsp_subscription: Subscription,
    owns_document: bool,
}

const REMOTE_LINK_TIMEOUT: Duration = Duration::from_secs(30);

/// Project-relative directory that holds the virtual documents of notebooks.
const NOTEBOOK_DOCUMENT_DIRECTORY: &str = ".zed/notebook";

async fn link_sibling_modules(
    fs: &dyn Fs,
    notebook_directory: &Path,
    document_directory: &Path,
    document_path: &Path,
) -> Result<()> {
    let mut entries = fs.read_dir(notebook_directory).await?;
    while let Some(entry) = entries.next().await {
        let entry = entry?;
        let Some(name) = entry.file_name() else {
            continue;
        };
        if name.to_string_lossy().starts_with('.') || Some(name) == document_path.file_name() {
            continue;
        }
        let link = document_directory.join(name);
        if fs.metadata(&link).await?.is_some() {
            continue;
        }
        fs.create_symlink(&link, entry.clone())
            .await
            .with_context(|| format!("link {} into the notebook cache", entry.display()))?;
    }
    Ok(())
}

/// The remote counterpart of [`link_sibling_modules`]: the project filesystem has no
/// remote symlink operation, so the links are made by a short script over SSH.
fn link_remote_sibling_modules(
    project: &Entity<Project>,
    notebook_directory: &Path,
    document_path: &Path,
    cx: &App,
) -> Task<Result<()>> {
    const SCRIPT: &str = r#"
notebook_directory="$1"; document_directory="$2"; skipped="$3"
cd "$notebook_directory" || exit 0
for entry in *; do
    [ "$entry" = "$skipped" ] && continue
    [ -e "$entry" ] || continue
    [ -e "$document_directory/$entry" ] || [ -L "$document_directory/$entry" ] \
        || ln -s "$notebook_directory/$entry" "$document_directory/$entry"
done
"#;
    let Some(remote_client) = project.read(cx).remote_client() else {
        return Task::ready(Ok(()));
    };
    let (Some(document_directory), Some(document_name)) =
        (document_path.parent(), document_path.file_name())
    else {
        return Task::ready(Ok(()));
    };
    let arguments = [
        "-lc".to_string(),
        SCRIPT.to_string(),
        "link-siblings".to_string(),
        notebook_directory.to_string_lossy().into_owned(),
        document_directory.to_string_lossy().into_owned(),
        document_name.to_string_lossy().into_owned(),
    ];
    let command = remote_client.read(cx).build_command(
        Some("bash".to_string()),
        &arguments,
        &HashMap::default(),
        None,
        None,
        remote::Interactive::No,
    );
    let timer = cx.background_executor().timer(REMOTE_LINK_TIMEOUT);
    cx.background_spawn(async move {
        let command = command.context("build the ssh command to link sibling modules")?;
        let mut builder = util::command::new_command(&command.program);
        builder.args(&command.args).envs(&command.env);
        let output = smol::future::or(
            async { builder.output().await.map_err(anyhow::Error::from) },
            async {
                timer.await;
                Err(anyhow::anyhow!("timed out"))
            },
        )
        .await?;
        anyhow::ensure!(
            output.status.success(),
            "linking sibling modules failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        Ok(())
    })
}

impl NotebookLsp {
    pub fn new(
        project: Entity<Project>,
        notebook_path: PathBuf,
        language: Option<Arc<Language>>,
        cx: &mut Context<Self>,
    ) -> Self {
        let lsp_store = project.read(cx).lsp_store();
        let lsp_subscription = cx.subscribe(&lsp_store, |this, _lsp_store, event, cx| {
            match event {
                LspStoreEvent::DiagnosticsUpdated { server_id, paths } => {
                    if let Some(document) = this.document.as_ref()
                        && paths.contains(&document.path)
                    {
                        this.forward_diagnostics(*server_id, cx);
                    }
                }
                LspStoreEvent::LanguageServerAdded(..) => this.pull_document_diagnostics(cx),
                _ => {}
            }
        });

        let mut this = Self {
            project,
            notebook_path,
            language,
            document: None,
            cells: Vec::new(),
            python_interpreter: None,
            kernel_completer: None,
            _rewrite_task: None,
            _setup_task: None,
            _toolchain_task: None,
            _lsp_subscription: lsp_subscription,
            owns_document: false,
        };

        cx.on_release(|this, cx| {
            let (true, Some(document)) = (this.owns_document, this.document.as_ref()) else {
                return;
            };
            let path = document.path.clone();
            this.project.update(cx, |project, cx| {
                if let Some(task) = project.delete_file(path, cx) {
                    task.detach_and_log_err(cx);
                }
            });
        })
        .detach();

        this.spawn_setup(cx);
        this
    }

    /// Reconciles the tracked code cells with the notebook's current cells,
    /// rewriting the virtual document and installing providers as needed.
    pub fn set_cells(&mut self, cells: Vec<NotebookCell>, cx: &mut Context<Self>) {
        let mut new_states = Vec::with_capacity(cells.len());
        for cell in cells {
            if let Some(index) = self.cells.iter().position(|existing| existing.id == cell.id) {
                let mut state = self.cells.swap_remove(index);
                state.editor = cell.editor;
                state.buffer = cell.buffer;
                new_states.push(state);
            } else {
                let subscription = cx.subscribe(
                    &cell.buffer,
                    |this, _buffer, event: &BufferEvent, cx| {
                        if matches!(event, BufferEvent::Edited { .. }) {
                            this.schedule_rewrite(cx);
                        }
                    },
                );
                new_states.push(CellState {
                    id: cell.id,
                    editor: cell.editor,
                    buffer: cell.buffer,
                    span: None,
                    providers_installed: false,
                    _subscription: subscription,
                });
            }
        }

        self.cells = new_states;
        log::info!("notebook_lsp: synced {} code cells", self.cells.len());
        self.rewrite_document(cx);
        self.install_providers(cx);
    }

    /// Flushes the virtual document to disk, mirroring the notebook file.
    pub fn save(&self, cx: &mut App) {
        // Saving would write the virtual document into the remote project.
        if self.project.read(cx).is_via_remote_server() {
            return;
        }
        let Some(document) = self.document.as_ref() else {
            return;
        };
        let buffer = document.buffer.clone();
        let task = self
            .project
            .update(cx, |project, cx| project.save_buffer(buffer, cx));
        task.detach_and_log_err(cx);
    }

    /// Installs the kernel connection used to complement completions and hovers
    /// with live interpreter state. Passing a completer replaces any previous
    /// one, which is how a restarted kernel re-points at the new process.
    pub fn set_kernel_completer(&mut self, completer: Rc<dyn KernelCompleter>) {
        self.kernel_completer = Some(completer);
    }

    /// Points the language servers serving the virtual document at the Python
    /// interpreter of the notebook's active kernel.
    ///
    /// Notebook cells are mirrored into a virtual file that lives in its own
    /// throwaway worktree, so the project's own toolchain detection never runs
    /// for it. Without this, a language server resolving the virtual document
    /// uses the ambient interpreter and cannot see packages (for example
    /// `numpy`) that are only installed in the notebook's virtual environment.
    pub fn set_python_interpreter(&mut self, interpreter: PathBuf, cx: &mut Context<Self>) {
        if self.python_interpreter.as_ref() == Some(&interpreter) {
            return;
        }
        log::info!(
            "notebook_lsp: active kernel python interpreter is {}",
            interpreter.display()
        );
        self.python_interpreter = Some(interpreter);
        self.apply_python_interpreter(cx);
    }

    /// Resolves the interpreter recorded by [`NotebookLsp::set_python_interpreter`]
    /// into a toolchain and activates it for the virtual document's path, so
    /// that the Python language servers pick it up. A no-op until both the
    /// virtual document and an interpreter are known.
    fn apply_python_interpreter(&mut self, cx: &mut Context<Self>) {
        let (Some(document), Some(interpreter)) =
            (self.document.as_ref(), self.python_interpreter.clone())
        else {
            return;
        };

        let project = self.project.clone();
        let document_path = document.path.clone();
        let resolve = project.update(cx, |project, cx| {
            project.resolve_toolchain(
                interpreter.clone(),
                LanguageName::new_static("Python"),
                cx,
            )
        });

        let task = cx.spawn(async move |_this, cx| {
            let toolchain = match resolve.await {
                Ok(toolchain) => toolchain,
                Err(error) => {
                    log::warn!(
                        "notebook_lsp: could not resolve python toolchain for {}: {error:#}",
                        interpreter.display()
                    );
                    return;
                }
            };
            log::info!(
                "notebook_lsp: activating python toolchain '{}' ({}) for the virtual document",
                toolchain.name,
                toolchain.path
            );
            let activation = project.update(cx, |project, cx| {
                project.activate_toolchain(document_path, toolchain, cx)
            });
            if activation.await.is_none() {
                log::warn!("notebook_lsp: project does not support activating toolchains");
            }
        });
        self._toolchain_task = Some(task);
    }

    fn spawn_setup(&mut self, cx: &mut Context<Self>) {
        let project = self.project.clone();
        let language = self.language.clone();
        let notebook_path = self.notebook_path.clone();

        let setup_task = cx.spawn(async move |this, cx| {
            let setup = async {
                let extension = language
                    .as_ref()
                    .and_then(|language| language.path_suffixes().first().cloned())
                    .map(|suffix| suffix.to_string())
                    .unwrap_or_else(|| "txt".to_string());
                let stem = notebook_path
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .unwrap_or("notebook")
                    .to_string();
                let notebook_directory = notebook_path
                    .parent()
                    .context("notebook has no parent directory")?
                    .to_path_buf();
                // The document lives under `.zed/notebook` in the project that holds the
                // notebook, mirroring the notebook's own directory so two notebooks never
                // collide. Language servers then see the project's real config and, being
                // inside the project, run wherever the project does (including SSH hosts).
                let root = project
                    .read_with(cx, |project, cx| {
                        project.find_worktree(&notebook_path, cx).and_then(|(tree, _)| {
                            let tree = tree.read(cx);
                            (!tree.is_single_file()).then(|| tree.abs_path().to_path_buf())
                        })
                    })
                    .unwrap_or_else(|| notebook_directory.clone());
                let (worktree, root_path) = project
                    .update(cx, |project, cx| {
                        project.find_or_create_worktree(&root, false, cx)
                    })
                    .await
                    .context("find worktree of the notebook")?;
                let worktree_id = worktree.read_with(cx, |worktree, _| worktree.id());
                let path_style = project.read_with(cx, |project, cx| project.path_style(cx));
                let relative_directory = notebook_directory
                    .strip_prefix(&root)
                    .unwrap_or(Path::new(""));
                let layout = Path::new(NOTEBOOK_DOCUMENT_DIRECTORY)
                    .join(relative_directory)
                    .join(format!("{stem}.{extension}"));
                let relative_path: Arc<RelPath> = root_path
                    .join(&RelPath::new(&layout, path_style)?)
                    .into();
                let ignore_path = ProjectPath {
                    worktree_id,
                    path: root_path
                        .join(&RelPath::new(
                            &Path::new(NOTEBOOK_DOCUMENT_DIRECTORY).join(".gitignore"),
                            path_style,
                        )?)
                        .into(),
                };
                let project_path = ProjectPath {
                    worktree_id,
                    path: relative_path.clone(),
                };

                // Remote servers only open (and serve language servers for) files that
                // exist, so the document is a real empty file, removed on release.
                let exists = |path: &ProjectPath, cx: &mut AsyncApp| {
                    project.read_with(cx, |project, cx| project.entry_for_path(path, cx).is_some())
                };
                if !exists(&ignore_path, cx) {
                    // Self-ignoring directory: the files are internal, so git, project
                    // search and the file finder skip them. Deleting this file opts out.
                    project
                        .update(cx, |project, cx| {
                            project.create_entry(ignore_path.clone(), false, cx)
                        })
                        .await
                        .context("create notebook document ignore file")?;
                    let ignore_buffer = project
                        .update(cx, |project, cx| project.open_buffer(ignore_path.clone(), cx))
                        .await?;
                    ignore_buffer.update(cx, |buffer, cx| buffer.edit([(0..0, "*\n")], None, cx));
                    project
                        .update(cx, |project, cx| project.save_buffer(ignore_buffer, cx))
                        .await?;
                }
                if !exists(&project_path, cx) {
                    project
                        .update(cx, |project, cx| {
                            project.create_entry(project_path.clone(), false, cx)
                        })
                        .await
                        .context("create virtual notebook document")?;
                    this.update(cx, |this, _| this.owns_document = true)?;
                }

                // Language servers resolve imports relative to the importing file first,
                // so the modules next to the notebook are linked in beside the document.
                let document_path = root.join(&layout);
                let is_remote = project.read_with(cx, |project, _| project.is_via_remote_server());
                if is_remote {
                    cx.update(|cx| {
                        link_remote_sibling_modules(
                            &project,
                            &notebook_directory,
                            &document_path,
                            cx,
                        )
                    })
                    .await
                    .log_err();
                } else if let Some(document_directory) = document_path.parent() {
                    let fs = project.read_with(cx, |project, _| project.fs().clone());
                    link_sibling_modules(
                        fs.as_ref(),
                        &notebook_directory,
                        document_directory,
                        &document_path,
                    )
                    .await
                    .log_err();
                }
                log::info!(
                    "notebook_lsp: virtual document {} for {}",
                    relative_path.as_unix_str(),
                    notebook_path.display()
                );
                let buffer = project
                    .update(cx, |project, cx| {
                        project.open_buffer(project_path.clone(), cx)
                    })
                    .await
                    .context("open virtual notebook buffer")?;
                buffer.update(cx, |buffer, cx| buffer.set_language(language.clone(), cx));
                log::info!("notebook_lsp: opened virtual buffer");
                let lsp_handle = project.update(cx, |project, cx| {
                    project.register_buffer_with_language_servers(&buffer, cx)
                });
                log::info!("notebook_lsp: registered virtual buffer with language servers");

                this.update(cx, |this, cx| {
                    this.document = Some(DocumentState {
                        path: project_path,
                        buffer,
                        _lsp_handle: lsp_handle,
                        text: String::new(),
                    });
                    this.rewrite_document(cx);
                    this.install_providers(cx);
                    this.apply_python_interpreter(cx);
                })?;

                Ok::<(), anyhow::Error>(())
            };

            setup.await.log_err();
        });

        self._setup_task = Some(setup_task);
    }

    fn schedule_rewrite(&mut self, cx: &mut Context<Self>) {
        self._rewrite_task = Some(cx.spawn(async move |this, cx| {
            cx.background_executor().timer(REWRITE_DEBOUNCE).await;
            this.update(cx, |this, cx| this.rewrite_document(cx)).ok();
        }));
    }

    fn rewrite_document(&mut self, cx: &mut Context<Self>) {
        if self.document.is_none() {
            return;
        }

        let cells: Vec<(CellId, String)> = self
            .cells
            .iter()
            .map(|cell| (cell.id.clone(), cell.buffer.read(cx).text()))
            .collect();
        let (new_text, spans) = build_document(&cells);

        for (cell, span) in self.cells.iter_mut().zip(spans) {
            cell.span = Some(span);
        }

        let Some(document) = self.document.as_mut() else {
            return;
        };
        if document.text == new_text {
            return;
        }

        let old_length = document.text.len();
        let buffer = document.buffer.clone();
        buffer.update(cx, |buffer, cx| {
            buffer.edit([(0..old_length, new_text.clone())], None, cx);
        });
        document.text = new_text;
        log::info!(
            "notebook_lsp: rewrote virtual document ({} bytes)",
            document.text.len()
        );
        self.pull_document_diagnostics(cx);
    }

    /// Servers such as ruff publish diagnostics only when asked, and editors do the
    /// asking for the buffers they show. Nothing displays the virtual document, so it
    /// is pulled here whenever its text or the set of servers changes.
    fn pull_document_diagnostics(&self, cx: &mut Context<Self>) {
        let Some(document) = self.document.as_ref() else {
            return;
        };
        let buffer = document.buffer.clone();
        let pull = self.project.read(cx).lsp_store().update(cx, |lsp_store, cx| {
            lsp_store.pull_diagnostics_for_buffer(buffer, cx)
        });
        pull.detach_and_log_err(cx);
    }

    /// Applies a transaction produced against the virtual document to the cell
    /// buffers that own the edited ranges, and resynchronizes the document from
    /// those cells.
    ///
    /// The returned transaction contains the cell buffers instead of the
    /// virtual document, so the editor never opens an internal cache file. Any
    /// entries for other files are passed through untouched.
    fn apply_document_transaction(
        &mut self,
        mut transaction: ProjectTransaction,
        document_version_before: Global,
        cx: &mut Context<Self>,
    ) -> Result<ProjectTransaction> {
        let Some(document) = self.document.as_ref() else {
            return Ok(transaction);
        };
        let document_buffer = document.buffer.clone();
        if transaction.0.remove(&document_buffer).is_none() {
            return Ok(transaction);
        }

        let document_after = document_buffer.read(cx).snapshot();
        let document_edits: Vec<text::Edit<usize>> = document_after
            .edits_since::<usize>(&document_version_before)
            .collect();

        let mut edits_per_cell: Vec<(Entity<Buffer>, Vec<(Range<usize>, String)>)> = Vec::new();
        for edit in document_edits {
            let Some((cell_buffer, span)) =
                self.cells.iter().find_map(|cell| {
                    let span = cell.span.as_ref()?;
                    (span.start_offset <= edit.old.start && edit.old.end <= span.end_offset)
                        .then(|| (cell.buffer.clone(), span.clone()))
                })
            else {
                log::warn!(
                    "notebook_lsp: dropping a document edit outside every cell ({:?})",
                    edit.old
                );
                continue;
            };

            let replacement: String = document_after
                .text_for_range(edit.new.clone())
                .collect();
            let cell_range = (edit.old.start - span.start_offset)
                ..(edit.old.end - span.start_offset);

            match edits_per_cell
                .iter_mut()
                .find(|(buffer, _)| *buffer == cell_buffer)
            {
                Some((_, edits)) => edits.push((cell_range, replacement)),
                None => edits_per_cell.push((cell_buffer, vec![(cell_range, replacement)])),
            }
        }

        for (cell_buffer, edits) in edits_per_cell {
            let cell_length = cell_buffer.read(cx).len();
            let mut applicable_edits = Vec::with_capacity(edits.len());
            for (range, replacement) in edits {
                if range.end > cell_length {
                    log::warn!(
                        "notebook_lsp: dropping a document edit past the end of a cell ({range:?})"
                    );
                    continue;
                }
                applicable_edits.push((range, replacement));
            }
            if applicable_edits.is_empty() {
                continue;
            }

            let transaction_id = cell_buffer.update(cx, |buffer, cx| {
                buffer.start_transaction();
                buffer.edit(applicable_edits, None, cx);
                buffer.end_transaction(cx)
            });
            let Some(transaction_id) = transaction_id else {
                continue;
            };
            let cell_transaction = cell_buffer.update(cx, |buffer, _| {
                buffer.finalize_last_transaction();
                buffer.get_transaction(transaction_id).cloned()
            });
            if let Some(cell_transaction) = cell_transaction {
                transaction.0.insert(cell_buffer, cell_transaction);
            }
        }

        self.resync_document(cx);
        Ok(transaction)
    }

    /// Rewrites the virtual document from the current cell text, even when the
    /// cached text is stale because the document was edited directly (for
    /// example by a rename). This keeps the document consistent with the cells
    /// without replaying the same edit twice.
    fn resync_document(&mut self, cx: &mut Context<Self>) {
        let cells: Vec<(CellId, String)> = self
            .cells
            .iter()
            .map(|cell| (cell.id.clone(), cell.buffer.read(cx).text()))
            .collect();
        let (new_text, spans) = build_document(&cells);
        for (cell, span) in self.cells.iter_mut().zip(spans) {
            cell.span = Some(span);
        }

        let Some(document) = self.document.as_mut() else {
            return;
        };
        let current_length = document.buffer.read(cx).len();
        let current_text = document.buffer.read(cx).text();
        if current_text != new_text {
            let new_text_for_buffer = new_text.clone();
            document.buffer.update(cx, |buffer, cx| {
                buffer.edit(
                    [(0..current_length, new_text_for_buffer)],
                    None,
                    cx,
                );
            });
        }
        document.text = new_text;
    }

    fn install_providers(&mut self, cx: &mut Context<Self>) {
        if self.document.is_none() {
            return;
        }

        let notebook = cx.entity().downgrade();
        let project = self.project.downgrade();
        for cell in &mut self.cells {
            if cell.providers_installed {
                continue;
            }

            let completion = Rc::new(CellCompletionProvider {
                notebook: notebook.clone(),
                cell_id: cell.id.clone(),
            });
            let semantics = Rc::new(CellSemanticsProvider {
                notebook: notebook.clone(),
                cell_id: cell.id.clone(),
                project: project.clone(),
            });
            let code_actions = Rc::new(CellCodeActionProvider {
                notebook: notebook.clone(),
                cell_id: cell.id.clone(),
            });
            cell.editor.update(cx, |editor, _cx| {
                editor.set_completion_provider(Some(completion));
                editor.set_semantics_provider(Some(semantics));
                editor.add_code_action_provider(code_actions);
            });
            cell.providers_installed = true;
        }
    }

    fn forward_diagnostics(&self, server_id: LanguageServerId, cx: &mut Context<Self>) {
        let Some(document) = self.document.as_ref() else {
            return;
        };

        let source_entries: Vec<(Range<Anchor>, Diagnostic)> = document
            .buffer
            .read(cx)
            .buffer_diagnostics(Some(server_id))
            .into_iter()
            .map(|entry| (entry.range.clone(), entry.diagnostic.clone()))
            .collect();
        let document_snapshot = document.buffer.read(cx).snapshot();

        let cell_buffers: Vec<(Entity<Buffer>, CellSpan)> = self
            .cells
            .iter()
            .filter_map(|cell| Some((cell.buffer.clone(), cell.span.clone()?)))
            .collect();

        for (cell_buffer, span) in cell_buffers {
            let cell_snapshot = cell_buffer.read(cx).snapshot();
            let entries: Vec<DiagnosticEntry<Anchor>> = source_entries
                .iter()
                .filter_map(|(range, diagnostic)| {
                    let start =
                        translate_anchor(&range.start, &document_snapshot, &cell_snapshot, &span)?;
                    let end =
                        translate_anchor(&range.end, &document_snapshot, &cell_snapshot, &span)?;
                    Some(DiagnosticEntry {
                        range: start..end,
                        diagnostic: diagnostic.clone(),
                        related_information: None,
                    })
                })
                .collect();
            let set = DiagnosticSet::from_sorted_entries(entries, &cell_snapshot);
            cell_buffer.update(cx, |buffer, cx| buffer.update_diagnostics(server_id, set, cx));
        }
    }
}

fn translate_anchor(
    anchor: &Anchor,
    document: &BufferSnapshot,
    cell: &BufferSnapshot,
    span: &CellSpan,
) -> Option<Anchor> {
    let document_offset = anchor.to_offset(document);
    let cell_offset = cell_offset_for_document(span, document_offset)?;
    Some(cell.anchor_before(cell_offset))
}

/// Translates a range in the virtual document into the buffer of the cell that
/// owns it. Both ends must lie inside the cell; a range that reaches into a
/// separator or another cell belongs to no single cell and is rejected.
fn translate_range(
    range: &Range<Anchor>,
    document: &BufferSnapshot,
    cell: &BufferSnapshot,
    span: &CellSpan,
) -> Option<Range<Anchor>> {
    let start = translate_anchor(&range.start, document, cell, span)?;
    let end = translate_anchor(&range.end, document, cell, span)?;
    Some(start..end)
}

fn translate_completion(
    completion: &mut Completion,
    document: &BufferSnapshot,
    cell: &BufferSnapshot,
    span: &CellSpan,
) -> bool {
    let Some(start) = translate_anchor(&completion.replace_range.start, document, cell, span)
    else {
        return false;
    };
    let Some(end) = translate_anchor(&completion.replace_range.end, document, cell, span) else {
        return false;
    };
    completion.replace_range = start..end;

    completion.match_start =
        completion
            .match_start
            .and_then(|anchor| translate_anchor(&anchor, document, cell, span));

    if let CompletionSource::Lsp { insert_range, .. } = &mut completion.source {
        *insert_range = insert_range.take().and_then(|range| {
            let start = translate_anchor(&range.start, document, cell, span)?;
            let end = translate_anchor(&range.end, document, cell, span)?;
            Some(start..end)
        });
    }

    true
}

/// Resolves a Unicode code point index into the byte offset of the character
/// that starts there, or the string length when the index points just past the
/// end. Returns `None` when the index is beyond the string.
fn code_point_to_byte(text: &str, code_point: usize) -> Option<usize> {
    if code_point == 0 {
        return Some(0);
    }
    let mut seen = 0;
    for (offset, _) in text.char_indices() {
        if seen == code_point {
            return Some(offset);
        }
        seen += 1;
    }
    (seen == code_point).then_some(text.len())
}

/// Appends the kernel's completions that the language server did not already
/// return, matched by label. The kernel's `cursor_start`/`cursor_end` are code
/// point positions within the cell, converted here to a byte range in the cell
/// buffer. Nothing is appended when the reply falls outside the cell or repeats
/// a label already present.
fn merge_kernel_completions(
    responses: &mut Vec<CompletionResponse>,
    cell: &BufferSnapshot,
    completion: KernelCompletion,
) {
    let cell_text = cell.text();
    let (Some(start), Some(end)) = (
        code_point_to_byte(&cell_text, completion.cursor_start),
        code_point_to_byte(&cell_text, completion.cursor_end),
    ) else {
        log::warn!("notebook_lsp: kernel completion cursor was outside the cell");
        return;
    };
    if start > end {
        log::warn!("notebook_lsp: kernel completion cursor range was inverted");
        return;
    }

    let mut labels: HashSet<String> = responses
        .iter()
        .flat_map(|response| response.completions.iter())
        .map(|completion| completion.label.text.clone())
        .collect();

    let replace_range = cell.anchor_before(start)..cell.anchor_before(end);
    let mut completions = Vec::new();
    for matched in completion.matches {
        if !labels.insert(matched.clone()) {
            continue;
        }
        completions.push(Completion {
            replace_range: replace_range.clone(),
            new_text: matched.clone(),
            label: CodeLabel::plain(matched, None),
            documentation: None,
            source: CompletionSource::Custom,
            icon_path: None,
            icon_color: None,
            match_start: None,
            snippet_deduplication_key: None,
            insert_text_mode: None,
            confirm: None,
            group: None,
        });
    }

    if !completions.is_empty() {
        responses.push(CompletionResponse {
            completions,
            display_options: CompletionDisplayOptions::default(),
            is_incomplete: false,
        });
    }
}

/// Builds a hover from the kernel's inspection of the symbol under the cursor.
fn hover_from_inspection(inspection: KernelInspection) -> Hover {
    let kind = match inspection.kind {
        KernelInspectionKind::PlainText => HoverBlockKind::PlainText,
        KernelInspectionKind::Markdown => HoverBlockKind::Markdown,
    };
    Hover {
        contents: vec![HoverBlock {
            text: inspection.text,
            kind,
        }],
        range: None,
        language: None,
    }
}

/// Awaits a kernel completion, discarding the result if the kernel is slow or
/// errors so that the language server's completions can be returned on their
/// own.
async fn await_kernel_completion(
    request: Task<Result<Option<KernelCompletion>>>,
    executor: &BackgroundExecutor,
) -> Option<KernelCompletion> {
    match request.with_timeout(KERNEL_REQUEST_TIMEOUT, executor).await {
        Ok(Ok(completion)) => completion,
        Ok(Err(error)) => {
            log::warn!("notebook_lsp: kernel completion failed: {error:#}");
            None
        }
        Err(_) => {
            log::info!("notebook_lsp: kernel completion timed out");
            None
        }
    }
}

/// Awaits a kernel inspection, discarding the result on error or timeout so a
/// hover can still show the language server's result or nothing at all.
async fn await_kernel_inspection(
    request: Task<Result<Option<KernelInspection>>>,
    executor: &BackgroundExecutor,
) -> Option<KernelInspection> {
    match request.with_timeout(KERNEL_REQUEST_TIMEOUT, executor).await {
        Ok(Ok(inspection)) => inspection,
        Ok(Err(error)) => {
            log::warn!("notebook_lsp: kernel inspection failed: {error:#}");
            None
        }
        Err(_) => {
            log::info!("notebook_lsp: kernel inspection timed out");
            None
        }
    }
}

struct CellCompletionProvider {
    notebook: WeakEntity<NotebookLsp>,
    cell_id: CellId,
}

impl CompletionProvider for CellCompletionProvider {
    fn completions(
        &self,
        buffer: &Entity<Buffer>,
        buffer_position: Anchor,
        trigger: CompletionContext,
        _window: &mut Window,
        cx: &mut Context<Editor>,
    ) -> Task<Result<Vec<CompletionResponse>>> {
        let Some(notebook) = self.notebook.upgrade() else {
            return Task::ready(Ok(Vec::new()));
        };
        let Some((span, document_buffer, project, completer)) = notebook
            .update(cx, |notebook, cx| {
                // The debounced rewrite may not have run since the last keystroke.
                notebook.rewrite_document(cx);
                let document = notebook.document.as_ref()?;
                let cell = notebook.cells.iter().find(|cell| cell.id == self.cell_id)?;
                Some((
                    cell.span()?.clone(),
                    document.buffer.clone(),
                    notebook.project.clone(),
                    notebook.kernel_completer.clone(),
                ))
            })
        else {
            return Task::ready(Ok(Vec::new()));
        };

        let cell_snapshot = buffer.read(cx).snapshot();
        let cell_offset = buffer_position.to_offset(&cell_snapshot);
        let Some(document_offset) = document_offset_for_cell(&span, cell_offset) else {
            return Task::ready(Ok(Vec::new()));
        };
        let document_point = document_buffer
            .read(cx)
            .snapshot()
            .offset_to_point(document_offset);
        log::info!(
            "notebook_lsp: completion request for cell {:?} translated to document point {:?}",
            self.cell_id,
            document_point
        );

        let request = project.update(cx, |project, cx| {
            project.completions(&document_buffer, document_point, trigger, cx)
        });

        // The kernel request is started before awaiting the language server so
        // the two run concurrently; the language server's results still take
        // precedence when they are merged.
        let cell_text = cell_snapshot.text();
        let cursor_pos = cell_text[..cell_offset].chars().count();
        let kernel_request = completer
            .map(|completer| completer.complete(cell_text.clone(), cursor_pos));

        let cell_buffer = buffer.clone();
        let cell_id = self.cell_id.clone();
        cx.spawn(async move |_editor, cx| {
            let mut responses = request.await?;
            let document_snapshot = cx.update(|cx| document_buffer.read(cx).snapshot());
            let cell_snapshot = cx.update(|cx| cell_buffer.read(cx).snapshot());
            for response in &mut responses {
                response.completions.retain_mut(|completion| {
                    translate_completion(completion, &document_snapshot, &cell_snapshot, &span)
                });
            }

            if let Some(kernel_request) = kernel_request {
                let completion =
                    await_kernel_completion(kernel_request, cx.background_executor()).await;
                if let Some(completion) = completion {
                    merge_kernel_completions(&mut responses, &cell_snapshot, completion);
                }
            }

            let completion_count: usize = responses.iter().map(|r| r.completions.len()).sum();
            log::info!(
                "notebook_lsp: completion request for cell {:?} returned {} completions",
                cell_id,
                completion_count
            );
            Ok(responses)
        })
    }

    fn resolve_completions(
        &self,
        _buffer: Entity<Buffer>,
        completion_indices: Vec<usize>,
        completions: Rc<RefCell<Box<[Completion]>>>,
        cx: &mut Context<Editor>,
    ) -> Task<Result<bool>> {
        let Some(notebook) = self.notebook.upgrade() else {
            return Task::ready(Ok(false));
        };
        let Some((document_buffer, lsp_store)) = notebook
            .read_with(cx, |notebook, cx| {
                let document = notebook.document.as_ref()?;
                Some((document.buffer.clone(), notebook.project.read(cx).lsp_store()))
            })
        else {
            return Task::ready(Ok(false));
        };

        // Only completions that came from a language server can be resolved
        // remotely; the kernel completions carry `CompletionSource::Custom`.
        let completion_indices: Vec<usize> = completion_indices
            .into_iter()
            .filter(|index| {
                completions
                    .borrow()
                    .get(*index)
                    .is_some_and(|completion| completion.source.server_id().is_some())
            })
            .collect();
        if completion_indices.is_empty() {
            return Task::ready(Ok(false));
        }

        lsp_store.update(cx, |lsp_store, cx| {
            lsp_store.resolve_completions(
                document_buffer,
                completion_indices,
                completions,
                cx,
            )
        })
    }

    fn is_completion_trigger(
        &self,
        buffer: &Entity<Buffer>,
        position: Anchor,
        text: &str,
        trigger_in_words: bool,
        cx: &mut Context<Editor>,
    ) -> bool {
        let mut chars = text.chars();
        let Some(character) = chars.next() else {
            log::info!(
                "notebook_lsp: is_completion_trigger cell {:?} text={text:?} -> false (empty text)",
                self.cell_id
            );
            return false;
        };
        if chars.next().is_some() {
            log::info!(
                "notebook_lsp: is_completion_trigger cell {:?} text={text:?} -> false (multi character)",
                self.cell_id
            );
            return false;
        }

        // A code cell is an anonymous local buffer that is not itself registered
        // with a language server, so its `completion_triggers` is always empty.
        // The server-supplied trigger characters live on the virtual document,
        // which is registered with the language servers. Classify the typed
        // character against the cell buffer, which is always up to date locally,
        // so the result never depends on the debounced document rewrite.
        let is_word_trigger = trigger_in_words
            && buffer
                .read(cx)
                .snapshot()
                .char_classifier_at(position)
                .scope_context(Some(CharScopeContext::Completion))
                .is_word(character);

        let is_server_trigger = self.notebook.upgrade().is_some_and(|notebook| {
            notebook
                .read_with(cx, |notebook, cx| {
                    notebook.document.as_ref().is_some_and(|document| {
                        document
                            .buffer
                            .read(cx)
                            .completion_triggers()
                            .contains(text)
                    })
                })
        });

        let result = is_word_trigger || is_server_trigger;
        log::info!(
            "notebook_lsp: is_completion_trigger cell {:?} text={text:?} trigger_in_words={trigger_in_words} word={is_word_trigger} server={is_server_trigger} -> {result}",
            self.cell_id
        );
        result
    }
}

struct CellSemanticsProvider {
    notebook: WeakEntity<NotebookLsp>,
    cell_id: CellId,
    project: WeakEntity<Project>,
}

impl SemanticsProvider for CellSemanticsProvider {
    fn hover(
        &self,
        buffer: &Entity<Buffer>,
        position: Anchor,
        cx: &mut App,
    ) -> Option<Task<Option<Vec<Hover>>>> {
        let notebook = self.notebook.upgrade()?;
        let (span, document_buffer, project, completer) = notebook
            .update(cx, |notebook, cx| {
                // The debounced rewrite may not have run since the last keystroke.
                notebook.rewrite_document(cx);
                let document = notebook.document.as_ref()?;
                let cell = notebook.cells.iter().find(|cell| cell.id == self.cell_id)?;
                Some((
                    cell.span()?.clone(),
                    document.buffer.clone(),
                    notebook.project.clone(),
                    notebook.kernel_completer.clone(),
                ))
            })?;

        let cell_snapshot = buffer.read(cx).snapshot();
        let cell_offset = position.to_offset(&cell_snapshot);
        let document_offset = document_offset_for_cell(&span, cell_offset)?;
        let document_point = document_buffer
            .read(cx)
            .snapshot()
            .offset_to_point(document_offset);

        let request =
            project.update(cx, |project, cx| {
                project.hover(&document_buffer, document_point, cx)
            });

        let cell_text = cell_snapshot.text();
        let cursor_pos = cell_text[..cell_offset].chars().count();
        let cell_buffer = buffer.clone();
        Some(cx.spawn(async move |cx| {
            let hovers = request.await.unwrap_or_default();
            let document_snapshot = cx.update(|cx| document_buffer.read(cx).snapshot());
            let cell_snapshot = cx.update(|cx| cell_buffer.read(cx).snapshot());
            let translated: Vec<Hover> = hovers
                .into_iter()
                .filter_map(|mut hover| {
                    hover.range = match hover.range {
                        Some(range) => {
                            let start = translate_anchor(
                                &range.start,
                                &document_snapshot,
                                &cell_snapshot,
                                &span,
                            )?;
                            let end = translate_anchor(
                                &range.end,
                                &document_snapshot,
                                &cell_snapshot,
                                &span,
                            )?;
                            Some(start..end)
                        }
                        None => None,
                    };
                    Some(hover)
                })
                .collect();

            // The kernel only fills the gap the language server left; when it
            // has something to say, prefer it over an empty hover. A failure or
            // timeout simply leaves the language server's result untouched.
            let Some(completer) = completer else {
                return Some(translated);
            };
            if !translated.is_empty() {
                return Some(translated);
            }

            let inspection = await_kernel_inspection(
                completer.inspect(cell_text, cursor_pos),
                cx.background_executor(),
            )
            .await;
            match inspection {
                Some(inspection) => Some(vec![hover_from_inspection(inspection)]),
                None => Some(translated),
            }
        }))
    }

    fn inline_values(
        &self,
        buffer_handle: Entity<Buffer>,
        range: Range<Anchor>,
        cx: &mut App,
    ) -> Option<Task<Result<Vec<InlayHint>>>> {
        self.project.inline_values(buffer_handle, range, cx)
    }

    fn applicable_inlay_chunks(
        &self,
        buffer: &Entity<Buffer>,
        ranges: &[Range<Anchor>],
        cx: &mut App,
    ) -> Vec<Range<BufferRow>> {
        self.project.applicable_inlay_chunks(buffer, ranges, cx)
    }

    fn invalidate_inlay_hints(&self, for_buffers: &HashSet<text::BufferId>, cx: &mut App) {
        self.project.invalidate_inlay_hints(for_buffers, cx)
    }

    fn inlay_hints(
        &self,
        invalidate: InvalidationStrategy,
        buffer: Entity<Buffer>,
        ranges: Vec<Range<Anchor>>,
        known_chunks: Option<(Global, HashSet<Range<BufferRow>>)>,
        cx: &mut App,
    ) -> Option<HashMap<Range<BufferRow>, Task<Result<CacheInlayHints>>>> {
        self.project
            .inlay_hints(invalidate, buffer, ranges, known_chunks, cx)
    }

    fn semantic_tokens(
        &self,
        buffer: Entity<Buffer>,
        cx: &mut App,
    ) -> Option<Shared<Task<Result<BufferSemanticTokens, Arc<anyhow::Error>>>>> {
        self.project.semantic_tokens(buffer, cx)
    }

    fn supports_inlay_hints(&self, buffer: &Entity<Buffer>, cx: &mut App) -> bool {
        self.project.supports_inlay_hints(buffer, cx)
    }

    fn supports_semantic_tokens(&self, buffer: &Entity<Buffer>, cx: &mut App) -> bool {
        self.project.supports_semantic_tokens(buffer, cx)
    }

    fn document_highlights(
        &self,
        buffer: &Entity<Buffer>,
        position: Anchor,
        cx: &mut App,
    ) -> Option<Task<Result<Vec<DocumentHighlight>>>> {
        let notebook = self.notebook.upgrade()?;
        let (span, document_buffer, project) = notebook.update(cx, |notebook, cx| {
            notebook.rewrite_document(cx);
            let document = notebook.document.as_ref()?;
            let cell = notebook.cells.iter().find(|cell| cell.id == self.cell_id)?;
            Some((
                cell.span()?.clone(),
                document.buffer.clone(),
                notebook.project.clone(),
            ))
        })?;

        let cell_buffer = buffer.clone();
        let cell_snapshot = cell_buffer.read(cx).snapshot();
        let cell_offset = position.to_offset(&cell_snapshot);
        let document_offset = document_offset_for_cell(&span, cell_offset)?;

        let request = project.update(cx, |project, cx| {
            project.document_highlights(&document_buffer, document_offset, cx)
        });

        Some(cx.spawn(async move |cx| {
            let highlights = request.await?;
            let (document_snapshot, cell_snapshot) = cx.update(|cx| {
                (
                    document_buffer.read(cx).snapshot(),
                    cell_buffer.read(cx).snapshot(),
                )
            });
            // Highlights are useless outside the cell that asked for them, so
            // anything that does not translate into that cell is dropped.
            let translated: Vec<DocumentHighlight> = highlights
                .into_iter()
                .filter_map(|mut highlight| {
                    let start = translate_anchor(
                        &highlight.range.start,
                        &document_snapshot,
                        &cell_snapshot,
                        &span,
                    )?;
                    let end = translate_anchor(
                        &highlight.range.end,
                        &document_snapshot,
                        &cell_snapshot,
                        &span,
                    )?;
                    highlight.range = start..end;
                    Some(highlight)
                })
                .collect();
            Ok(translated)
        }))
    }

    fn definitions(
        &self,
        buffer: &Entity<Buffer>,
        position: Anchor,
        kind: GotoDefinitionKind,
        cx: &mut App,
    ) -> Option<Task<Result<Option<Vec<LocationLink>>>>> {
        let notebook = self.notebook.upgrade()?;
        let (span, document_buffer, cells, project) = notebook.update(cx, |notebook, cx| {
            notebook.rewrite_document(cx);
            let document = notebook.document.as_ref()?;
            let cell_span = notebook
                .cells
                .iter()
                .find(|cell| cell.id == self.cell_id)?
                .span()?
                .clone();
            let cells = notebook
                .cells
                .iter()
                .filter_map(|cell| {
                    Some((
                        cell.id.clone(),
                        cell.buffer.clone(),
                        cell.span()?.clone(),
                    ))
                })
                .collect::<Vec<_>>();
            Some((
                cell_span,
                document.buffer.clone(),
                cells,
                notebook.project.clone(),
            ))
        })?;

        let cell_snapshot = buffer.read(cx).snapshot();
        let cell_offset = position.to_offset(&cell_snapshot);
        let document_offset = document_offset_for_cell(&span, cell_offset)?;

        let request = project.update(cx, |project, cx| match kind {
            GotoDefinitionKind::Symbol => project.definitions(&document_buffer, document_offset, cx),
            GotoDefinitionKind::Declaration => {
                project.declarations(&document_buffer, document_offset, cx)
            }
            GotoDefinitionKind::Type => {
                project.type_definitions(&document_buffer, document_offset, cx)
            }
            GotoDefinitionKind::Implementation => {
                project.implementations(&document_buffer, document_offset, cx)
            }
        });

        Some(cx.spawn(async move |cx| {
            let Some(links) = request.await? else {
                return Ok(None);
            };
            let remapped: Vec<LocationLink> = cx.update(|cx| {
                let document_snapshot = document_buffer.read(cx).snapshot();
                let cells = cells
                    .into_iter()
                    .map(|(cell_id, buffer, span)| {
                        let snapshot = buffer.read(cx).snapshot();
                        (cell_id, buffer, snapshot, span)
                    })
                    .collect::<Vec<_>>();
                links
                    .into_iter()
                    .filter_map(|link| {
                        remap_location_link(
                            link,
                            &document_buffer,
                            &document_snapshot,
                            &cells,
                        )
                    })
                    .collect()
            });
            Ok(Some(remapped))
        }))
    }

    fn references(
        &self,
        buffer: &Entity<Buffer>,
        position: Anchor,
        cx: &mut App,
    ) -> Option<Task<Result<Option<Vec<Location>>>>> {
        let notebook = self.notebook.upgrade()?;
        let (span, document_buffer, cells, project) = notebook.update(cx, |notebook, cx| {
            // The debounced rewrite may not have run since the last keystroke.
            notebook.rewrite_document(cx);
            let document = notebook.document.as_ref()?;
            let cell_span = notebook
                .cells
                .iter()
                .find(|cell| cell.id == self.cell_id)?
                .span()?
                .clone();
            let cells = notebook
                .cells
                .iter()
                .filter_map(|cell| {
                    Some((cell.id.clone(), cell.buffer.clone(), cell.span()?.clone()))
                })
                .collect::<Vec<_>>();
            Some((
                cell_span,
                document.buffer.clone(),
                cells,
                notebook.project.clone(),
            ))
        })?;

        let cell_snapshot = buffer.read(cx).snapshot();
        let cell_offset = position.to_offset(&cell_snapshot);
        let document_offset = document_offset_for_cell(&span, cell_offset)?;

        let request = project.update(cx, |project, cx| {
            project.references(&document_buffer, document_offset, cx)
        });

        Some(cx.spawn(async move |cx| {
            let Some(locations) = request.await? else {
                return Ok(None);
            };
            let remapped: Vec<Location> = cx.update(|cx| {
                let document_snapshot = document_buffer.read(cx).snapshot();
                let cells = cells
                    .into_iter()
                    .map(|(cell_id, buffer, span)| {
                        let snapshot = buffer.read(cx).snapshot();
                        (cell_id, buffer, snapshot, span)
                    })
                    .collect::<Vec<_>>();
                // References are only useful where the user can see them, so a
                // location on a separator (or in a cell that no longer exists)
                // is dropped; files outside the document are kept as they are.
                locations
                    .into_iter()
                    .filter_map(|location| {
                        remap_document_location(
                            location,
                            &document_buffer,
                            &document_snapshot,
                            &cells,
                        )
                    })
                    .collect()
            });
            Ok(Some(remapped))
        }))
    }

    fn format(
        &self,
        buffer: &Entity<Buffer>,
        ranges: Option<Vec<Range<Anchor>>>,
        trigger: FormatTrigger,
        cx: &mut App,
    ) -> Option<Task<Result<ProjectTransaction>>> {
        let notebook = self.notebook.upgrade()?;
        let (document_buffer, project, document_version_before, format_target) =
            notebook.update(cx, |notebook, cx| {
                notebook.rewrite_document(cx);
                let document = notebook.document.as_ref()?;
                let cell = notebook.cells.iter().find(|cell| cell.id == self.cell_id)?;
                let span = cell.span()?;
                let cell_snapshot = buffer.read(cx).snapshot();

                let document_ranges = match ranges {
                    Some(ranges) => ranges
                        .into_iter()
                        .filter_map(|range| {
                            let start = range.start.to_offset(&cell_snapshot);
                            let end = range.end.to_offset(&cell_snapshot);
                            let start = document_offset_for_cell(span, start)?;
                            let end = document_offset_for_cell(span, end)?;
                            Some(start..end)
                        })
                        .collect::<Vec<_>>(),
                    None => vec![span.start_offset..span.end_offset],
                };
                if document_ranges.is_empty() {
                    return None;
                }

                let document_snapshot = document.buffer.read(cx).snapshot();
                let document_ranges = document_ranges
                    .into_iter()
                    .map(|range| {
                        document_snapshot.anchor_before(range.start)
                            ..document_snapshot.anchor_after(range.end)
                    })
                    .collect::<Vec<_>>();
                let mut ranges_by_buffer = std::collections::BTreeMap::new();
                ranges_by_buffer.insert(document_snapshot.remote_id(), document_ranges);

                Some((
                    document.buffer.clone(),
                    notebook.project.clone(),
                    document_snapshot.version.clone(),
                    LspFormatTarget::Ranges(ranges_by_buffer),
                ))
            })?;

        let request = project.update(cx, |project, cx| {
            project.format(
                HashSet::from_iter([document_buffer]),
                format_target,
                true,
                trigger,
                cx,
            )
        });

        Some(cx.spawn(async move |cx| {
            let transaction = request.await?;
            notebook.update(cx, |notebook, cx| {
                notebook.apply_document_transaction(transaction, document_version_before, cx)
            })
        }))
    }

    fn range_for_rename(
        &self,
        buffer: &Entity<Buffer>,
        position: Anchor,
        cx: &mut App,
    ) -> Task<Result<Option<RenameTarget>>> {
        let Some(notebook) = self.notebook.upgrade() else {
            return Task::ready(Ok(None));
        };
        let Some((span, document_buffer, project)) = notebook.update(cx, |notebook, cx| {
            // The debounced rewrite may not have run since the last keystroke.
            notebook.rewrite_document(cx);
            let document = notebook.document.as_ref()?;
            let cell = notebook.cells.iter().find(|cell| cell.id == self.cell_id)?;
            Some((
                cell.span()?.clone(),
                document.buffer.clone(),
                notebook.project.clone(),
            ))
        }) else {
            return Task::ready(Ok(None));
        };

        let cell_snapshot = buffer.read(cx).snapshot();
        let cell_offset = position.to_offset(&cell_snapshot);
        let Some(document_offset) = document_offset_for_cell(&span, cell_offset) else {
            return Task::ready(Ok(None));
        };
        let document_point = document_buffer
            .read(cx)
            .snapshot()
            .offset_to_point(document_offset);

        let request = project.update(cx, |project, cx| {
            project.prepare_rename(document_buffer.clone(), document_point, cx)
        });

        let cell_buffer = buffer.clone();
        cx.spawn(async move |cx| {
            let response = request.await?;
            let (cell_snapshot, document_snapshot) = cx.update(|cx| {
                (
                    cell_buffer.read(cx).snapshot(),
                    document_buffer.read(cx).snapshot(),
                )
            });
            Ok(match response {
                PrepareRenameResponse::Success {
                    range,
                    language_server_id,
                } => translate_range(&range, &document_snapshot, &cell_snapshot, &span)
                    .map(|range| RenameTarget::new(range, language_server_id)),
                PrepareRenameResponse::InvalidPosition => None,
                PrepareRenameResponse::OnlyUnpreparedRenameSupported => {
                    // Fall back to the cell's own word boundaries; the cell text
                    // is what the user sees, and no server range is available.
                    let (range, kind) = cell_snapshot.surrounding_word(cell_offset, None);
                    if kind != Some(language::CharKind::Word) {
                        return Ok(None);
                    }
                    Some(RenameTarget::new(
                        cell_snapshot.anchor_before(range.start)
                            ..cell_snapshot.anchor_after(range.end),
                        None,
                    ))
                }
            })
        })
    }

    fn perform_rename(
        &self,
        buffer: &Entity<Buffer>,
        position: Anchor,
        new_name: String,
        language_server_id: Option<LanguageServerId>,
        cx: &mut App,
    ) -> Option<Task<Result<ProjectTransaction>>> {
        let notebook = self.notebook.upgrade()?;
        let (span, document_buffer) = notebook.update(cx, |notebook, cx| {
            notebook.rewrite_document(cx);
            let document = notebook.document.as_ref()?;
            let cell = notebook.cells.iter().find(|cell| cell.id == self.cell_id)?;
            Some((cell.span()?.clone(), document.buffer.clone()))
        })?;

        let cell_snapshot = buffer.read(cx).snapshot();
        let cell_offset = position.to_offset(&cell_snapshot);
        let document_offset = document_offset_for_cell(&span, cell_offset)?;
        let document_snapshot = document_buffer.read(cx).snapshot();
        let document_point = document_snapshot.offset_to_point(document_offset);
        let document_version_before = document_snapshot.version.clone();

        let project = self.project.upgrade()?;
        let request = project.update(cx, |project, cx| {
            project.perform_rename(
                document_buffer,
                document_point,
                new_name,
                language_server_id,
                cx,
            )
        });

        Some(cx.spawn(async move |cx| {
            let transaction = request.await?;
            notebook.update(cx, |notebook, cx| {
                notebook.apply_document_transaction(transaction, document_version_before, cx)
            })
        }))
    }
}

/// Serves code actions for one notebook cell by asking the language servers
/// about the virtual document and translating everything back into the cell.
struct CellCodeActionProvider {
    notebook: WeakEntity<NotebookLsp>,
    cell_id: CellId,
}

impl CodeActionProvider for CellCodeActionProvider {
    fn id(&self) -> Arc<str> {
        format!("notebook-cell-{}", self.cell_id).into()
    }

    fn code_actions(
        &self,
        buffer: &Entity<Buffer>,
        range: Range<Anchor>,
        _window: &mut Window,
        cx: &mut App,
    ) -> Task<Result<Vec<CodeAction>>> {
        let Some(notebook) = self.notebook.upgrade() else {
            return Task::ready(Ok(Vec::new()));
        };
        let Some((span, document_buffer, project)) = notebook.update(cx, |notebook, cx| {
            notebook.rewrite_document(cx);
            let document = notebook.document.as_ref()?;
            let cell = notebook.cells.iter().find(|cell| cell.id == self.cell_id)?;
            Some((
                cell.span()?.clone(),
                document.buffer.clone(),
                notebook.project.clone(),
            ))
        }) else {
            return Task::ready(Ok(Vec::new()));
        };

        let cell_snapshot = buffer.read(cx).snapshot();
        let cell_start = range.start.to_offset(&cell_snapshot);
        let cell_end = range.end.to_offset(&cell_snapshot);
        let (Some(document_start), Some(document_end)) = (
            document_offset_for_cell(&span, cell_start),
            document_offset_for_cell(&span, cell_end),
        ) else {
            return Task::ready(Ok(Vec::new()));
        };
        let document_range = document_buffer
            .read(cx)
            .anchor_before(document_start)
            ..document_buffer.read(cx).anchor_after(document_end);

        let request = project.update(cx, |project, cx| {
            project.code_actions(&document_buffer, document_range, None, cx)
        });

        let cell_buffer = buffer.clone();
        cx.spawn(async move |cx| {
            let Some(actions) = request.await? else {
                return Ok(Vec::new());
            };
            let (document_snapshot, cell_snapshot) = cx.update(|cx| {
                (
                    document_buffer.read(cx).snapshot(),
                    cell_buffer.read(cx).snapshot(),
                )
            });
            Ok(actions
                .into_iter()
                .filter_map(|action| {
                    let range =
                        translate_range(&action.range, &document_snapshot, &cell_snapshot, &span)?;
                    Some(CodeAction { range, ..action })
                })
                .collect())
        })
    }

    fn apply_code_action(
        &self,
        _buffer_handle: Entity<Buffer>,
        action: CodeAction,
        push_to_history: bool,
        _window: &mut Window,
        cx: &mut App,
    ) -> Task<Result<ProjectTransaction>> {
        let Some(notebook) = self.notebook.upgrade() else {
            return Task::ready(Ok(ProjectTransaction::default()));
        };
        let Some((document_buffer, project, document_version_before)) =
            notebook.update(cx, |notebook, cx| {
                let document = notebook.document.as_ref()?;
                Some((
                    document.buffer.clone(),
                    notebook.project.clone(),
                    document.buffer.read(cx).snapshot().version.clone(),
                ))
            })
        else {
            return Task::ready(Ok(ProjectTransaction::default()));
        };

        let request = project.update(cx, |project, cx| {
            project.apply_code_action(document_buffer, action, push_to_history, cx)
        });

        cx.spawn(async move |cx| {
            let transaction = request.await?;
            notebook.update(cx, |notebook, cx| {
                notebook.apply_document_transaction(transaction, document_version_before, cx)
            })
        })
    }
}

/// A cell buffer together with the snapshot and span needed to translate a
/// document location into that cell.
type CellTranslation = (CellId, Entity<Buffer>, BufferSnapshot, CellSpan);

/// Translates a location in the virtual document into the buffer of the cell
/// that owns it. Locations in other buffers (other files or libraries) are
/// returned unchanged.
fn remap_document_location(
    location: Location,
    document_buffer: &Entity<Buffer>,
    document: &BufferSnapshot,
    cells: &[CellTranslation],
) -> Option<Location> {
    if location.buffer != *document_buffer {
        return Some(location);
    }

    let start_offset = location.range.start.to_offset(document);
    let (_, cell_buffer, cell_snapshot, span) = cells
        .iter()
        .find(|(_, _, _, span)| {
            start_offset >= span.start_offset && start_offset <= span.end_offset
        })?;

    let start = translate_anchor(&location.range.start, document, cell_snapshot, span)?;
    // A definition may end on a separator or in the next cell; the target still
    // belongs to the cell that owns its start, so clamp the end to that cell.
    let end = translate_anchor(&location.range.end, document, cell_snapshot, span)
        .unwrap_or_else(|| cell_snapshot.anchor_after(span.end_offset - span.start_offset));
    Some(Location {
        buffer: cell_buffer.clone(),
        range: start..end,
    })
}

fn remap_location_link(
    link: LocationLink,
    document_buffer: &Entity<Buffer>,
    document: &BufferSnapshot,
    cells: &[CellTranslation],
) -> Option<LocationLink> {
    let origin = link
        .origin
        .and_then(|origin| remap_document_location(origin, document_buffer, document, cells));
    let target = remap_document_location(link.target, document_buffer, document, cells)?;
    Some(LocationLink { origin, target })
}

#[cfg(test)]
mod tests {
    use super::*;
    use gpui::TestAppContext;
    use project::FakeFs;
    use settings::SettingsStore;

    fn cell_id(name: &str) -> CellId {
        CellId::new(name).expect("test cell id should be valid")
    }

    fn cells(entries: &[(&str, &str)]) -> Vec<(CellId, String)> {
        entries
            .iter()
            .map(|(id, text)| (cell_id(id), text.to_string()))
            .collect()
    }

    #[test]
    fn builds_offsets_for_multiple_cells() {
        let (text, spans) = build_document(&cells(&[
            ("cell-one", "import numpy\nx = 1"),
            ("cell-two", "y = x + 1"),
        ]));

        assert_eq!(spans.len(), 2);
        assert_eq!(spans[0].start_offset, 0);
        assert_eq!(
            spans[0].end_offset,
            "import numpy\nx = 1".len(),
            "first cell ends at the end of its verbatim text"
        );
        assert_eq!(spans[0].start_row, 0);

        let second = &spans[1];
        let expected_start = spans[0].end_offset + 2;
        assert_eq!(second.start_offset, expected_start);
        assert_eq!(
            &text[second.start_offset..second.end_offset],
            "y = x + 1"
        );
        assert!(text[second.start_offset..].starts_with("y = x + 1"));
    }

    #[test]
    fn round_trips_offsets_across_cells() {
        let (text, spans) = build_document(&cells(&[
            ("cell-one", "a = 1\nb = 2"),
            ("cell-two", "c = a + b"),
        ]));

        for span in &spans {
            let length = span.end_offset - span.start_offset;
            for cell_offset in [0, length / 2, length] {
                let document_offset =
                    document_offset_for_cell(span, cell_offset).expect("in-cell offset");
                assert_eq!(
                    cell_offset_for_document(span, document_offset),
                    Some(cell_offset),
                    "round trip within {}",
                    span.cell_id
                );
                assert!(text.is_char_boundary(document_offset));
            }
        }

        let second = &spans[1];
        let document_offset = second.start_offset + 2;
        let (owner, cell_offset) =
            cell_at_document_offset(&spans, document_offset).expect("offset belongs to a cell");
        assert_eq!(owner.cell_id, second.cell_id);
        assert_eq!(cell_offset, 2);
    }

    #[test]
    fn separator_offsets_belong_to_no_cell() {
        let (_, spans) = build_document(&cells(&[
            ("cell-one", "first"),
            ("cell-two", "second"),
        ]));

        let separator_offset = spans[0].end_offset + 1;
        assert!(cell_offset_for_document(&spans[0], separator_offset).is_none());
        assert!(cell_offset_for_document(&spans[1], separator_offset).is_none());
        assert!(cell_at_document_offset(&spans, separator_offset).is_none());
    }

    #[test]
    fn empty_cell_has_zero_length_span() {
        let (_, spans) = build_document(&cells(&[("empty", ""), ("next", "value = 1")]));

        assert_eq!(spans[0].end_offset, spans[0].start_offset);
        assert_eq!(document_offset_for_cell(&spans[0], 0), Some(0));
        assert!(document_offset_for_cell(&spans[0], 1).is_none());
        assert_eq!(spans[1].start_offset, 2, "empty cell still emits a separator");
    }

    #[test]
    fn rebuild_after_middle_edit_moves_later_spans() {
        let (_, before) = build_document(&cells(&[
            ("cell-one", "a = 1"),
            ("cell-two", "b = 2"),
            ("cell-three", "c = 3"),
        ]));
        let (_, after) = build_document(&cells(&[
            ("cell-one", "a = 1"),
            ("cell-two", "b = 2\nb = b + 1\nb = b + 2"),
            ("cell-three", "c = 3"),
        ]));

        assert_eq!(before[0].start_offset, after[0].start_offset);
        assert_eq!(before[1].start_offset, after[1].start_offset);
        assert!(after[2].start_offset > before[2].start_offset);
        assert_eq!(
            after[2].start_offset - after[1].start_offset,
            after[1].end_offset - after[1].start_offset + 2
        );
    }

    #[test]
    fn inserting_and_removing_cells_updates_mapping() {
        let (_, three) = build_document(&cells(&[
            ("cell-one", "a"),
            ("cell-two", "b"),
            ("cell-three", "c"),
        ]));
        let (_, removed) = build_document(&cells(&[("cell-one", "a"), ("cell-three", "c")]));

        let three_three = three
            .iter()
            .find(|span| span.cell_id == cell_id("cell-three"))
            .expect("third cell");
        let removed_three = removed
            .iter()
            .find(|span| span.cell_id == cell_id("cell-three"))
            .expect("third cell");
        assert!(removed_three.start_offset < three_three.start_offset);

        let (_, inserted) = build_document(&cells(&[
            ("cell-one", "a"),
            ("cell-new", "middle"),
            ("cell-two", "b"),
        ]));
        let inserted_middle = inserted
            .iter()
            .find(|span| span.cell_id == cell_id("cell-new"))
            .expect("inserted cell");
        assert_eq!(inserted_middle.start_offset, three[0].end_offset + 2);
    }

    #[gpui::test]
    async fn virtual_document_concatenates_code_cells(cx: &mut TestAppContext) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.create_dir(Path::new("/notebooks")).await.unwrap();
        let project = Project::test(fs.clone(), [] as [&Path; 0], cx).await;

        let notebook_path = PathBuf::from("/notebooks/demo.ipynb");
        let notebook = cx.update(|cx| {
            cx.new(|cx| {
                NotebookLsp::new(
                    project.clone(),
                    notebook_path,
                    None,
                    cx,
                )
            })
        });

        let window = cx.add_empty_window();
        let mut cells = Vec::new();
        for (id, text) in [("cell-one", "x = 1"), ("cell-two", "y = x + 1")] {
            let buffer = window.update(|_window, cx| cx.new(|cx| Buffer::local(text, cx)));
            let editor = window.update(|window, cx| {
                cx.new(|cx| Editor::for_buffer(buffer.clone(), Some(project.clone()), window, cx))
            });
            cells.push(NotebookCell {
                id: cell_id(id),
                editor,
                buffer,
            });
        }

        notebook.update(cx, |notebook, cx| notebook.set_cells(cells, cx));

        let setup_task = notebook
            .update(cx, |notebook, _| notebook._setup_task.take())
            .expect("notebook LSP setup task should have been spawned");
        setup_task.await;
        cx.run_until_parked();

        let (document_text, buffer_text) = notebook.read_with(cx, |notebook, cx| {
            let document = notebook
                .document
                .as_ref()
                .expect("virtual document should have been created");
            (document.text.clone(), document.buffer.read(cx).text())
        });

        let (expected, _) = build_document(&[
            (cell_id("cell-one"), "x = 1".to_string()),
            (cell_id("cell-two"), "y = x + 1".to_string()),
        ]);
        assert_eq!(document_text, expected);
        assert_eq!(buffer_text, expected);
    }

    #[gpui::test]
    async fn virtual_document_mirrors_the_notebook_directory_under_zed_notebook(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.insert_tree(
            "/repo",
            serde_json::json!({ "sub": { "nested": { "analysis.ipynb": "" } } }),
        )
        .await;
        let project = Project::test(fs.clone(), [Path::new("/repo")], cx).await;
        let notebook = cx.update(|cx| {
            cx.new(|cx| {
                NotebookLsp::new(
                    project.clone(),
                    PathBuf::from("/repo/sub/nested/analysis.ipynb"),
                    None,
                    cx,
                )
            })
        });
        let setup_task = notebook
            .update(cx, |notebook, _| notebook._setup_task.take())
            .expect("notebook LSP setup task should have been spawned");
        setup_task.await;
        cx.run_until_parked();

        let document_path = Path::new("/repo/.zed/notebook/sub/nested/analysis.txt");
        assert!(fs.is_file(document_path).await);
        assert_eq!(
            fs.load(Path::new("/repo/.zed/notebook/.gitignore"))
                .await
                .unwrap(),
            "*\n"
        );

        cx.update(|_| drop(notebook));
        cx.run_until_parked();
        assert!(!fs.is_file(document_path).await);
        assert!(
            fs.is_file(Path::new("/repo/.zed/notebook/.gitignore"))
                .await
        );
    }

    #[gpui::test]
    async fn is_completion_trigger_uses_document_triggers_and_local_classifier(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.create_dir(Path::new("/notebooks")).await.unwrap();
        let project = Project::test(fs.clone(), [] as [&Path; 0], cx).await;

        let notebook = cx.update(|cx| {
            cx.new(|cx| {
                NotebookLsp::new(
                    project.clone(),
                    PathBuf::from("/notebooks/trigger.ipynb"),
                    None,
                    cx,
                )
            })
        });

        let window = cx.add_empty_window();
        let buffer = window.update(|_window, cx| cx.new(|cx| Buffer::local("os.", cx)));
        let editor = window.update(|window, cx| {
            cx.new(|cx| Editor::for_buffer(buffer.clone(), Some(project.clone()), window, cx))
        });

        notebook.update(cx, |notebook, cx| {
            notebook.set_cells(
                vec![NotebookCell {
                    id: cell_id("cell-trigger"),
                    editor: editor.clone(),
                    buffer: buffer.clone(),
                }],
                cx,
            )
        });

        let setup_task = notebook
            .update(cx, |notebook, _| notebook._setup_task.take())
            .expect("notebook LSP setup task should have been spawned");
        setup_task.await;
        cx.run_until_parked();

        // The virtual document is the buffer registered with the language
        // servers, so it is where server-supplied trigger characters live. The
        // cell buffer's own trigger set stays empty.
        notebook.update(cx, |notebook, cx| {
            let document = notebook
                .document
                .as_ref()
                .expect("virtual document should have been created");
            document.buffer.update(cx, |buffer, cx| {
                buffer.set_completion_triggers(
                    LanguageServerId(0),
                    [".".to_string()].into_iter().collect(),
                    cx,
                );
            });
        });

        let anchor = buffer.read_with(cx, |buffer, _cx| {
            let snapshot = buffer.snapshot();
            snapshot.anchor_after(snapshot.len())
        });
        let provider = editor
            .read_with(cx, |editor, _cx| editor.completion_provider())
            .expect("cell editor should have a completion provider installed");

        let dot = editor.update(cx, |_editor, cx| {
            provider.is_completion_trigger(&buffer, anchor, ".", false, cx)
        });
        assert!(dot, "`.` is a language server trigger character");

        let letter = editor.update(cx, |_editor, cx| {
            provider.is_completion_trigger(&buffer, anchor, "p", true, cx)
        });
        assert!(letter, "a word character triggers completion in words");

        let letter_without_words = editor.update(cx, |_editor, cx| {
            provider.is_completion_trigger(&buffer, anchor, "p", false, cx)
        });
        assert!(
            !letter_without_words,
            "a word character alone is not a server trigger"
        );

        let multi_character = editor.update(cx, |_editor, cx| {
            provider.is_completion_trigger(&buffer, anchor, "os", false, cx)
        });
        assert!(
            !multi_character,
            "only single characters can trigger completion"
        );
    }

    #[gpui::test]
    async fn hover_translation_round_trips_through_anchors(cx: &mut TestAppContext) {
        let cell_text = "import os\nos.path";
        let (document_text, spans) = build_document(&[
            (cell_id("cell-one"), "x = 1".to_string()),
            (cell_id("cell-two"), cell_text.to_string()),
        ]);
        let span = spans[1].clone();

        let document = cx.new(|cx| Buffer::local(document_text, cx));
        let cell = cx.new(|cx| Buffer::local(cell_text, cx));
        let document_snapshot = document.read_with(cx, |buffer, _cx| buffer.snapshot());
        let cell_snapshot = cell.read_with(cx, |buffer, _cx| buffer.snapshot());

        // A document offset inside the second cell, at the end of the `path`
        // identifier, as a hover request would supply it.
        let cell_offset = cell_text.find("path").expect("identifier") + "path".len();
        let document_offset = document_offset_for_cell(&span, cell_offset).expect("in-cell offset");
        let document_anchor = document_snapshot.anchor_after(document_offset);

        let translated =
            translate_anchor(&document_anchor, &document_snapshot, &cell_snapshot, &span)
                .expect("a document anchor inside a cell should translate into that cell");
        assert_eq!(translated.to_offset(&cell_snapshot), cell_offset);

        // A document offset on the separator between cells belongs to no cell,
        // so the hover range is dropped rather than mapped to a wrong position.
        let separator_anchor = document_snapshot.anchor_after(spans[0].end_offset + 1);
        assert!(
            translate_anchor(&separator_anchor, &document_snapshot, &cell_snapshot, &span).is_none(),
            "an anchor on a separator must not translate into a cell"
        );
    }

    #[gpui::test]
    async fn highlight_ranges_round_trip_through_the_cell(cx: &mut TestAppContext) {
        let cell_text = "total = 1\nprint(total)";
        let (document_text, spans) = build_document(&[
            (cell_id("cell-one"), "x = 1".to_string()),
            (cell_id("cell-two"), cell_text.to_string()),
        ]);
        let span = spans[1].clone();

        let document = cx.new(|cx| Buffer::local(document_text, cx));
        let cell = cx.new(|cx| Buffer::local(cell_text, cx));
        let document_snapshot = document.read_with(cx, |buffer, _cx| buffer.snapshot());
        let cell_snapshot = cell.read_with(cx, |buffer, _cx| buffer.snapshot());

        let identifiers = [
            cell_text.find("total").expect("definition"),
            cell_text.rfind("total").expect("reference"),
        ];
        for cell_offset in [0, identifiers[0], identifiers[1], cell_text.len()] {
            let document_offset =
                document_offset_for_cell(&span, cell_offset).expect("in-cell offset");
            assert_eq!(
                cell_offset_for_document(&span, document_offset),
                Some(cell_offset)
            );

            let document_anchor = document_snapshot.anchor_after(document_offset);
            let translated =
                translate_anchor(&document_anchor, &document_snapshot, &cell_snapshot, &span)
                    .expect("an anchor inside the cell should translate");
            assert_eq!(translated.to_offset(&cell_snapshot), cell_offset);

            let cell_anchor = cell_snapshot.anchor_after(cell_offset);
            assert_eq!(
                document_offset_for_cell(&span, cell_anchor.to_offset(&cell_snapshot)),
                Some(document_offset)
            );
        }

        let start_offset = span.start_offset + identifiers[1];
        let end_offset = start_offset + "total".len();
        let highlight = DocumentHighlight {
            range: document_snapshot.anchor_before(start_offset)
                ..document_snapshot.anchor_after(end_offset),
            kind: lsp::DocumentHighlightKind::TEXT,
        };
        let start = translate_anchor(&highlight.range.start, &document_snapshot, &cell_snapshot, &span)
            .expect("highlight start");
        let end = translate_anchor(&highlight.range.end, &document_snapshot, &cell_snapshot, &span)
            .expect("highlight end");
        assert_eq!(start.to_offset(&cell_snapshot), identifiers[1]);
        assert_eq!(end.to_offset(&cell_snapshot), identifiers[1] + "total".len());
    }

    #[gpui::test]
    async fn definition_targets_inside_document_remap_to_their_cell(cx: &mut TestAppContext) {
        let first_text = "x = 1";
        let second_text = "y = 2\nprint(y)";
        let (document_text, spans) = build_document(&[
            (cell_id("cell-one"), first_text.to_string()),
            (cell_id("cell-two"), second_text.to_string()),
        ]);

        let document = cx.new(|cx| Buffer::local(document_text, cx));
        let first_cell = cx.new(|cx| Buffer::local(first_text, cx));
        let second_cell = cx.new(|cx| Buffer::local(second_text, cx));
        let external = cx.new(|cx| Buffer::local("y = 2", cx));

        let (document_snapshot, first_snapshot, second_snapshot, external_snapshot) =
            cx.update(|cx| {
                (
                    document.read(cx).snapshot(),
                    first_cell.read(cx).snapshot(),
                    second_cell.read(cx).snapshot(),
                    external.read(cx).snapshot(),
                )
            });
        let cells = vec![
            (
                cell_id("cell-one"),
                first_cell.clone(),
                first_snapshot.clone(),
                spans[0].clone(),
            ),
            (
                cell_id("cell-two"),
                second_cell.clone(),
                second_snapshot.clone(),
                spans[1].clone(),
            ),
        ];

        let target_start = spans[1].start_offset;
        let origin_start = spans[0].start_offset;
        let link = LocationLink {
            origin: Some(Location {
                buffer: document.clone(),
                range: document_snapshot.anchor_before(origin_start)
                    ..document_snapshot.anchor_after(origin_start + 1),
            }),
            target: Location {
                buffer: document.clone(),
                range: document_snapshot.anchor_before(target_start)
                    ..document_snapshot.anchor_after(target_start + 1),
            },
        };

        let remapped = remap_location_link(link, &document, &document_snapshot, &cells)
            .expect("a location inside the document should be remapped");
        assert_eq!(remapped.target.buffer, second_cell);
        assert_eq!(
            remapped.target.range.start.to_offset(&second_snapshot),
            0
        );
        assert_eq!(
            remapped.target.range.end.to_offset(&second_snapshot),
            1
        );
        let origin = remapped.origin.expect("origin should be remapped too");
        assert_eq!(origin.buffer, first_cell);
        assert_eq!(origin.range.start.to_offset(&first_snapshot), 0);

        let external_link = LocationLink {
            origin: None,
            target: Location {
                buffer: external.clone(),
                range: external_snapshot.anchor_before(0)..external_snapshot.anchor_after(1),
            },
        };
        let unchanged =
            remap_location_link(external_link, &document, &document_snapshot, &cells)
                .expect("locations outside the document are kept as they are");
        assert_eq!(unchanged.target.buffer, external);
        assert_eq!(unchanged.target.range.start.to_offset(&external_snapshot), 0);

        let separator_start = spans[0].end_offset + 1;
        let separator_link = LocationLink {
            origin: None,
            target: Location {
                buffer: document.clone(),
                range: document_snapshot.anchor_before(separator_start)
                    ..document_snapshot.anchor_after(separator_start + 1),
            },
        };
        assert!(
            remap_location_link(separator_link, &document, &document_snapshot, &cells).is_none(),
            "a target on a separator belongs to no cell and is dropped"
        );
    }

    fn test_python_language() -> Arc<Language> {
        Arc::new(Language::new(
            language::LanguageConfig {
                name: "Python".into(),
                matcher: language::LanguageMatcher {
                    path_suffixes: vec!["py".into()],
                    ..Default::default()
                }
                .into(),
                ..Default::default()
            },
            None,
        ))
    }

    /// Sets up a notebook whose virtual document is served by a fake language
    /// server, returning the cells and the channel that yields every server
    /// instance that starts.
    async fn notebook_with_fake_lsp(
        name: &str,
        entries: &[(&str, &str)],
        adapter: language::FakeLspAdapter,
        cx: &mut TestAppContext,
    ) -> (
        Entity<NotebookLsp>,
        Vec<Entity<Buffer>>,
        Vec<Entity<Editor>>,
        futures::channel::mpsc::UnboundedReceiver<lsp::FakeLanguageServer>,
    ) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.create_dir(Path::new("/notebooks")).await.unwrap();
        let project = Project::test(fs.clone(), [] as [&Path; 0], cx).await;
        let languages = project.read_with(cx, |project, _| project.languages().clone());
        let python = test_python_language();
        languages.add(python.clone());
        let fake_servers = languages.register_fake_lsp("Python", adapter);

        let notebook = cx.update(|cx| {
            cx.new(|cx| {
                NotebookLsp::new(
                    project.clone(),
                    PathBuf::from(format!("/notebooks/{name}.ipynb")),
                    Some(python.clone()),
                    cx,
                )
            })
        });

        let window = cx.add_empty_window();
        let mut buffers = Vec::new();
        let mut editors = Vec::new();
        let mut cells = Vec::new();
        for (id, text) in entries {
            let buffer = window.update(|_window, cx| cx.new(|cx| Buffer::local(*text, cx)));
            let editor = window.update(|window, cx| {
                cx.new(|cx| Editor::for_buffer(buffer.clone(), Some(project.clone()), window, cx))
            });
            buffers.push(buffer.clone());
            editors.push(editor.clone());
            cells.push(NotebookCell {
                id: cell_id(id),
                editor,
                buffer,
            });
        }

        notebook.update(cx, |notebook, cx| notebook.set_cells(cells, cx));
        let setup_task = notebook
            .update(cx, |notebook, _| notebook._setup_task.take())
            .expect("notebook LSP setup task should have been spawned");
        setup_task.await;
        cx.run_until_parked();

        (notebook, buffers, editors, fake_servers)
    }

    #[gpui::test]
    async fn references_are_resolved_in_the_document_and_remapped_to_cells(
        cx: &mut TestAppContext,
    ) {
        let (notebook, buffers, editors, mut fake_servers) = notebook_with_fake_lsp(
            "references",
            &[("cell-one", "alpha = 1"), ("cell-two", "beta = alpha + 1")],
            language::FakeLspAdapter {
                capabilities: lsp::ServerCapabilities {
                    references_provider: Some(lsp::OneOf::Left(true)),
                    ..Default::default()
                },
                initializer: Some(Box::new(|server| {
                    server.set_request_handler::<lsp::request::References, _, _>(
                        |params, _| async move {
                            let uri = params.text_document_position.text_document.uri;
                            let location = |row, start, end| lsp::Location {
                                uri: uri.clone(),
                                range: lsp::Range::new(
                                    lsp::Position::new(row, start),
                                    lsp::Position::new(row, end),
                                ),
                            };
                            // The virtual document is "alpha = 1\n\nbeta = alpha + 1":
                            // one reference in each cell plus one on the separator,
                            // which must be dropped.
                            Ok(Some(vec![
                                location(2, 7, 12),
                                location(0, 0, 5),
                                location(1, 0, 0),
                            ]))
                        },
                    );
                })),
                ..language::FakeLspAdapter::default()
            },
            cx,
        )
        .await;

        let fake_server = fake_servers
            .next()
            .await
            .expect("the virtual document should start the language server");
        cx.run_until_parked();

        let provider = editors[1]
            .read_with(cx, |editor, _| editor.semantics_provider())
            .expect("the cell editor should have a semantics provider");
        let cell_anchor = buffers[1].read_with(cx, |buffer, _| {
            buffer.snapshot().anchor_after("beta = ".len())
        });
        let locations = cx
            .update(|cx| provider.references(&buffers[1], cell_anchor, cx))
            .expect("the notebook should resolve references itself")
            .await
            .expect("the references request should succeed")
            .expect("the fake server should answer");

        assert_eq!(
            locations.len(),
            2,
            "the separator reference is dropped and the in-cell ones are kept: {locations:?}"
        );
        assert_eq!(locations[0].buffer, buffers[1]);
        assert_eq!(
            locations[0]
                .range
                .start
                .to_offset(&buffers[1].read_with(cx, |buffer, _| buffer.snapshot())),
            "beta = ".len()
        );
        assert_eq!(locations[1].buffer, buffers[0]);
        assert_eq!(
            locations[1]
                .range
                .start
                .to_offset(&buffers[0].read_with(cx, |buffer, _| buffer.snapshot())),
            0
        );

        drop(fake_server);
        drop(notebook);
        cx.run_until_parked();
    }

    #[gpui::test]
    async fn applying_a_code_action_edits_the_owning_cell_and_keeps_the_document_consistent(
        cx: &mut TestAppContext,
    ) {
        let (notebook, buffers, editors, mut fake_servers) = notebook_with_fake_lsp(
            "code-action",
            &[("cell-one", "alpha = 1"), ("cell-two", "beta = alpha + 1")],
            language::FakeLspAdapter {
                capabilities: lsp::ServerCapabilities {
                    code_action_provider: Some(lsp::CodeActionProviderCapability::Simple(true)),
                    ..Default::default()
                },
                initializer: Some(Box::new(|server| {
                    server.set_request_handler::<lsp::request::CodeActionRequest, _, _>(
                        |params, _| async move {
                            let uri = params.text_document.uri;
                            Ok(Some(vec![lsp::CodeActionOrCommand::CodeAction(
                                lsp::CodeAction {
                                    title: "Organize imports".to_string(),
                                    kind: Some(lsp::CodeActionKind::SOURCE_ORGANIZE_IMPORTS),
                                    edit: Some(lsp::WorkspaceEdit::new(
                                        [(
                                            uri,
                                            vec![lsp::TextEdit::new(
                                                lsp::Range::new(
                                                    lsp::Position::new(2, 7),
                                                    lsp::Position::new(2, 12),
                                                ),
                                                "sum".to_string(),
                                            )],
                                        )]
                                        .into_iter()
                                        .collect(),
                                    )),
                                    ..Default::default()
                                },
                            )]))
                        },
                    );
                })),
                ..language::FakeLspAdapter::default()
            },
            cx,
        )
        .await;

        let _fake_server = fake_servers
            .next()
            .await
            .expect("the virtual document should start the language server");
        cx.executor().run_until_parked();

        let provider = editors[1]
            .read_with(cx, |editor, _| {
                editor
                    .code_action_providers()
                    .iter()
                    .find(|provider| provider.id().starts_with("notebook-cell-"))
                    .cloned()
                    .expect("the cell should have a notebook code action provider")
            });

        let cell_snapshot = buffers[1].read_with(cx, |buffer, _| buffer.snapshot());
        let cell_range = cell_snapshot.anchor_before(0)
            ..cell_snapshot.anchor_after(cell_snapshot.len());
        let (span_before, document_buffer) = notebook.read_with(cx, |notebook, _| {
            (
                notebook.cells[1].span().cloned().expect("cell span"),
                notebook
                    .document
                    .as_ref()
                    .expect("virtual document")
                    .buffer
                    .clone(),
            )
        });

        let window = cx.add_empty_window();
        let actions = window
            .update(|window, cx| {
                provider.code_actions(&buffers[1], cell_range, window, cx)
            })
            .await
            .expect("the code action request should succeed");
        assert!(!actions.is_empty(), "expected at least one action");
        let action = actions
            .into_iter()
            .find(|action| action.lsp_action.title() == "Organize imports")
            .expect("the fake server's action should be returned");

        let transaction = window
            .update(|window, cx| {
                provider.apply_code_action(buffers[1].clone(), action, true, window, cx)
            })
            .await
            .expect("the code action should apply without error");

        assert!(
            transaction.0.contains_key(&buffers[1]),
            "the returned transaction must report the edited cell, not the virtual document"
        );
        assert!(
            !transaction.0.contains_key(&document_buffer),
            "the virtual document must never leak into the returned transaction"
        );
        window.update(|_, cx| {
            buffers[1].read_with(cx, |buffer, _| {
                assert_eq!(buffer.text(), "beta = sum + 1");
            });
            buffers[0].read_with(cx, |buffer, _| {
                assert_eq!(buffer.text(), "alpha = 1", "other cells stay untouched");
            });
            buffers[1].read_with(cx, |buffer, _| {
                assert!(buffer.is_dirty(), "the edited cell becomes dirty");
            });
        });

        let (cached_text, document_text, span_after) =
            window.update(|_, cx| notebook.read_with(cx, |notebook, cx| {
                let document = notebook.document.as_ref().expect("virtual document");
                (
                    document.text.clone(),
                    document.buffer.read(cx).text(),
                    notebook.cells[1].span().cloned().expect("cell span"),
                )
            }));
        let (expected, _) = build_document(&[
            (cell_id("cell-one"), "alpha = 1".to_string()),
            (cell_id("cell-two"), "beta = sum + 1".to_string()),
        ]);
        assert_eq!(cached_text, expected);
        assert_eq!(
            document_text, expected,
            "the virtual document mirrors the cells after the action"
        );
        assert_ne!(span_before.end_offset, span_after.end_offset);

        window.update(|_, cx| {
            buffers[1].update(cx, |buffer, cx| buffer.undo(cx));
            buffers[1].read_with(cx, |buffer, _| {
                assert_eq!(
                    buffer.text(),
                    "beta = alpha + 1",
                    "a code action applied from the cell is undoable"
                );
            });
        });
    }

    #[gpui::test]
    async fn setting_python_interpreter_before_document_is_remembered(
        cx: &mut TestAppContext,
    ) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
        });

        let fs = FakeFs::new(cx.executor());
        fs.create_dir(Path::new("/notebooks")).await.unwrap();
        let project = Project::test(fs.clone(), [] as [&Path; 0], cx).await;

        let notebook = cx.new(|cx| {
            NotebookLsp::new(
                project,
                PathBuf::from("/notebooks/env.ipynb"),
                None,
                cx,
            )
        });

        // Applying before the virtual document exists must not panic, and the
        // interpreter must still be recorded for when the document shows up.
        let interpreter = PathBuf::from("/project/.venv/bin/python");
        notebook.update(cx, |notebook, cx| {
            notebook.set_python_interpreter(interpreter.clone(), cx);
        });

        notebook.read_with(cx, |notebook, _| {
            assert_eq!(notebook.python_interpreter.as_ref(), Some(&interpreter));
            assert!(notebook.document.is_none());
        });
    }

    #[test]
    fn neutralizes_line_magics_shell_escapes_and_help() {
        let cell = "%matplotlib inline\n!ls -la\nx = 1\ndf.head??\n";
        let neutralized = neutralize_magics(cell);

        assert_eq!(
            neutralized.len(),
            cell.len(),
            "neutralization must preserve the byte length"
        );
        assert_eq!(
            neutralized.matches('\n').count(),
            cell.matches('\n').count(),
            "neutralization must preserve the line count"
        );

        let lines: Vec<&str> = neutralized.split('\n').collect();
        assert!(lines[0].starts_with('#'), "line magic became a comment");
        assert!(!lines[0].contains('%'), "the magic marker is gone");
        assert!(lines[1].starts_with('#'), "shell escape became a comment");
        assert!(!lines[1].contains('!'), "the shell marker is gone");
        assert_eq!(lines[2], "x = 1", "plain code is untouched");
        assert_eq!(lines[3], "df.head  ", "the trailing help characters are blanked");
    }

    #[test]
    fn cell_magic_turns_the_whole_cell_into_comments() {
        let cell = "%%time\nimport time\ntime.sleep(1)";
        let neutralized = neutralize_magics(cell);

        assert_eq!(neutralized.len(), cell.len());
        for line in neutralized.split('\n') {
            assert!(
                line.is_empty() || line.starts_with('#'),
                "line {line:?} should be a comment"
            );
        }
    }

    #[test]
    fn neutralization_keeps_offsets_valid_for_multibyte_magics() {
        let cell = "%%time\nnaïve = 'café'";
        let neutralized = neutralize_magics(cell);

        assert_eq!(neutralized.len(), cell.len());
        assert!(neutralized.is_ascii(), "replacement is all ASCII so every offset is a boundary");
        for offset in 0..=neutralized.len() {
            assert!(neutralized.is_char_boundary(offset));
        }
    }

    #[test]
    fn magic_neutralization_preserves_cell_position_mapping() {
        let cells = cells(&[("cell-magic", "%time x = 1\nprint(x)")]);
        let (text, spans) = build_document(&cells);
        let span = &spans[0];

        assert_eq!(
            span.end_offset - span.start_offset,
            cells[0].1.len(),
            "a neutralized cell keeps its byte length"
        );

        let cell_offset = cells[0].1.find("print").expect("identifier");
        let document_offset =
            document_offset_for_cell(span, cell_offset).expect("in-cell offset");
        assert_eq!(
            cell_offset_for_document(span, document_offset),
            Some(cell_offset)
        );
        assert!(text.is_char_boundary(document_offset));
    }

    #[test]
    fn converts_code_points_to_bytes_for_emoji_and_accents() {
        let text = "a😀é";
        assert_eq!(code_point_to_byte(text, 0), Some(0));
        assert_eq!(code_point_to_byte(text, 1), Some(1), "start of the emoji");
        assert_eq!(code_point_to_byte(text, 2), Some(5), "start of the accented char");
        assert_eq!(code_point_to_byte(text, 3), Some(7), "one past the end");
        assert_eq!(code_point_to_byte(text, 4), None, "beyond the string");

        assert_eq!(code_point_to_byte("", 0), Some(0));
        assert_eq!(code_point_to_byte("", 1), None);
    }

    fn custom_completion(snapshot: &BufferSnapshot, label: &str) -> Completion {
        Completion {
            replace_range: snapshot.anchor_before(0)..snapshot.anchor_before(0),
            new_text: label.to_string(),
            label: CodeLabel::plain(label.to_string(), None),
            documentation: None,
            source: CompletionSource::Custom,
            icon_path: None,
            icon_color: None,
            match_start: None,
            snippet_deduplication_key: None,
            insert_text_mode: None,
            confirm: None,
            group: None,
        }
    }

    #[gpui::test]
    async fn merges_kernel_completions_after_lsp_without_duplicates(cx: &mut TestAppContext) {
        let text = "frame.😀col";
        let buffer = cx.new(|cx| Buffer::local(text, cx));
        let snapshot = buffer.read_with(cx, |buffer, _| buffer.snapshot());

        let mut responses = vec![CompletionResponse {
            completions: vec![custom_completion(&snapshot, "shape")],
            display_options: CompletionDisplayOptions::default(),
            is_incomplete: false,
        }];

        let cursor_start = text.chars().count() - "col".chars().count();
        let cursor_end = text.chars().count();
        merge_kernel_completions(
            &mut responses,
            &snapshot,
            KernelCompletion {
                matches: vec![
                    "shape".to_string(),
                    "columns".to_string(),
                    "columns".to_string(),
                    "index".to_string(),
                ],
                cursor_start,
                cursor_end,
            },
        );

        assert_eq!(responses.len(), 2, "kernel completions are a separate response");
        let labels: Vec<String> = responses
            .iter()
            .flat_map(|response| response.completions.iter())
            .map(|completion| completion.label.text.clone())
            .collect();
        assert_eq!(
            labels,
            vec!["shape", "columns", "index"],
            "LSP stays first and the duplicate is dropped"
        );

        let columns = responses[1]
            .completions
            .iter()
            .find(|completion| completion.label.text == "columns")
            .expect("columns completion");
        assert_eq!(
            columns.replace_range.start.to_offset(&snapshot),
            code_point_to_byte(text, cursor_start).expect("start"),
        );
        assert_eq!(
            columns.replace_range.end.to_offset(&snapshot),
            text.len(),
        );
    }

    #[gpui::test]
    async fn kernel_cursor_outside_the_cell_is_ignored(cx: &mut TestAppContext) {
        let buffer = cx.new(|cx| Buffer::local("x = 1", cx));
        let snapshot = buffer.read_with(cx, |buffer, _| buffer.snapshot());

        let mut responses = Vec::new();
        merge_kernel_completions(
            &mut responses,
            &snapshot,
            KernelCompletion {
                matches: vec!["y".to_string()],
                cursor_start: 0,
                cursor_end: 99,
            },
        );
        assert!(responses.is_empty());
    }

    #[gpui::test]
    async fn rename_and_code_action_ranges_translate_back_into_the_owning_cell(
        cx: &mut TestAppContext,
    ) {
        let first_text = "x = 1";
        let second_text = "print(x)";
        let (document_text, spans) = build_document(&[
            (cell_id("cell-one"), first_text.to_string()),
            (cell_id("cell-two"), second_text.to_string()),
        ]);

        let document = cx.new(|cx| Buffer::local(document_text, cx));
        let cell = cx.new(|cx| Buffer::local(second_text, cx));
        let document_snapshot = document.read_with(cx, |buffer, _| buffer.snapshot());
        let cell_snapshot = cell.read_with(cx, |buffer, _| buffer.snapshot());

        let cell_offset = second_text.find('x').expect("reference");
        let document_offset =
            document_offset_for_cell(&spans[1], cell_offset).expect("in-cell offset");
        let document_range = document_snapshot.anchor_before(document_offset)
            ..document_snapshot.anchor_after(document_offset + 1);

        let translated = translate_range(
            &document_range,
            &document_snapshot,
            &cell_snapshot,
            &spans[1],
        )
        .expect("a range inside the cell translates");
        assert_eq!(translated.start.to_offset(&cell_snapshot), cell_offset);
        assert_eq!(translated.end.to_offset(&cell_snapshot), cell_offset + 1);

        // A range that reaches into another cell must not be silently clipped.
        let crossing = document_snapshot.anchor_before(spans[0].start_offset)
            ..document_snapshot.anchor_after(document_offset + 1);
        assert!(translate_range(&crossing, &document_snapshot, &cell_snapshot, &spans[1]).is_none());
    }

    fn document_transaction_editing(
        buffer: &Entity<Buffer>,
        edits: Vec<(Range<usize>, String)>,
        cx: &mut TestAppContext,
    ) -> language::Transaction {
        buffer
            .update(cx, |buffer, cx| {
                buffer.finalize_last_transaction();
                buffer.start_transaction();
                buffer.edit(edits, None, cx);
                let transaction_id = buffer.end_transaction(cx)?;
                buffer.finalize_last_transaction();
                buffer.get_transaction(transaction_id).cloned()
            })
            .expect("the document edit should produce a transaction")
    }

    async fn notebook_with_cells(
        paths: &str,
        entries: &[(&str, &str)],
        cx: &mut TestAppContext,
    ) -> (Entity<NotebookLsp>, Vec<Entity<Buffer>>) {
        cx.update(|cx| {
            let settings_store = SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
            editor::init(cx);
        });

        let fs = FakeFs::new(cx.executor());
        fs.create_dir(Path::new("/notebooks")).await.unwrap();
        let project = Project::test(fs.clone(), [] as [&Path; 0], cx).await;
        let notebook = cx.update(|cx| {
            cx.new(|cx| {
                NotebookLsp::new(
                    project.clone(),
                    PathBuf::from(format!("/notebooks/{paths}.ipynb")),
                    None,
                    cx,
                )
            })
        });

        let window = cx.add_empty_window();
        let mut buffers = Vec::new();
        let mut cells = Vec::new();
        for (id, text) in entries {
            let buffer = window.update(|_window, cx| cx.new(|cx| Buffer::local(*text, cx)));
            let editor = window.update(|window, cx| {
                cx.new(|cx| Editor::for_buffer(buffer.clone(), Some(project.clone()), window, cx))
            });
            buffers.push(buffer.clone());
            cells.push(NotebookCell {
                id: cell_id(id),
                editor,
                buffer,
            });
        }

        notebook.update(cx, |notebook, cx| notebook.set_cells(cells, cx));
        let setup_task = notebook
            .update(cx, |notebook, _| notebook._setup_task.take())
            .expect("notebook LSP setup task should have been spawned");
        setup_task.await;
        cx.run_until_parked();

        (notebook, buffers)
    }

    #[gpui::test]
    async fn applies_multi_cell_rename_to_cells_and_resynchronizes_document(
        cx: &mut TestAppContext,
    ) {
        let (notebook, buffers) = notebook_with_cells(
            "rename",
            &[
                ("cell-one", "x = 1"),
                ("cell-two", "y = x + 1"),
                ("cell-three", "z = 3"),
            ],
            cx,
        )
        .await;

        let (document_buffer, version_before, spans) = notebook.read_with(cx, |notebook, cx| {
            let document = notebook.document.as_ref().expect("virtual document");
            (
                document.buffer.clone(),
                document.buffer.read(cx).snapshot().version.clone(),
                notebook
                    .cells
                    .iter()
                    .filter_map(|cell| cell.span().cloned())
                    .collect::<Vec<_>>(),
            )
        });

        let reference_offset = spans[1].start_offset + "y = ".len();
        let document_transaction = document_transaction_editing(
            &document_buffer,
            vec![
                (spans[0].start_offset..spans[0].start_offset + 1, "total".to_string()),
                (reference_offset..reference_offset + 1, "total".to_string()),
            ],
            cx,
        );
        let mut transaction = ProjectTransaction::default();
        transaction.0.insert(document_buffer, document_transaction);

        let returned = notebook
            .update(cx, |notebook, cx| {
                notebook.apply_document_transaction(transaction, version_before, cx)
            })
            .expect("the document transaction should be applied to cells");

        assert_eq!(returned.0.len(), 2, "only the edited cells are reported");
        assert!(returned.0.contains_key(&buffers[0]));
        assert!(returned.0.contains_key(&buffers[1]));
        assert!(!returned.0.contains_key(&buffers[2]));

        assert_eq!(buffers[0].read_with(cx, |buffer, _| buffer.text()), "total = 1");
        assert_eq!(
            buffers[1].read_with(cx, |buffer, _| buffer.text()),
            "y = total + 1"
        );
        assert_eq!(buffers[2].read_with(cx, |buffer, _| buffer.text()), "z = 3");

        let (cached_text, buffer_text) = notebook.read_with(cx, |notebook, cx| {
            let document = notebook.document.as_ref().expect("virtual document");
            (document.text.clone(), document.buffer.read(cx).text())
        });
        let (expected, _) = build_document(&[
            (cell_id("cell-one"), "total = 1".to_string()),
            (cell_id("cell-two"), "y = total + 1".to_string()),
            (cell_id("cell-three"), "z = 3".to_string()),
        ]);
        assert_eq!(cached_text, expected);
        assert_eq!(buffer_text, expected, "the document mirrors the cells once");

        buffers[0].update(cx, |buffer, cx| buffer.undo(cx));
        assert_eq!(buffers[0].read_with(cx, |buffer, _| buffer.text()), "x = 1");
    }

    #[gpui::test]
    async fn formatting_one_cell_leaves_neighboring_cells_untouched(cx: &mut TestAppContext) {
        let (notebook, buffers) = notebook_with_cells(
            "format",
            &[
                ("cell-one", "x = 1"),
                ("cell-two", "y =  x + 1"),
                ("cell-three", "z = 3"),
            ],
            cx,
        )
        .await;

        let (document_buffer, version_before, span) =
            notebook.read_with(cx, |notebook, cx| {
                let document = notebook.document.as_ref().expect("virtual document");
                let span = notebook.cells[1].span().cloned().expect("second cell span");
                (
                    document.buffer.clone(),
                    document.buffer.read(cx).snapshot().version.clone(),
                    span,
                )
            });

        let extra_space = span.start_offset + "y = ".len();
        let document_transaction = document_transaction_editing(
            &document_buffer,
            vec![(extra_space..extra_space + 1, String::new())],
            cx,
        );
        let mut transaction = ProjectTransaction::default();
        transaction.0.insert(document_buffer, document_transaction);

        let returned = notebook
            .update(cx, |notebook, cx| {
                notebook.apply_document_transaction(transaction, version_before, cx)
            })
            .expect("the formatting edit should be applied to the cell");

        assert_eq!(returned.0.len(), 1);
        assert_eq!(
            buffers[1].read_with(cx, |buffer, _| buffer.text()),
            "y = x + 1"
        );
        assert_eq!(buffers[0].read_with(cx, |buffer, _| buffer.text()), "x = 1");
        assert_eq!(buffers[2].read_with(cx, |buffer, _| buffer.text()), "z = 3");
    }

    struct FakeKernelCompleter {
        completion: Option<KernelCompletion>,
        busy: bool,
        slow: bool,
        executor: gpui::BackgroundExecutor,
    }

    impl KernelCompleter for FakeKernelCompleter {
        fn complete(
            &self,
            _code: String,
            _cursor_pos: usize,
        ) -> Task<Result<Option<KernelCompletion>>> {
            if self.busy {
                return Task::ready(Ok(None));
            }
            if self.slow {
                return self
                    .executor
                    .spawn(std::future::pending::<Result<Option<KernelCompletion>>>());
            }
            Task::ready(Ok(self.completion.clone()))
        }

        fn inspect(
            &self,
            _code: String,
            _cursor_pos: usize,
        ) -> Task<Result<Option<KernelInspection>>> {
            Task::ready(Ok(None))
        }
    }

    #[gpui::test]
    async fn ready_kernel_completion_is_returned(cx: &mut TestAppContext) {
        let completer = FakeKernelCompleter {
            completion: Some(KernelCompletion {
                matches: vec!["columns".to_string()],
                cursor_start: 0,
                cursor_end: 1,
            }),
            busy: false,
            slow: false,
            executor: cx.executor(),
        };
        let completion =
            await_kernel_completion(completer.complete("x".to_string(), 1), &cx.executor()).await;
        assert_eq!(
            completion,
            Some(KernelCompletion {
                matches: vec!["columns".to_string()],
                cursor_start: 0,
                cursor_end: 1,
            })
        );
    }

    #[gpui::test]
    async fn busy_kernel_yields_no_kernel_completion(cx: &mut TestAppContext) {
        let completer = FakeKernelCompleter {
            completion: Some(KernelCompletion {
                matches: vec!["columns".to_string()],
                cursor_start: 0,
                cursor_end: 1,
            }),
            busy: true,
            slow: false,
            executor: cx.executor(),
        };
        let completion =
            await_kernel_completion(completer.complete("x".to_string(), 1), &cx.executor()).await;
        assert_eq!(completion, None);
    }

    #[gpui::test]
    async fn slow_kernel_times_out(cx: &mut TestAppContext) {
        let completer = FakeKernelCompleter {
            completion: None,
            busy: false,
            slow: true,
            executor: cx.executor(),
        };
        let completion =
            await_kernel_completion(completer.complete("x".to_string(), 1), &cx.executor()).await;
        assert_eq!(completion, None, "a slow kernel must not block completion");
    }

    #[gpui::test]
    async fn failing_kernel_is_ignored(cx: &mut TestAppContext) {
        let failed: Task<Result<Option<KernelCompletion>>> =
            Task::ready(Err(anyhow::anyhow!("kernel exploded")));
        let completion = await_kernel_completion(failed, &cx.executor()).await;
        assert_eq!(completion, None);
    }

    #[gpui::test]
    async fn kernel_inspection_becomes_markdown_hover(_cx: &mut TestAppContext) {
        let inspection = KernelInspection {
            kind: KernelInspectionKind::Markdown,
            text: "**frame** — a DataFrame".to_string(),
        };
        let hover = hover_from_inspection(inspection.clone());
        assert_eq!(hover.contents.len(), 1);
        assert_eq!(hover.contents[0].text, inspection.text);
        assert_eq!(hover.contents[0].kind, HoverBlockKind::Markdown);

        let inspection = KernelInspection {
            kind: KernelInspectionKind::PlainText,
            text: "int".to_string(),
        };
        let hover = hover_from_inspection(inspection);
        assert_eq!(hover.contents[0].kind, HoverBlockKind::PlainText);
    }
}
