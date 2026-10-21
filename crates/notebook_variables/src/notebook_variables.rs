//! Notebook variable introspection and the variables/data-viewer panel.
//!
//! The kernel side of this crate is deliberately a single, printable Python
//! snippet. Keeping it as a constant makes it testable and lets the REPL send
//! the exact same request after every execution and from the refresh action.

use anyhow::{Context as _, Result, anyhow};
use gpui::{Context, EventEmitter, Window, actions, px};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use ui::{
    Button, IconButton, IconName, Label, LabelSize, Styled, Tooltip, h_flex, prelude::*, v_flex,
};

pub const VARIABLES_PAGE_SIZE: usize = 200;
pub const VARIABLES_REQUEST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(10);

pub const VARIABLES_PANEL_HORIZONTAL_PADDING: f32 = 16.0;
pub const VARIABLES_HEADER_TITLE_WIDTH: f32 = 72.0;
pub const VARIABLES_HEADER_REFRESH_BUTTON_WIDTH: f32 = 28.0;
pub const VARIABLES_HEADER_MINIMUM_GAP: f32 = 8.0;
pub const VARIABLES_RIGHT_MIN_WIDTH: f32 = 200.0;
pub const VARIABLES_PANEL_VERTICAL_PADDING: f32 = 8.0;
// The panel itself is bordered, and border widths are part of its border-box height.
pub const VARIABLES_PANEL_BORDER_WIDTH: f32 = 1.0;
pub const VARIABLES_PANEL_HEADER_HEIGHT: f32 = 32.0;
pub const VARIABLES_COLUMN_HEADER_HEIGHT: f32 = 26.0;
pub const VARIABLES_ROW_HEIGHT: f32 = 26.0;
pub const VARIABLES_ROW_GAP: f32 = 4.0;
pub const VARIABLES_VISIBLE_ROWS: usize = 3;
pub const VARIABLES_NAME_COLUMN_WIDTH: f32 = 140.0;
pub const VARIABLES_TYPE_COLUMN_WIDTH: f32 = 140.0;

/// Debug selectors for the first visible variable rows, so tests can assert their bounds.
pub const VARIABLES_ROW_DEBUG_SELECTORS: [&str; 3] = [
    "notebook-variable-row-0",
    "notebook-variable-row-1",
    "notebook-variable-row-2",
];

pub fn variable_row_debug_selector(index: usize) -> &'static str {
    VARIABLES_ROW_DEBUG_SELECTORS
        .get(index)
        .copied()
        .unwrap_or("notebook-variable-row")
}

pub fn variables_right_min_width() -> f32 {
    VARIABLES_PANEL_HORIZONTAL_PADDING
        + VARIABLES_HEADER_TITLE_WIDTH
        + VARIABLES_HEADER_REFRESH_BUTTON_WIDTH
        + VARIABLES_HEADER_MINIMUM_GAP
}

pub fn clamp_variables_right_width(width: f32, viewport_width: f32) -> f32 {
    let minimum = variables_right_min_width().max(VARIABLES_RIGHT_MIN_WIDTH);
    let maximum = (viewport_width * 0.7).max(minimum);
    width.clamp(minimum, maximum)
}

pub fn variables_bottom_min_height() -> f32 {
    let rows = VARIABLES_VISIBLE_ROWS as f32;
    // The column list is a `v_flex` with the column header plus one row per
    // variable, and a gap between each of those children.
    let column_gaps = rows;
    VARIABLES_PANEL_BORDER_WIDTH
        + VARIABLES_PANEL_VERTICAL_PADDING
        + VARIABLES_PANEL_HEADER_HEIGHT
        + VARIABLES_ROW_GAP
        + VARIABLES_COLUMN_HEADER_HEIGHT
        + rows * VARIABLES_ROW_HEIGHT
        + column_gaps * VARIABLES_ROW_GAP
}

pub fn clamp_variables_bottom_height(height: f32, viewport_height: f32) -> f32 {
    let minimum = variables_bottom_min_height();
    let maximum = (viewport_height * 0.7).max(minimum);
    height.clamp(minimum, maximum)
}

actions!(
    notebook_variables,
    [
        /// Refreshes the notebook variables list.
        RefreshVariables,
        /// Shows or hides the notebook variables panel.
        ToggleVariablesPanel,
        /// Moves the notebook variables panel between the right side and the bottom.
        ToggleVariablesPanelPosition,
    ]
);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum VariablesPanelPosition {
    #[default]
    Right,
    Bottom,
}

/// A Python snippet that publishes the user variables as JSON in the hidden
/// `__zed_variables_json__` name. The execute request evaluates that name through
/// `user_expressions`, so the request can stay silent.
pub const VARIABLES_INTROSPECTION_CODE: &str = r#"
import json as __zed_json
import types as __zed_types

def __zed_variable_shape(value):
    try:
        shape = getattr(value, "shape", None)
        if shape is not None:
            return " x ".join(str(part) for part in tuple(shape))
    except Exception:
        pass
    try:
        if isinstance(value, (list, dict, set, tuple)):
            return "len " + str(len(value))
    except Exception:
        pass
    return None

def __zed_variable_row(name, value):
    value_type = type(value)
    module_name = getattr(value_type, "__module__", "")
    type_name = value_type.__name__
    if module_name in ("builtins", "__main__", ""):
        qualified_type = type_name
    else:
        qualified_type = module_name + "." + type_name
    try:
        preview = repr(value)
    except Exception as error:
        preview = "<repr failed: %s>" % (error,)
    preview = " ".join(preview.split())
    if len(preview) > 120:
        preview = preview[:117] + "..."
    expandable = (
        (module_name.startswith("pandas") and type_name in ("DataFrame", "Series"))
        or (module_name.startswith("numpy") and type_name == "ndarray")
        or (isinstance(value, list) and all(isinstance(item, dict) for item in value))
    )
    return {
        "name": name,
        "type": qualified_type,
        "shape": __zed_variable_shape(value),
        "value": preview,
        "expandable": bool(expandable),
    }

def __zed_variables_payload():
    try:
        shell = get_ipython()
        namespace = shell.user_ns if shell is not None else {}
    except Exception:
        namespace = {}
    ignored_names = {"In", "Out", "get_ipython", "exit", "quit", "open", "display"}
    function_types = (
        __zed_types.FunctionType,
        __zed_types.BuiltinFunctionType,
        __zed_types.MethodType,
    )
    rows = []
    for name in sorted(namespace):
        if name.startswith("_") or name in ignored_names:
            continue
        value = namespace[name]
        if isinstance(value, __zed_types.ModuleType) or isinstance(value, function_types):
            continue
        try:
            rows.append(__zed_variable_row(name, value))
        except Exception as error:
            rows.append({
                "name": name,
                "type": type(value).__name__,
                "shape": None,
                "value": "<unavailable: %s>" % (error,),
                "expandable": False,
            })
    return rows

class __ZedRawText(str):
    # user_expressions reports repr(), which would wrap the JSON in quotes and escape it.
    def __repr__(self):
        return str(self)

globals()["__zed_variables_json__"] = __ZedRawText(__zed_json.dumps(
    {"variables": __zed_variables_payload()},
    default=str,
))
"#;

pub const VARIABLES_INTROSPECTION_EXPRESSION: &str = "__zed_variables_json__";
pub const DATA_PAGE_EXPRESSION: &str = "__zed_data_page_json__";

const DATA_PAGE_CODE_TEMPLATE: &str = r#"
import json as __zed_json
import math as __zed_math

def __zed_safe_value(value):
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if __zed_math.isnan(value) or __zed_math.isinf(value):
            return str(value)
        return value
    return str(value)

def __zed_build_data_page():
    try:
        shell = get_ipython()
        namespace = shell.user_ns if shell is not None else {}
    except Exception:
        namespace = {}
    name = __VARIABLE_LITERAL__
    if name not in namespace:
        return {"error": "Variable '" + name + "' is no longer defined."}
    value = namespace[name]
    sort_column = __SORT_LITERAL__
    ascending = __ASCENDING_LITERAL__
    offset = __OFFSET_LITERAL__
    limit = __LIMIT_LITERAL__
    type_name = type(value).__name__
    module_name = getattr(type(value), "__module__", "")
    try:
        if module_name.startswith("pandas") and type_name in ("DataFrame", "Series"):
            if type_name == "DataFrame":
                columns = [str(column) for column in value.columns]
                ordered = value
                if sort_column is not None:
                    actual = None
                    for column in value.columns:
                        if str(column) == sort_column:
                            actual = column
                            break
                    if actual is None:
                        return {"error": "Column '" + sort_column + "' was not found."}
                    ordered = value.sort_values(by=actual, ascending=ascending, kind="mergesort")
                total = int(len(ordered))
                page = ordered.iloc[offset:offset + limit]
                rows = [
                    [__zed_safe_value(item) for item in record]
                    for record in page.itertuples(index=False, name=None)
                ]
                return {"columns": columns, "rows": rows, "offset": offset, "total_rows": total}
            columns = ["index", "value"]
            ordered = value
            if sort_column == "value":
                ordered = value.sort_values(ascending=ascending, kind="mergesort")
            elif sort_column == "index":
                ordered = value.sort_index(ascending=ascending)
            elif sort_column is not None:
                return {"error": "Series can only be sorted by 'index' or 'value'."}
            total = int(len(ordered))
            page = ordered.iloc[offset:offset + limit]
            rows = [
                [__zed_safe_value(index), __zed_safe_value(item)]
                for index, item in page.items()
            ]
            return {"columns": columns, "rows": rows, "offset": offset, "total_rows": total}
        if module_name.startswith("numpy") and type_name == "ndarray":
            if value.ndim == 1:
                columns = ["index", "value"]
                total = int(value.shape[0])
                indices = list(range(total))
                if sort_column == "index":
                    indices = sorted(indices, reverse=not ascending)
                elif sort_column == "value":
                    indices = sorted(indices, key=lambda index: value[index], reverse=not ascending)
                elif sort_column is not None:
                    return {"error": "One-dimensional arrays can only be sorted by 'index' or 'value'."}
                rows = [
                    [__zed_safe_value(index), __zed_safe_value(value[index])]
                    for index in indices[offset:offset + limit]
                ]
                return {"columns": columns, "rows": rows, "offset": offset, "total_rows": total}
            if value.ndim == 2:
                columns = [str(index) for index in range(value.shape[1])]
                total = int(value.shape[0])
                indices = list(range(total))
                if sort_column is not None:
                    if sort_column not in columns:
                        return {"error": "Column '" + sort_column + "' was not found."}
                    position = int(sort_column)
                    indices = sorted(
                        indices,
                        key=lambda index: value[index, position],
                        reverse=not ascending,
                    )
                rows = [
                    [__zed_safe_value(item) for item in value[index]]
                    for index in indices[offset:offset + limit]
                ]
                return {"columns": columns, "rows": rows, "offset": offset, "total_rows": total}
            return {"error": "Only one- and two-dimensional arrays can be shown in the Data Viewer."}
        if isinstance(value, list) and all(isinstance(item, dict) for item in value):
            columns = []
            for item in value:
                for key in item:
                    key_name = str(key)
                    if key_name not in columns:
                        columns.append(key_name)
            total = len(value)
            ordered = value
            if sort_column is not None:
                if sort_column not in columns:
                    return {"error": "Column '" + sort_column + "' was not found."}

                def __zed_sort_key(item):
                    for key in item:
                        if str(key) == sort_column:
                            return str(item.get(key))
                    return ""

                ordered = sorted(value, key=__zed_sort_key, reverse=not ascending)
            rows = []
            for item in ordered[offset:offset + limit]:
                row = []
                for column in columns:
                    actual = None
                    for key in item:
                        if str(key) == column:
                            actual = key
                            break
                    row.append(__zed_safe_value(item.get(actual)))
                rows.append(row)
            return {"columns": columns, "rows": rows, "offset": offset, "total_rows": total}
        return {
            "error": (
                "The Data Viewer does not support values of type "
                + (module_name + "." if module_name else "")
                + type_name
                + ". Supported values are pandas DataFrames/Series, numpy arrays and lists of dicts."
            )
        }
    except ImportError as error:
        return {"error": "pandas or numpy is not installed in this kernel: " + str(error)}
    except Exception as error:
        return {"error": type(error).__name__ + ": " + str(error)}

class __ZedRawText(str):
    # user_expressions reports repr(), which would wrap the JSON in quotes and escape it.
    def __repr__(self):
        return str(self)

globals()["__zed_data_page_json__"] = __ZedRawText(__zed_json.dumps(
    __zed_build_data_page(),
    default=str,
))
"#;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VariableInfo {
    pub name: String,
    #[serde(rename = "type")]
    pub type_name: String,
    #[serde(default)]
    pub shape: Option<String>,
    #[serde(rename = "value", default)]
    pub preview: String,
    #[serde(default)]
    pub expandable: bool,
}

#[derive(Deserialize)]
struct VariablesPayload {
    variables: Vec<VariableInfo>,
}

pub fn parse_variables_json(json: &str) -> Result<Vec<VariableInfo>> {
    let value: Value =
        serde_json::from_str(json).context("the kernel returned invalid variables JSON")?;
    match value {
        Value::Array(_) => serde_json::from_value(value).context("invalid variables array"),
        Value::Object(_) => serde_json::from_value::<VariablesPayload>(value)
            .map(|payload| payload.variables)
            .context("invalid variables payload"),
        _ => Err(anyhow!(
            "the variables payload must be an object or an array"
        )),
    }
}

pub fn is_python_language(language: &str) -> bool {
    language.to_ascii_lowercase().contains("python")
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DataPageRequest {
    pub variable: String,
    pub offset: usize,
    pub limit: usize,
    pub sort_column: Option<String>,
    pub ascending: bool,
}

impl DataPageRequest {
    pub fn first(variable: impl Into<String>) -> Self {
        Self {
            variable: variable.into(),
            offset: 0,
            limit: VARIABLES_PAGE_SIZE,
            sort_column: None,
            ascending: true,
        }
    }
}

pub fn data_page_code(request: &DataPageRequest) -> Result<String> {
    let variable = serde_json::to_string(&request.variable)
        .context("could not encode the variable name for the kernel")?;
    let sort_column = match &request.sort_column {
        Some(column) => serde_json::to_string(column)
            .context("could not encode the sort column for the kernel")?,
        None => "None".to_string(),
    };
    let limit = request.limit.clamp(1, 1000);
    Ok(DATA_PAGE_CODE_TEMPLATE
        .replace("__VARIABLE_LITERAL__", &variable)
        .replace("__SORT_LITERAL__", &sort_column)
        .replace(
            "__ASCENDING_LITERAL__",
            if request.ascending { "True" } else { "False" },
        )
        .replace("__OFFSET_LITERAL__", &request.offset.to_string())
        .replace("__LIMIT_LITERAL__", &limit.to_string()))
}

#[derive(Clone, Debug, PartialEq)]
pub struct DataPage {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
    pub offset: usize,
    pub total_rows: usize,
}

#[derive(Deserialize)]
struct RawDataPage {
    #[serde(default)]
    columns: Vec<String>,
    #[serde(default)]
    rows: Vec<Vec<Value>>,
    #[serde(default)]
    offset: usize,
    #[serde(default)]
    total_rows: usize,
    #[serde(default)]
    error: Option<String>,
}

pub fn parse_data_page_json(json: &str) -> Result<DataPage> {
    let mut raw: RawDataPage =
        serde_json::from_str(json).context("the kernel returned invalid data page JSON")?;
    if let Some(error) = raw.error.take() {
        return Err(anyhow!(error));
    }
    if !raw.columns.is_empty() {
        for row in &mut raw.rows {
            row.resize(raw.columns.len(), Value::Null);
            row.truncate(raw.columns.len());
        }
    }
    Ok(DataPage {
        columns: raw.columns,
        rows: raw.rows,
        offset: raw.offset,
        total_rows: raw.total_rows,
    })
}

impl DataPage {
    pub fn has_previous(&self) -> bool {
        self.offset > 0
    }

    pub fn has_next(&self) -> bool {
        self.offset.saturating_add(self.rows.len()) < self.total_rows
    }

    pub fn previous_offset(&self) -> usize {
        self.offset.saturating_sub(VARIABLES_PAGE_SIZE)
    }

    pub fn next_offset(&self) -> usize {
        self.offset.saturating_add(VARIABLES_PAGE_SIZE)
    }

    pub fn visible_range(&self) -> (usize, usize) {
        if self.rows.is_empty() {
            (0, 0)
        } else {
            (self.offset + 1, self.offset + self.rows.len())
        }
    }

    pub fn range_label(&self) -> String {
        let (start, end) = self.visible_range();
        if self.total_rows == 0 {
            "0 rows".to_string()
        } else if start == 0 {
            format!("0 of {} rows", self.total_rows)
        } else {
            format!("{start}-{end} of {} rows", self.total_rows)
        }
    }
}

pub fn format_variable_type(variable: &VariableInfo) -> String {
    match &variable.shape {
        Some(shape) if !shape.is_empty() => format!("{} ({})", variable.type_name, shape),
        _ => variable.type_name.clone(),
    }
}

pub fn truncate_preview(preview: &str, max_chars: usize) -> String {
    if preview.chars().count() <= max_chars {
        return preview.to_string();
    }
    let mut truncated = preview
        .chars()
        .take(max_chars.saturating_sub(3))
        .collect::<String>();
    truncated.push_str("...");
    truncated
}

#[derive(Clone, Debug)]
struct SortState {
    column: String,
    ascending: bool,
}

#[derive(Clone, Debug)]
struct DataViewerState {
    variable: String,
    page: Option<DataPage>,
    error: Option<String>,
    loading: bool,
    sort: Option<SortState>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VariablesPanelEvent {
    RefreshRequested,
    OpenVariable {
        name: String,
    },
    LoadPage {
        variable: String,
        offset: usize,
        sort_column: Option<String>,
        ascending: bool,
    },
}

pub struct VariablesPanel {
    variables: Vec<VariableInfo>,
    message: Option<String>,
    error: Option<String>,
    loading: bool,
    viewer: Option<DataViewerState>,
    position: VariablesPanelPosition,
    visible: bool,
}

impl VariablesPanel {
    pub fn new() -> Self {
        Self {
            variables: Vec::new(),
            message: None,
            error: None,
            loading: false,
            viewer: None,
            position: VariablesPanelPosition::Right,
            visible: true,
        }
    }

    pub fn position(&self) -> VariablesPanelPosition {
        self.position
    }

    pub fn is_visible(&self) -> bool {
        self.visible
    }

    pub fn set_position(&mut self, position: VariablesPanelPosition, cx: &mut Context<Self>) {
        if self.position != position {
            self.position = position;
            cx.notify();
        }
    }

    pub fn set_visible(&mut self, visible: bool, cx: &mut Context<Self>) {
        if self.visible != visible {
            self.visible = visible;
            cx.notify();
        }
    }

    pub fn toggle_position(&mut self, cx: &mut Context<Self>) -> VariablesPanelPosition {
        self.position = match self.position {
            VariablesPanelPosition::Right => VariablesPanelPosition::Bottom,
            VariablesPanelPosition::Bottom => VariablesPanelPosition::Right,
        };
        cx.notify();
        self.position
    }

    pub fn toggle_visibility(&mut self, cx: &mut Context<Self>) -> bool {
        self.visible = !self.visible;
        cx.notify();
        self.visible
    }

    pub fn variables(&self) -> &[VariableInfo] {
        &self.variables
    }

    pub fn message(&self) -> Option<&str> {
        self.message.as_deref()
    }

    pub fn error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn viewer_page(&self) -> Option<&DataPage> {
        self.viewer.as_ref().and_then(|viewer| viewer.page.as_ref())
    }

    pub fn begin_loading(&mut self, cx: &mut Context<Self>) {
        self.loading = true;
        self.error = None;
        cx.notify();
    }

    pub fn set_variables(&mut self, variables: Vec<VariableInfo>, cx: &mut Context<Self>) {
        self.variables = variables;
        self.message = None;
        self.error = None;
        self.loading = false;
        if let Some(viewer) = &self.viewer {
            if !self
                .variables
                .iter()
                .any(|variable| variable.name == viewer.variable)
            {
                self.viewer = None;
            }
        }
        cx.notify();
    }

    pub fn set_message(&mut self, message: impl Into<String>, cx: &mut Context<Self>) {
        self.variables.clear();
        self.viewer = None;
        self.message = Some(message.into());
        self.error = None;
        self.loading = false;
        cx.notify();
    }

    pub fn set_error(&mut self, error: impl Into<String>, cx: &mut Context<Self>) {
        self.error = Some(error.into());
        self.message = None;
        self.loading = false;
        cx.notify();
    }

    pub fn set_data_page(&mut self, variable: &str, page: DataPage, cx: &mut Context<Self>) {
        if let Some(viewer) = &mut self.viewer {
            if viewer.variable == variable {
                viewer.page = Some(page);
                viewer.error = None;
                viewer.loading = false;
            }
        }
        cx.notify();
    }

    pub fn set_data_error(
        &mut self,
        variable: &str,
        error: impl Into<String>,
        cx: &mut Context<Self>,
    ) {
        if let Some(viewer) = &mut self.viewer {
            if viewer.variable == variable {
                viewer.error = Some(error.into());
                viewer.page = None;
                viewer.loading = false;
            }
        }
        cx.notify();
    }

    fn open_variable(&mut self, name: String, cx: &mut Context<Self>) {
        self.viewer = Some(DataViewerState {
            variable: name.clone(),
            page: None,
            error: None,
            loading: true,
            sort: None,
        });
        cx.emit(VariablesPanelEvent::OpenVariable { name });
        cx.notify();
    }

    fn close_viewer(&mut self, cx: &mut Context<Self>) {
        self.viewer = None;
        cx.notify();
    }

    fn request_page(&mut self, offset: usize, cx: &mut Context<Self>) {
        let Some(viewer) = &mut self.viewer else {
            return;
        };
        let variable = viewer.variable.clone();
        let sort_column = viewer.sort.as_ref().map(|sort| sort.column.clone());
        let ascending = viewer
            .sort
            .as_ref()
            .map(|sort| sort.ascending)
            .unwrap_or(true);
        viewer.loading = true;
        viewer.error = None;
        cx.emit(VariablesPanelEvent::LoadPage {
            variable,
            offset,
            sort_column,
            ascending,
        });
        cx.notify();
    }

    fn toggle_sort(&mut self, column: String, cx: &mut Context<Self>) {
        let Some(viewer) = &mut self.viewer else {
            return;
        };
        let ascending = match &viewer.sort {
            Some(sort) if sort.column == column => !sort.ascending,
            _ => true,
        };
        viewer.sort = Some(SortState { column, ascending });
        self.request_page(0, cx);
    }

    fn render_list(&self, cx: &mut Context<Self>) -> AnyElement {
        if let Some(message) = &self.message {
            return Label::new(message.clone())
                .size(LabelSize::Small)
                .color(Color::Muted)
                .into_any_element();
        }
        if let Some(error) = &self.error {
            return Label::new(error.clone())
                .size(LabelSize::Small)
                .color(Color::Error)
                .into_any_element();
        }
        if self.loading && self.variables.is_empty() {
            return Label::new("Loading variables...")
                .size(LabelSize::Small)
                .color(Color::Muted)
                .into_any_element();
        }
        if self.variables.is_empty() {
            return Label::new("No user variables yet.")
                .size(LabelSize::Small)
                .color(Color::Muted)
                .into_any_element();
        }

        match self.position {
            VariablesPanelPosition::Right => self.render_stacked_list(cx),
            VariablesPanelPosition::Bottom => self.render_column_list(cx),
        }
    }

    fn render_stacked_list(&self, cx: &mut Context<Self>) -> AnyElement {
        v_flex()
            .gap_1()
            .children(self.variables.iter().enumerate().map(|(index, variable)| {
                let name = variable.name.clone();
                v_flex()
                    .id(("notebook-variable", index))
                    .w_full()
                    .px_1()
                    .py_0p5()
                    .rounded_sm()
                    .cursor_pointer()
                    .hover(|style| style.bg(cx.theme().colors().element_hover))
                    .child(
                        Label::new(variable.name.clone())
                            .size(LabelSize::Small)
                            .weight(gpui::FontWeight::SEMIBOLD)
                            .truncate(),
                    )
                    .child(
                        Label::new(format_variable_type(variable))
                            .size(LabelSize::Small)
                            .color(Color::Muted)
                            .truncate(),
                    )
                    .child(
                        Label::new(truncate_preview(&variable.preview, 120))
                            .size(LabelSize::Small)
                            .color(Color::Muted)
                            .truncate(),
                    )
                    .on_click(cx.listener(move |this, _, _, cx| {
                        this.open_variable(name.clone(), cx);
                    }))
            }))
            .into_any_element()
    }

    fn render_column_list(&self, cx: &mut Context<Self>) -> AnyElement {
        let header = h_flex()
            .w_full()
            .h(px(VARIABLES_COLUMN_HEADER_HEIGHT))
            .px_1()
            .py_0p5()
            .border_b_1()
            .border_color(cx.theme().colors().border)
            .child(
                div().w(px(VARIABLES_NAME_COLUMN_WIDTH)).flex_none().child(
                    Label::new("Name")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                ),
            )
            .child(
                div().w(px(VARIABLES_TYPE_COLUMN_WIDTH)).flex_none().child(
                    Label::new("Type")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                ),
            )
            .child(
                div().flex_1().min_w_0().child(
                    Label::new("Value")
                        .size(LabelSize::Small)
                        .color(Color::Muted),
                ),
            );

        let rows = self.variables.iter().enumerate().map(|(index, variable)| {
            let name = variable.name.clone();
            h_flex()
                .id(("notebook-variable-row", index))
                .debug_selector(move || variable_row_debug_selector(index).to_owned())
                .w_full()
                .h(px(VARIABLES_ROW_HEIGHT))
                .px_1()
                .py_0p5()
                .rounded_sm()
                .cursor_pointer()
                .hover(|style| style.bg(cx.theme().colors().element_hover))
                .child(
                    div()
                        .w(px(VARIABLES_NAME_COLUMN_WIDTH))
                        .flex_none()
                        .min_w_0()
                        .child(
                            Label::new(variable.name.clone())
                                .size(LabelSize::Small)
                                .weight(gpui::FontWeight::SEMIBOLD)
                                .truncate(),
                        ),
                )
                .child(
                    div()
                        .w(px(VARIABLES_TYPE_COLUMN_WIDTH))
                        .flex_none()
                        .min_w_0()
                        .child(
                            Label::new(format_variable_type(variable))
                                .size(LabelSize::Small)
                                .color(Color::Muted)
                                .truncate(),
                        ),
                )
                .child(
                    div().flex_1().min_w_0().child(
                        Label::new(truncate_preview(&variable.preview, 120))
                            .size(LabelSize::Small)
                            .color(Color::Muted)
                            .truncate(),
                    ),
                )
                .on_click(cx.listener(move |this, _, _, cx| {
                    this.open_variable(name.clone(), cx);
                }))
        });

        v_flex()
            .w_full()
            .gap_1()
            .child(header)
            .children(rows)
            .into_any_element()
    }

    fn render_table(&self, viewer: &DataViewerState, cx: &mut Context<Self>) -> AnyElement {
        let Some(page) = &viewer.page else {
            let text = if viewer.loading {
                "Loading data..."
            } else {
                "No data."
            };
            return Label::new(text)
                .size(LabelSize::Small)
                .color(Color::Muted)
                .into_any_element();
        };

        let mut widths = Vec::with_capacity(page.columns.len());
        for (column_index, column) in page.columns.iter().enumerate() {
            let mut width = column.chars().count();
            for row in &page.rows {
                if let Some(value) = row.get(column_index) {
                    width = width.max(display_cell(value).chars().count());
                }
            }
            widths.push((width.clamp(6, 40) as f32) * 7.0);
        }

        let header = h_flex()
            .flex_none()
            .children(
                page.columns
                    .iter()
                    .enumerate()
                    .map(|(column_index, column)| {
                        let column_name = column.clone();
                        let sort_indicator = match &viewer.sort {
                            Some(sort) if sort.column == *column => {
                                if sort.ascending {
                                    " ^"
                                } else {
                                    " v"
                                }
                            }
                            _ => "",
                        };
                        div()
                            .id(("data-header", column_index))
                            .w(px(widths[column_index]))
                            .flex_none()
                            .px_1()
                            .py_0p5()
                            .cursor_pointer()
                            .border_1()
                            .border_color(cx.theme().colors().border)
                            .bg(cx.theme().colors().element_background)
                            .on_click(cx.listener(move |this, _, _, cx| {
                                this.toggle_sort(column_name.clone(), cx);
                            }))
                            .child(
                                Label::new(format!("{column}{sort_indicator}"))
                                    .size(LabelSize::Small),
                            )
                    }),
            );

        let rows = page.rows.iter().enumerate().map(|(row_index, row)| {
            h_flex().id(("data-row", row_index)).flex_none().children(
                page.columns.iter().enumerate().map(|(column_index, _)| {
                    let value = row.get(column_index).map(display_cell).unwrap_or_default();
                    div()
                        .w(px(widths[column_index]))
                        .flex_none()
                        .px_1()
                        .py_0p5()
                        .border_1()
                        .border_color(cx.theme().colors().border)
                        .child(Label::new(value).size(LabelSize::Small))
                }),
            )
        });

        v_flex()
            .w_full()
            .gap_1()
            .child(
                h_flex()
                    .w_full()
                    .justify_between()
                    .child(Label::new(page.range_label()).size(LabelSize::Small))
                    .child(
                        h_flex()
                            .gap_1()
                            .child(
                                Button::new("variables-prev-page", "Previous")
                                    .disabled(!page.has_previous())
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        let offset = this
                                            .viewer
                                            .as_ref()
                                            .and_then(|viewer| viewer.page.as_ref())
                                            .map(DataPage::previous_offset)
                                            .unwrap_or(0);
                                        this.request_page(offset, cx);
                                    })),
                            )
                            .child(
                                Button::new("variables-next-page", "Next")
                                    .disabled(!page.has_next())
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        let offset = this
                                            .viewer
                                            .as_ref()
                                            .and_then(|viewer| viewer.page.as_ref())
                                            .map(DataPage::next_offset)
                                            .unwrap_or(0);
                                        this.request_page(offset, cx);
                                    })),
                            ),
                    ),
            )
            .child(
                div()
                    .id("notebook-variables-data-table")
                    .w_full()
                    .overflow_x_scroll()
                    .child(v_flex().w_full().child(header).children(rows)),
            )
            .into_any_element()
    }

    fn render_viewer(&self, viewer: &DataViewerState, cx: &mut Context<Self>) -> AnyElement {
        v_flex()
            .w_full()
            .h_full()
            .gap_1()
            .child(
                h_flex()
                    .w_full()
                    .gap_1()
                    .child(
                        IconButton::new("variables-back", IconName::ArrowLeft)
                            .tooltip(Tooltip::text("Back to variables"))
                            .on_click(cx.listener(|this, _, _, cx| this.close_viewer(cx))),
                    )
                    .child(
                        Label::new(viewer.variable.clone())
                            .size(LabelSize::Small)
                            .weight(gpui::FontWeight::SEMIBOLD),
                    ),
            )
            .child(if let Some(error) = &viewer.error {
                Label::new(error.clone())
                    .size(LabelSize::Small)
                    .color(Color::Error)
                    .into_any_element()
            } else {
                self.render_table(viewer, cx)
            })
            .into_any_element()
    }
}

impl Default for VariablesPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl EventEmitter<VariablesPanelEvent> for VariablesPanel {}

fn display_cell(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::String(text) => text.clone(),
        Value::Bool(boolean) => boolean.to_string(),
        Value::Number(number) => number.to_string(),
        Value::Array(array) => format!("{array:?}"),
        Value::Object(object) => format!("{object:?}"),
    }
}

impl Render for VariablesPanel {
    fn render(&mut self, _window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        v_flex()
            .id("notebook-variables-panel")
            .debug_selector(|| "notebook-variables-panel".to_owned())
            .size_full()
            .gap_1()
            .px_2()
            .py_1()
            .when(self.position == VariablesPanelPosition::Right, |this| {
                this.border_l_1()
            })
            .when(self.position == VariablesPanelPosition::Bottom, |this| {
                this.border_t_1()
            })
            .border_color(cx.theme().colors().border)
            .child(
                h_flex()
                    .w_full()
                    .h(px(VARIABLES_PANEL_HEADER_HEIGHT))
                    .justify_between()
                    .child(
                        Label::new("Variables")
                            .size(LabelSize::Small)
                            .weight(gpui::FontWeight::SEMIBOLD),
                    )
                    .child(
                        IconButton::new("variables-refresh", IconName::RotateCw)
                            .tooltip(Tooltip::text("Refresh variables"))
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.loading = true;
                                this.error = None;
                                cx.emit(VariablesPanelEvent::RefreshRequested);
                                cx.notify();
                            })),
                    ),
            )
            .child(if let Some(viewer) = &self.viewer {
                self.render_viewer(viewer, cx)
            } else {
                div()
                    .id("notebook-variables-list")
                    .w_full()
                    .flex_1()
                    .overflow_y_scroll()
                    .child(self.render_list(cx))
                    .into_any_element()
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_variables_payload() {
        let variables = parse_variables_json(
            r#"{"variables":[
                {"name":"df","type":"pandas.core.frame.DataFrame","shape":"3 x 2","value":"   a  b","expandable":true},
                {"name":"items","type":"list","shape":"len 2","value":"[1, 2]"}
            ]}"#,
        )
        .expect("payload should parse");

        assert_eq!(variables.len(), 2);
        assert_eq!(variables[0].name, "df");
        assert_eq!(variables[0].type_name, "pandas.core.frame.DataFrame");
        assert_eq!(variables[0].shape.as_deref(), Some("3 x 2"));
        assert_eq!(variables[0].preview, "   a  b");
        assert!(variables[0].expandable);
        assert!(!variables[1].expandable);
    }

    #[test]
    fn parses_bare_variables_array() {
        let variables = parse_variables_json(r#"[{"name":"x","type":"int","value":"1"}]"#)
            .expect("array payload should parse");
        assert_eq!(variables.len(), 1);
        assert_eq!(variables[0].shape, None);
        assert_eq!(variables[0].preview, "1");
    }

    #[test]
    fn rejects_invalid_variables_payload() {
        assert!(parse_variables_json("not json").is_err());
        assert!(parse_variables_json(r#"{"other": []}"#).is_err());
    }

    #[test]
    fn formats_variable_rows() {
        let with_shape = VariableInfo {
            name: "df".to_string(),
            type_name: "DataFrame".to_string(),
            shape: Some("3 x 2".to_string()),
            preview: "a b".to_string(),
            expandable: true,
        };
        assert_eq!(format_variable_type(&with_shape), "DataFrame (3 x 2)");
        assert_eq!(truncate_preview("hello", 10), "hello");
        assert_eq!(truncate_preview("hello world", 8), "hello...");
    }

    #[test]
    fn parses_data_page_and_computes_pagination() {
        let page = parse_data_page_json(
            r#"{"columns":["a","b"],"rows":[[1,"x"],[2]],"offset":200,"total_rows":450}"#,
        )
        .expect("page should parse");

        assert_eq!(page.columns, vec!["a".to_string(), "b".to_string()]);
        assert_eq!(page.rows[1], vec![Value::from(2), Value::Null]);
        assert!(page.has_previous());
        assert!(page.has_next());
        assert_eq!(page.previous_offset(), 0);
        assert_eq!(page.next_offset(), 400);
        assert_eq!(page.range_label(), "201-202 of 450 rows");
    }

    #[test]
    fn parses_data_page_errors() {
        let error = parse_data_page_json(r#"{"error":"pandas is not installed in this kernel"}"#)
            .expect_err("error payload should fail");
        assert!(error.to_string().contains("pandas is not installed"));
    }

    #[test]
    fn builds_data_page_code_without_python_injection() {
        let request = DataPageRequest {
            variable: "df'\nimport os".to_string(),
            offset: 40,
            limit: 200,
            sort_column: Some("na\"me".to_string()),
            ascending: false,
        };
        let code = data_page_code(&request).expect("code should build");

        assert!(code.contains(r#"name = "df'\nimport os""#));
        assert!(code.contains(r#"sort_column = "na\"me""#));
        assert!(code.contains("ascending = False"));
        assert!(code.contains("offset = 40"));
        assert!(code.contains("limit = 200"));
        assert!(!code.contains("__VARIABLE_LITERAL__"));
    }

    #[test]
    fn recognizes_python_kernels() {
        assert!(is_python_language("python"));
        assert!(is_python_language("Python 3"));
        assert!(!is_python_language("julia"));
        assert!(!is_python_language("r"));
    }

    #[test]
    fn right_minimum_width_fits_header_contents() {
        let header_contents = VARIABLES_PANEL_HORIZONTAL_PADDING
            + VARIABLES_HEADER_TITLE_WIDTH
            + VARIABLES_HEADER_REFRESH_BUTTON_WIDTH
            + VARIABLES_HEADER_MINIMUM_GAP;

        assert!(variables_right_min_width() >= header_contents);
        assert!(VARIABLES_RIGHT_MIN_WIDTH >= header_contents);
    }

    #[test]
    fn clamps_right_width_between_minimum_and_seventy_percent() {
        let minimum = variables_right_min_width().max(VARIABLES_RIGHT_MIN_WIDTH);

        assert_eq!(clamp_variables_right_width(10.0, 1000.0), minimum);
        assert_eq!(clamp_variables_right_width(400.0, 1000.0), 400.0);
        assert_eq!(clamp_variables_right_width(5000.0, 1000.0), 700.0);
        assert_eq!(clamp_variables_right_width(10.0, 100.0), minimum);
    }

    #[test]
    fn bottom_minimum_height_fits_header_columns_and_three_rows() {
        let expected = VARIABLES_PANEL_BORDER_WIDTH
            + VARIABLES_PANEL_VERTICAL_PADDING
            + VARIABLES_PANEL_HEADER_HEIGHT
            + VARIABLES_ROW_GAP
            + VARIABLES_COLUMN_HEADER_HEIGHT
            + VARIABLES_VISIBLE_ROWS as f32 * VARIABLES_ROW_HEIGHT
            + VARIABLES_VISIBLE_ROWS as f32 * VARIABLES_ROW_GAP;

        assert_eq!(variables_bottom_min_height(), expected);
        assert!(
            variables_bottom_min_height()
                >= VARIABLES_COLUMN_HEADER_HEIGHT
                    + VARIABLES_VISIBLE_ROWS as f32 * VARIABLES_ROW_HEIGHT
        );
    }

    #[test]
    fn clamps_bottom_height_between_minimum_and_seventy_percent() {
        let minimum = variables_bottom_min_height();

        assert_eq!(clamp_variables_bottom_height(10.0, 1000.0), minimum);
        assert_eq!(clamp_variables_bottom_height(400.0, 1000.0), 400.0);
        assert_eq!(clamp_variables_bottom_height(5000.0, 1000.0), 700.0);
        assert_eq!(clamp_variables_bottom_height(10.0, 100.0), minimum);
    }

    #[gpui::test]
    fn bottom_panel_shows_three_full_rows_at_minimum_height(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });

        let variables = (0..5)
            .map(|index| VariableInfo {
                name: format!("variable_{index}"),
                type_name: "int".to_string(),
                shape: None,
                preview: index.to_string(),
                expandable: false,
            })
            .collect::<Vec<_>>();

        let minimum_height = px(variables_bottom_min_height());
        let (_panel, cx) = cx.add_window_view(|_, cx| {
            let mut panel = VariablesPanel::new();
            panel.set_position(VariablesPanelPosition::Bottom, cx);
            panel.set_variables(variables, cx);
            panel
        });
        cx.simulate_resize(gpui::size(px(600.), minimum_height));
        cx.run_until_parked();

        let panel_bounds = cx
            .debug_bounds("notebook-variables-panel")
            .expect("the panel bounds should be recorded");

        for (index, selector) in VARIABLES_ROW_DEBUG_SELECTORS.iter().enumerate() {
            let bounds = cx
                .debug_bounds(selector)
                .unwrap_or_else(|| panic!("row {index} bounds should be recorded"));
            assert!(
                bounds.top() >= panel_bounds.top() && bounds.bottom() <= panel_bounds.bottom(),
                "row {index} at {bounds:?} should be fully inside {panel_bounds:?}"
            );
        }
    }

    #[gpui::test]
    fn renders_many_variables_in_both_positions(cx: &mut gpui::TestAppContext) {
        cx.update(|cx| {
            let settings_store = settings::SettingsStore::test(cx);
            cx.set_global(settings_store);
            theme_settings::init(theme::LoadThemes::JustBase, cx);
        });

        let variables = (0..250)
            .map(|index| VariableInfo {
                name: format!("variable_{index}"),
                type_name: "pandas.core.frame.DataFrame".to_string(),
                shape: Some("100 x 4".to_string()),
                preview: format!("{{\"column\": {index}}}"),
                expandable: true,
            })
            .collect::<Vec<_>>();

        for position in [
            VariablesPanelPosition::Right,
            VariablesPanelPosition::Bottom,
        ] {
            let variables = variables.clone();
            let (_panel, cx) = cx.add_window_view(|_, cx| {
                let mut panel = VariablesPanel::new();
                panel.set_position(position, cx);
                panel.set_variables(variables, cx);
                panel
            });
            cx.run_until_parked();
        }
    }

    #[gpui::test]
    fn panel_emits_open_and_page_events(cx: &mut gpui::TestAppContext) {
        let panel = cx.update(|cx| cx.new(|_| VariablesPanel::new()));
        let events = std::rc::Rc::new(std::cell::RefCell::new(Vec::new()));
        cx.update(|cx| {
            cx.subscribe(&panel, {
                let events = events.clone();
                move |_, event: &VariablesPanelEvent, _| {
                    events.borrow_mut().push(event.clone());
                }
            })
            .detach();
            panel.update(cx, |panel, cx| {
                panel.set_variables(
                    vec![VariableInfo {
                        name: "df".to_string(),
                        type_name: "DataFrame".to_string(),
                        shape: Some("3 x 2".to_string()),
                        preview: "a b".to_string(),
                        expandable: true,
                    }],
                    cx,
                );
                panel.set_data_page(
                    "df",
                    DataPage {
                        columns: vec!["a".to_string()],
                        rows: vec![vec![Value::from(1)], vec![Value::from(2)]],
                        offset: 0,
                        total_rows: 2,
                    },
                    cx,
                );
            });
        });

        cx.update(|cx| {
            panel.update(cx, |panel, cx| {
                panel.open_variable("df".to_string(), cx);
                panel.toggle_sort("a".to_string(), cx);
            });
        });

        let events = events.borrow();
        assert_eq!(
            events[0],
            VariablesPanelEvent::OpenVariable {
                name: "df".to_string()
            }
        );
        assert_eq!(
            events[1],
            VariablesPanelEvent::LoadPage {
                variable: "df".to_string(),
                offset: 0,
                sort_column: Some("a".to_string()),
                ascending: true,
            }
        );
    }
}
