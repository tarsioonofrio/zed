use collections::HashMap;

use editor::EditorSettings;
use gpui::App;
use settings::{
    NotebookCellNumbersContent, NotebookVariablesPanelContent, RegisterSetting, Settings,
};

#[derive(Debug, Default, RegisterSetting)]
pub struct JupyterSettings {
    pub kernel_selections: HashMap<String, String>,
    pub notebook_cell_numbers: NotebookCellNumbersContent,
    pub notebook_variables_panel: NotebookVariablesPanelContent,
}

impl JupyterSettings {
    pub fn enabled(cx: &App) -> bool {
        // In order to avoid a circular dependency between `editor` and `repl` crates,
        // we put the `enable` flag on its settings.
        // This allows the editor to set up context for key bindings/actions.
        EditorSettings::jupyter_enabled(cx)
    }
}

impl Settings for JupyterSettings {
    fn from_settings(content: &settings::SettingsContent) -> Self {
        let jupyter = content.editor.jupyter.clone().unwrap();
        Self {
            kernel_selections: jupyter.kernel_selections.unwrap_or_default(),
            notebook_cell_numbers: jupyter.notebook_cell_numbers.unwrap_or_default(),
            notebook_variables_panel: jupyter.notebook_variables_panel.unwrap_or_default(),
        }
    }
}
