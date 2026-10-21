use gpui::{App, AppContext, ClipboardItem, Context, Entity, Window, div, prelude::*};
use language::Buffer;
use markdown::{Markdown, MarkdownElement, MarkdownFont, MarkdownOptions, MarkdownStyle};

use crate::outputs::OutputContent;

pub struct MarkdownView {
    markdown: Entity<Markdown>,
}

impl MarkdownView {
    pub fn from(text: String, cx: &mut Context<Self>) -> Self {
        Self::from_source(text, cx)
    }

    /// Renders a `text/latex` output by wrapping it in math delimiters so the
    /// markdown math renderer can display it.
    pub fn from_latex(latex: String, cx: &mut Context<Self>) -> Self {
        let source = if latex.trim_start().starts_with('$') {
            latex
        } else {
            format!("$$\n{latex}\n$$")
        };
        Self::from_source(source, cx)
    }

    fn from_source(text: String, cx: &mut Context<Self>) -> Self {
        let markdown = cx.new(|cx| {
            Markdown::new_with_options(
                text.into(),
                None,
                None,
                MarkdownOptions {
                    render_math: true,
                    ..Default::default()
                },
                cx,
            )
        });

        Self { markdown }
    }
}

impl OutputContent for MarkdownView {
    fn clipboard_content(&self, _window: &Window, cx: &App) -> Option<ClipboardItem> {
        let source = self.markdown.read(cx).source().to_string();
        Some(ClipboardItem::new_string(source))
    }

    fn has_clipboard_content(&self, _window: &Window, _cx: &App) -> bool {
        true
    }

    fn has_buffer_content(&self, _window: &Window, _cx: &App) -> bool {
        true
    }

    fn buffer_content(&mut self, _: &mut Window, cx: &mut App) -> Option<Entity<Buffer>> {
        let source = self.markdown.read(cx).source().to_string();
        let buffer = cx.new(|cx| {
            let mut buffer =
                Buffer::local(source.clone(), cx).with_language(language::PLAIN_TEXT.clone(), cx);
            buffer.set_capability(language::Capability::ReadOnly, cx);
            buffer
        });
        Some(buffer)
    }
}

impl Render for MarkdownView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let style = markdown_style(window, cx);
        div()
            .w_full()
            .child(MarkdownElement::new(self.markdown.clone(), style))
    }
}

fn markdown_style(window: &Window, cx: &App) -> MarkdownStyle {
    MarkdownStyle::themed(MarkdownFont::Editor, window, cx)
}
