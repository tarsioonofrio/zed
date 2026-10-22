use editor::{Editor, EditorMode, HighlightKey, MultiBuffer, MultiBufferOffset, SizingBehavior};
use gpui::{AnyElement, App, Context, Entity, FontWeight, HighlightStyle, Window};
use language::{Buffer, language_settings::SoftWrap};
use ui::{IconButton, IconButtonShape, IconName, IconSize, Label, h_flex, prelude::*, v_flex};

const MAX_VISIBLE_TRACEBACK_LINES: usize = 20;

/// Userspace error from the kernel
#[derive(Clone)]
pub struct ErrorView {
    pub ename: String,
    pub evalue: String,
    /// The traceback with its ANSI escape sequences removed.
    pub traceback_text: String,
    body: Entity<ErrorBody>,
}

impl ErrorView {
    pub fn new(
        ename: String,
        evalue: String,
        traceback: &str,
        window: &mut Window,
        cx: &mut App,
    ) -> Self {
        let parsed = terminal::parse_ansi_text(traceback.as_bytes());
        let traceback_text = parsed.text.trim_end().to_string();
        let body = cx.new(|cx| {
            ErrorBody::new(
                ename.clone(),
                evalue.clone(),
                traceback_text.clone(),
                &parsed,
                window,
                cx,
            )
        });

        Self {
            ename,
            evalue,
            traceback_text,
            body,
        }
    }

    pub fn render(&self, _window: &mut Window, _cx: &mut App) -> Option<AnyElement> {
        Some(self.body.clone().into_any_element())
    }
}

struct ErrorBody {
    ename: String,
    evalue: String,
    editor: Entity<Editor>,
    collapsed: bool,
}

impl ErrorBody {
    fn new(
        ename: String,
        evalue: String,
        text: String,
        parsed: &terminal::ParsedAnsiText,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let text_length = text.len();
        let buffer = cx.new(|cx| Buffer::local(text, cx));
        let multi_buffer = cx.new(|cx| MultiBuffer::singleton(buffer, cx));
        let editor = cx.new(|cx| {
            let mut editor = Editor::new(
                EditorMode::Full {
                    scale_ui_elements_with_buffer_font_size: false,
                    show_active_line_background: false,
                    sizing_behavior: SizingBehavior::SizeByContent,
                },
                multi_buffer,
                None,
                window,
                cx,
            );
            editor.set_read_only(true);
            editor.set_show_gutter(false, cx);
            editor.set_soft_wrap_mode(SoftWrap::None, cx);
            editor.disable_mouse_wheel_zoom();
            editor.disable_scrollbars_and_minimap(window, cx);

            let snapshot = editor.buffer().read(cx).snapshot(cx);
            let range_of = |range: &std::ops::Range<usize>| {
                let start = range.start.min(text_length);
                let end = range.end.min(text_length);
                (start < end).then(|| {
                    (
                        start,
                        snapshot.anchor_after(MultiBufferOffset(start))
                            ..snapshot.anchor_before(MultiBufferOffset(end)),
                    )
                })
            };

            for (range, color) in &parsed.foreground_spans {
                let (Some(color), Some((start, range))) = (color, range_of(range)) else {
                    continue;
                };
                let style = HighlightStyle {
                    color: Some(terminal_view::terminal_element::convert_color(
                        color,
                        cx.theme(),
                    )),
                    ..Default::default()
                };
                editor.highlight_text_key(
                    HighlightKey::ConsoleAnsiHighlight(start),
                    vec![range],
                    style,
                    false,
                    cx,
                );
            }

            for (range, color) in &parsed.background_spans {
                let (Some(color), Some((start, range))) = (*color, range_of(range)) else {
                    continue;
                };
                editor.highlight_background(
                    HighlightKey::ConsoleAnsiHighlight(start),
                    &[range],
                    move |_, theme| {
                        if terminal::is_default_background_color(color) {
                            theme.colors().terminal_background
                        } else {
                            terminal_view::terminal_element::convert_color(&color, theme)
                        }
                    },
                    cx,
                );
            }

            editor
        });

        Self {
            ename,
            evalue,
            editor,
            collapsed: false,
        }
    }
}

impl Render for ErrorBody {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let padding = window.line_height() / 2.;
        let border_color = cx.theme().status().error_border;
        let collapsed = self.collapsed;

        v_flex()
            .gap_1()
            .child(
                h_flex()
                    .gap_1()
                    .font_buffer(cx)
                    .child(
                        IconButton::new(
                            "toggle-error-traceback",
                            if collapsed {
                                IconName::ChevronRight
                            } else {
                                IconName::ChevronDown
                            },
                        )
                        .shape(IconButtonShape::Square)
                        .icon_size(IconSize::Small)
                        .tooltip(ui::Tooltip::text(if collapsed {
                            "Expand Traceback"
                        } else {
                            "Collapse Traceback"
                        }))
                        .on_click(cx.listener(|this, _, _, cx| {
                            this.collapsed = !this.collapsed;
                            cx.notify();
                        })),
                    )
                    .child(
                        Label::new(format!("{}: ", self.ename))
                            .color(Color::Error)
                            .weight(FontWeight::BOLD),
                    )
                    .child(
                        div().min_w_0().overflow_hidden().child(
                            Label::new(self.evalue.lines().next().unwrap_or_default().to_string())
                                .weight(FontWeight::BOLD)
                                .single_line(),
                        ),
                    ),
            )
            .when(!collapsed, |this| {
                this.child(
                    div()
                        .id("error-traceback")
                        .w_full()
                        .max_h(window.line_height() * MAX_VISIBLE_TRACEBACK_LINES as f32)
                        .overflow_y_scroll()
                        .pl(padding)
                        .border_l_1()
                        .border_color(border_color)
                        .child(self.editor.clone()),
                )
            })
    }
}
