//! Kernel-backed completions and inspection for notebook code cells.
//!
//! The notebook's language server cannot see values that only exist in the live
//! interpreter (a DataFrame's columns, a dict's keys, the members of a runtime
//! object). This bridges [`notebook_lsp::KernelCompleter`] to the running
//! Jupyter kernel: requests are sent over the kernel's shell channel and the
//! matching replies, which arrive through the ordinary message routing, are
//! handed back through a oneshot channel.

use std::{cell::RefCell, collections::HashMap, rc::Rc, time::Duration};

use anyhow::Result;
use futures::channel::{mpsc, oneshot};
use gpui::{BackgroundExecutor, FutureExt, Task};
use jupyter_protocol::MediaType;
use notebook_lsp::{
    KernelCompletion, KernelCompleter, KernelInspection, KernelInspectionKind,
};
use runtimelib::{
    CompleteReply, CompleteRequest, InspectReply, InspectRequest, JupyterMessage,
    JupyterMessageContent,
};

/// Upper bound on how long a request may stay pending before its entry is
/// forgotten. The consumer applies its own shorter timeout; this only stops
/// abandoned entries from accumulating when a kernel goes silent.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(5);

/// A reply from the kernel, tagged by the request that produced it.
#[derive(Debug)]
enum KernelReply {
    Complete(CompleteReply),
    Inspect(InspectReply),
}

struct State {
    request_tx: Option<mpsc::Sender<JupyterMessage>>,
    idle: bool,
    pending: HashMap<String, oneshot::Sender<KernelReply>>,
}

/// Sends completion and inspection requests to one notebook's running kernel.
///
/// The shared [`State`] is owned by the same [`Rc`] as the notebook editor, so
/// the editor can publish the kernel's shell channel and idle status and can
/// resolve pending requests as replies are routed back.
pub struct NotebookKernelCompleter {
    state: Rc<RefCell<State>>,
    executor: BackgroundExecutor,
}

impl NotebookKernelCompleter {
    pub fn new(executor: BackgroundExecutor) -> Rc<Self> {
        Rc::new(Self {
            state: Rc::new(RefCell::new(State {
                request_tx: None,
                idle: false,
                pending: HashMap::default(),
            })),
            executor,
        })
    }

    /// Records the current kernel: its shell channel (or `None` when no kernel
    /// is running) and whether it is idle. Dropping the channel also clears any
    /// pending requests, since the old process can no longer answer them.
    pub fn set_kernel(&self, request_tx: Option<mpsc::Sender<JupyterMessage>>, idle: bool) {
        let mut state = self.state.borrow_mut();
        if request_tx.is_none() {
            state.pending.clear();
        }
        state.request_tx = request_tx;
        state.idle = idle;
    }

    /// Routes a kernel reply to the request that awaits it. Returns whether the
    /// message answered a pending completion or inspection request.
    pub fn resolve(&self, message: &JupyterMessage) -> bool {
        let Some(parent_header) = &message.parent_header else {
            return false;
        };
        let reply = match &message.content {
            JupyterMessageContent::CompleteReply(reply) => KernelReply::Complete(reply.clone()),
            JupyterMessageContent::InspectReply(reply) => KernelReply::Inspect(reply.clone()),
            _ => return false,
        };

        let sender = self.state.borrow_mut().pending.remove(&parent_header.msg_id);
        let Some(sender) = sender else {
            return false;
        };
        match sender.send(reply) {
            Ok(()) => true,
            Err(_) => {
                log::debug!("kernel completer: reply arrived after the requester gave up");
                false
            }
        }
    }

    fn send_request(&self, message: JupyterMessage) -> Option<oneshot::Receiver<KernelReply>> {
        let mut state = self.state.borrow_mut();
        if !state.idle {
            return None;
        }
        let mut sender = state.request_tx.clone()?;
        let message_id = message.header.msg_id.clone();
        let (reply_tx, reply_rx) = oneshot::channel();
        if sender.try_send(message).is_err() {
            log::debug!("kernel completer: dropped request because the kernel channel is full");
            return None;
        }
        state.pending.insert(message_id, reply_tx);
        Some(reply_rx)
    }
}

impl KernelCompleter for NotebookKernelCompleter {
    fn complete(&self, code: String, cursor_pos: usize) -> Task<Result<Option<KernelCompletion>>> {
        let request = CompleteRequest { code, cursor_pos };
        let Some(receiver) = self.send_request(request.into()) else {
            return Task::ready(Ok(None));
        };

        let executor = self.executor.clone();
        executor.spawn({
            let executor = executor.clone();
            async move {
                match receiver.with_timeout(REQUEST_TIMEOUT, &executor).await {
                    Ok(Ok(KernelReply::Complete(reply))) => Ok(Some(KernelCompletion {
                        matches: reply.matches,
                        cursor_start: reply.cursor_start,
                        cursor_end: reply.cursor_end,
                    })),
                    Ok(Ok(KernelReply::Inspect(_))) => Ok(None),
                    Ok(Err(_)) => {
                        log::debug!("kernel completer: completion request was abandoned");
                        Ok(None)
                    }
                    Err(_) => {
                        log::debug!("kernel completer: completion request timed out");
                        Ok(None)
                    }
                }
            }
        })
    }

    fn inspect(&self, code: String, cursor_pos: usize) -> Task<Result<Option<KernelInspection>>> {
        let request = InspectRequest {
            code,
            cursor_pos,
            detail_level: Some(0),
        };
        let Some(receiver) = self.send_request(request.into()) else {
            return Task::ready(Ok(None));
        };

        let executor = self.executor.clone();
        executor.spawn({
            let executor = executor.clone();
            async move {
                match receiver.with_timeout(REQUEST_TIMEOUT, &executor).await {
                    Ok(Ok(KernelReply::Inspect(reply))) => Ok(inspection_from_reply(&reply)),
                    Ok(Ok(KernelReply::Complete(_))) => Ok(None),
                    Ok(Err(_)) => {
                        log::debug!("kernel completer: inspection request was abandoned");
                        Ok(None)
                    }
                    Err(_) => {
                        log::debug!("kernel completer: inspection request timed out");
                        Ok(None)
                    }
                }
            }
        })
    }
}

/// Extracts the richest textual representation from an inspection reply,
/// preferring markdown over plain text.
fn inspection_from_reply(reply: &InspectReply) -> Option<KernelInspection> {
    if !reply.found {
        return None;
    }

    let markdown = reply.data.content.iter().find_map(|media| match media {
        MediaType::Markdown(text) => Some(text.clone()),
        _ => None,
    });
    if let Some(text) = markdown {
        return Some(KernelInspection {
            kind: KernelInspectionKind::Markdown,
            text,
        });
    }

    let plain = reply.data.content.iter().find_map(|media| match media {
        MediaType::Plain(text) => Some(text.clone()),
        _ => None,
    });
    plain.map(|text| KernelInspection {
        kind: KernelInspectionKind::PlainText,
        text,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt as _;
    use gpui::TestAppContext;
    use jupyter_protocol::media::Media;

    #[gpui::test]
    async fn complete_round_trips_through_the_kernel_channel(cx: &mut TestAppContext) {
        let completer = NotebookKernelCompleter::new(cx.background_executor.clone());
        let (request_tx, mut request_rx) = mpsc::channel(4);
        completer.set_kernel(Some(request_tx), true);

        let task = completer.complete("frame.".to_string(), 6);

        let request = request_rx.next().await.expect("completion request");
        assert!(matches!(
            request.content,
            JupyterMessageContent::CompleteRequest(_)
        ));

        let reply = CompleteReply {
            matches: vec!["columns".to_string()],
            cursor_start: 6,
            cursor_end: 6,
            ..Default::default()
        }
        .as_child_of(&request);
        assert!(completer.resolve(&reply), "the reply should resolve the request");

        let completion = task.await.expect("completion task should succeed");
        assert_eq!(
            completion,
            Some(KernelCompletion {
                matches: vec!["columns".to_string()],
                cursor_start: 6,
                cursor_end: 6,
            })
        );
    }

    #[gpui::test]
    async fn busy_kernel_does_not_send_a_request(cx: &mut TestAppContext) {
        let completer = NotebookKernelCompleter::new(cx.background_executor.clone());
        let (request_tx, mut request_rx) = mpsc::channel(4);
        completer.set_kernel(Some(request_tx), false);

        let completion = completer
            .complete("frame.".to_string(), 6)
            .await
            .expect("completion task should succeed");
        assert_eq!(completion, None);
        assert!(
            request_rx.try_recv().is_err(),
            "nothing should be sent while the kernel is busy"
        );
    }

    #[gpui::test]
    async fn stopping_the_kernel_clears_pending_requests(cx: &mut TestAppContext) {
        let completer = NotebookKernelCompleter::new(cx.background_executor.clone());
        let (request_tx, mut request_rx) = mpsc::channel(4);
        completer.set_kernel(Some(request_tx), true);
        let task = completer.complete("frame.".to_string(), 6);
        request_rx.next().await.expect("completion request");

        completer.set_kernel(None, false);
        assert_eq!(
            task.await.expect("completion task should succeed"),
            None
        );
    }

    #[test]
    fn inspection_prefers_markdown_over_plain() {
        let reply = InspectReply {
            found: true,
            data: Media {
                content: vec![
                    MediaType::Plain("plain".to_string()),
                    MediaType::Markdown("**markdown**".to_string()),
                ],
            },
            ..Default::default()
        };
        assert_eq!(
            inspection_from_reply(&reply),
            Some(KernelInspection {
                kind: KernelInspectionKind::Markdown,
                text: "**markdown**".to_string(),
            })
        );

        let reply = InspectReply {
            found: true,
            data: Media {
                content: vec![MediaType::Plain("int".to_string())],
            },
            ..Default::default()
        };
        assert_eq!(
            inspection_from_reply(&reply),
            Some(KernelInspection {
                kind: KernelInspectionKind::PlainText,
                text: "int".to_string(),
            })
        );

        let reply = InspectReply {
            found: false,
            ..Default::default()
        };
        assert_eq!(inspection_from_reply(&reply), None);
    }
}

