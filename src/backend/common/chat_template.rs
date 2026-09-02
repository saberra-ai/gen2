// The Jinja chat-template engine is constructed by backend loaders compiled out
// of the default feature surface the pre-commit clippy runs, so its items read
// as dead there while being live (22+ refs) under those backends.
#![allow(dead_code)]

use crate::types::message::{
    ChatTemplateInputs, Message, MessageBody, MessageChunk, MessageContent, TextMessage,
    TokenizerConfigToken, ToolSpec,
};
use anyhow::{Error, Result};
use chrono::Local;
use minijinja::{Environment, ErrorKind, Template};
use minijinja_contrib::pycompat;

#[derive(Clone)]
pub(crate) struct ChatTemplate {
    template: Template<'static, 'static>,
    bos_token: Option<String>,
    eos_token: Option<String>,
    use_default_tool_template: bool,
}

/// Raise a exception (custom function) used in the chat templates
pub(crate) fn raise_exception(err_text: String) -> Result<String, minijinja::Error> {
    Err(minijinja::Error::new(ErrorKind::SyntaxError, err_text))
}

/// Get the current date in a specific format (custom function), similar to `datetime.now().strftime()` in Python
pub(crate) fn strftime_now(format_str: String) -> Result<String, minijinja::Error> {
    Ok(Local::now().format(&format_str).to_string())
}
impl ChatTemplate {
    pub(crate) fn new(
        template: String,
        bos_token: Option<TokenizerConfigToken>,
        eos_token: Option<TokenizerConfigToken>,
    ) -> Self {
        let mut env = Box::new(Environment::new());
        // enable things like .strip() or .capitalize()
        env.set_unknown_method_callback(pycompat::unknown_method_callback);

        // Workaround: gemma3 templates index into content as a list of dicts,
        // but we pass content as a flat string. Rewrite the accessor.
        let mutated_template = template.replace(
            "messages[0]['content'][0]['text']",
            "messages[0]['content']",
        );
        // Workaround: Qwen3 uses Python slice syntax `[::-1]` to reverse lists,
        // which minijinja doesn't support. Use the `reverse` filter instead.
        let mutated_template = mutated_template.replace("[::-1]", "|reverse");
        // Strip `{% generation %}` / `{% endgeneration %}` blocks — these are
        // training-only assistant masking tags (HF `return_assistant_tokens_mask`)
        // and should be ignored at inference. Templates write them with arbitrary
        // whitespace control + spacing, e.g. LFM2.5's `{%- generation -%}`; match
        // every variant rather than a single literal (a literal `.replace` misses
        // the dash form and minijinja then panics on the unknown statement).
        static GENERATION_TAG: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| {
            regex::Regex::new(r"\{\%-?\s*(?:end)?generation\s*-?\%\}").expect("static regex")
        });
        let mutated_template = GENERATION_TAG
            .replace_all(&mutated_template, "")
            .into_owned();

        let template_str = mutated_template.into_boxed_str();
        env.add_function("raise_exception", raise_exception);
        env.add_function("strftime_now", strftime_now);

        // Leak env and template_str as read-only, process-lifetime resources.
        // Each ChatTemplate is created once per model load, so the leak is bounded.
        let template = Box::leak(env)
            .template_from_str(Box::leak(template_str))
            .expect("failed to parse chat template from model metadata");

        // get the list of variables that are used in the template
        let variables = template.undeclared_variables(true);
        // check if the `tools` variable is used in the template
        let use_default_tool_template = !variables.contains("tools");

        Self {
            template,
            bos_token: bos_token.map(|token| token.as_str().to_string()),
            eos_token: eos_token.map(|token| token.as_str().to_string()),
            use_default_tool_template,
        }
    }

    pub(crate) fn apply(
        &self,
        messages: Vec<Message>,
        tools_and_prompt: Option<(Vec<ToolSpec>, String)>,
        enable_thinking: Option<bool>,
    ) -> Result<String> {
        self.apply_with_options(messages, tools_and_prompt, enable_thinking, true)
    }

    /// Probe whether this template accepts a `system` role at message[0].
    ///
    /// Some upstream templates (notably Gemma 2's
    /// `gemma-2-*-it/tokenizer_config.json`) hard-error on `system` via
    /// `{{ raise_exception('System role not supported') }}`. Pio's
    /// `Session::new` injects a system message by default; without this
    /// probe + fold-into-first-user fallback, those models crash at
    /// inference time with `syntax error: System role not supported`.
    /// Surfaced by the 20-turn zoo matrix on gemma-2-2b.
    pub(crate) fn supports_system_role(&self) -> bool {
        let probe = vec![
            Message {
                role: "system".into(),
                body: MessageBody::Content {
                    content: MessageContent::SingleText("probe".into()),
                },
                name: None,
                tool_call_id: None,
            },
            Message {
                role: "user".into(),
                body: MessageBody::Content {
                    content: MessageContent::SingleText("hi".into()),
                },
                name: None,
                tool_call_id: None,
            },
        ];
        self.apply_with_options(probe, None, None, false).is_ok()
    }

    /// Like [`apply`] but exposes `add_generation_prompt`.
    ///
    /// Callers that need a pure prefix render (e.g. the MLX prefix cache, which
    /// tokenizes just the system segment and expects it to be a strict prefix
    /// of the full tokenization) must pass `false` — otherwise templates that
    /// append a generation suffix like `<|turn>model\n` will produce a render
    /// that is NOT a prefix of the full chat.
    pub(crate) fn apply_with_options(
        &self,
        mut messages: Vec<Message>,
        tools_and_prompt: Option<(Vec<ToolSpec>, String)>,
        enable_thinking: Option<bool>,
        add_generation_prompt: bool,
    ) -> Result<String> {
        let tools = match tools_and_prompt {
            Some((tools, tool_prompt)) => {
                // check if the `tools` variable is used in the template
                // if not, we need to append the tools to the last message
                let text = if self.use_default_tool_template {
                    match serde_json::to_string(&tools) {
                        Ok(tools_str) => format!("\n---\n{}\n{}", tools_str, tool_prompt),
                        Err(e) => return Err(Error::from(e)),
                    }
                } else {
                    // if the `tools` variable is used in the template, we just append the tool_prompt
                    format!("\n---\n{}", tool_prompt)
                };
                if let Some(last_message) = messages.last_mut()
                    && let MessageBody::Content { content } = &mut last_message.body
                {
                    content.push(MessageChunk::Text { text });
                }
                Some(tools)
            }
            None => None,
        };

        let messages: Vec<TextMessage> = messages.into_iter().map(|c| c.into()).collect();
        let final_message = messages.last().cloned();
        let mut rendered_template = self.template.render(ChatTemplateInputs {
            messages,
            bos_token: self.bos_token.as_deref(),
            eos_token: self.eos_token.as_deref(),
            add_generation_prompt,
            enable_thinking,
            tools,
        })?;

        // if the last message is from the assistant, continue the generation prompt
        rendered_template = match final_message {
            Some(msg) if msg.role == "assistant" => {
                match rendered_template.rfind(msg.content.as_str()) {
                    // implementation based on feature in transformers pipeline
                    // https://github.com/huggingface/transformers/blob/1cf17077bf2d4affed31387c0943251a4ba8fab7/src/transformers/pipelines/text_generation.py#L418
                    Some(index) => rendered_template[..index + msg.content.len()]
                        .trim_end()
                        .to_string(),
                    None => rendered_template,
                }
            }
            _ => rendered_template,
        };

        Ok(rendered_template)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::message::{MessageContent, Url};

    /// Tracer for `ChatTemplate::supports_system_role`. Gemma 2's
    /// upstream template explicitly raises when given a system message
    /// — Pio's session-start always injected one and crashed inference
    /// with `syntax error: System role not supported`. The matrix
    /// 20-turn run surfaced this on gemma-2-2b. Probe-rendering at
    /// session-start lets us route around it.
    #[test]
    fn supports_system_role_detects_gemma2_style_rejection() {
        // Minimal Gemma-2-style template: errors when role[0] == 'system'.
        let tpl = r#"{%- for m in messages %}{%- if m.role == 'system' %}{{ raise_exception('System role not supported') }}{%- endif %}<turn>{{ m.role }}: {{ m.content }}</turn>{%- endfor %}"#;
        let ct = ChatTemplate::new(tpl.to_string(), None, None);
        assert!(
            !ct.supports_system_role(),
            "must detect Gemma-2-style rejection",
        );
    }

    /// LFM2.5's embedded template marks the assistant span with Jinja
    /// whitespace-control tags `{%- generation -%}` / `{%- endgeneration -%}`
    /// (HF `return_assistant_tokens_mask`). minijinja has no `generation`
    /// statement, so an un-stripped tag panics at parse time → the model loads
    /// "ready" but every generation returns empty. The literal-`.replace`
    /// strip missed the dash form; this regression guards the regex strip
    /// across all whitespace/spacing variants.
    #[test]
    fn strips_generation_masking_tags_all_variants() {
        for tpl in [
            r#"{%- for m in messages %}{%- generation -%}{{ m.content }}{%- endgeneration -%}{%- endfor %}"#,
            r#"{% for m in messages %}{% generation %}{{ m.content }}{% endgeneration %}{% endfor %}"#,
            r#"{%- for m in messages %}{%-generation-%}{{ m.content }}{%-endgeneration-%}{%- endfor %}"#,
        ] {
            // Before the fix this panics in `ChatTemplate::new` (.expect on the
            // minijinja parse). After the fix it parses and renders the content.
            let ct = ChatTemplate::new(tpl.to_string(), None, None);
            let out = ct
                .apply(
                    vec![Message {
                        name: None,
                        tool_call_id: None,
                        role: "user".to_string(),
                        body: MessageBody::Content {
                            content: MessageContent::SingleText("hello".to_string()),
                        },
                    }],
                    None,
                    None,
                )
                .expect("template must render after generation-tag strip");
            assert!(
                out.contains("hello"),
                "rendered content must survive: {out:?}"
            );
        }
    }

    #[test]
    fn supports_system_role_passes_for_lenient_template() {
        // Llama-3-style template happily renders system role.
        let tpl =
            r#"{%- for m in messages %}<|h|>{{ m.role }}<|h|>{{ m.content }}<|e|>{%- endfor %}"#;
        let ct = ChatTemplate::new(tpl.to_string(), None, None);
        assert!(ct.supports_system_role(), "lenient template must pass");
    }

    #[test]
    fn test_chat_template() {
        let template = r#"
        {%- if tools %}
            {{- '<|im_start|>system\n' }}
            {%- if messages[0].role == 'system' %}
                {{- messages[0].content + '\n\n' }}
            {%- endif %}
            {{- " # Tools\n\nYou may call one or more functions to assist with the user query.\n\nYou are provided with function signatures within <tools></tools> XML tags:\n<tools>" }}
            {%- for tool in tools %}
                {{- "\n" }}
                {{- tool | tojson }}
            {%- endfor %}
            {{- "\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{\"name\": <function-name>, \"arguments\": <args-json-object>}\n</tool_call><|im_end|>\n" }}
            {%- else %}
            {%- if messages[0].role == 'system' %}
            {{- '<|im_start|>system\n' + messages[0].content + '<|im_end|>\n' }}
            {%- endif %}
            {%- endif %}
            {%- set ns = namespace(multi_step_tool=true, last_query_index=messages|length - 1) %}
            {%- for forward_message in messages %}
            {%- set index = (messages|length - 1) - loop.index0 %}
            {%- set message = messages[index] %}
            {%- set tool_start = '<tool_response>' %}
            {%- set tool_start_length = tool_start|length %}
            {%- set start_of_message = message.content[:tool_start_length] %}
            {%- set tool_end = '</tool_response>' %}
            {%- set tool_end_length = tool_end|length %}
            {%- set start_pos = (message.content|length) - tool_end_length %}
            {%- if start_pos < 0 %}
            {%- set start_pos = 0 %}
            {%- endif %}
            {%- set end_of_message = message.content[start_pos:] %}
            {%- if ns.multi_step_tool and message.role == "user" and not(start_of_message == tool_start and end_of_message == tool_end) %}
            {%- set ns.multi_step_tool = false %}
            {%- set ns.last_query_index = index %}
            {%- endif %}
            {%- endfor %}
            {%- for message in messages %}
            {%- if (message.role == "user") or (message.role == "system" and not loop.first) %}
            {{- '<|im_start|>' + message.role + '\n' + message.content + '<|im_end|>' + '\n' }}
            {%- elif message.role == "assistant" %}
            {%- set content = message.content %}
            {%- set reasoning_content = '' %}
            {%- if message.reasoning_content is defined and message.reasoning_content is not none %}
            {%- set reasoning_content = message.reasoning_content %}
            {%- else %}
            {%- if '</think>' in message.content %}
            {%- set content = (message.content.split('</think>')|last).lstrip('\n') %}
            {%- set reasoning_content = (message.content.split('</think>')|first).rstrip('\n') %}
            {%- set reasoning_content = (reasoning_content.split('<think>')|last).lstrip('\n') %}
            {%- endif %}
            {%- endif %}
            {%- if loop.index0 > ns.last_query_index %}
            {%- if loop.last or (not loop.last and reasoning_content) %}
            {{- '<|im_start|>' + message.role + '\n<think>\n' + reasoning_content.strip('\n') + '\n</think>\n\n' + content.lstrip('\n') }}
            {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
            {%- else %}
            {{- '<|im_start|>' + message.role + '\n' + content }}
            {%- endif %}
            {%- if message.tool_calls %}
            {%- for tool_call in message.tool_calls %}
            {%- if (loop.first and content) or (not loop.first) %}
            {{- '\n' }}
            {%- endif %}
            {%- if tool_call.function %}
            {%- set tool_call = tool_call.function %}
            {%- endif %}
            {{- '<tool_call>\n{"name": "' }}
                        {{- tool_call.name }}
                        {{- '", "arguments": ' }}
            {%- if tool_call.arguments is string %}
            {{- tool_call.arguments }}
            {%- else %}
            {{- tool_call.arguments | tojson }}
            {%- endif %}
            {{- '}\n</tool_call>' }}
            {%- endfor %}
            {%- endif %}
            {{- '<|im_end|>\n' }}
            {%- elif message.role == "tool" %}
            {%- if loop.first or (messages[loop.index0 - 1].role != "tool") %}
            {{- '<|im_start|>user' }}
            {%- endif %}
            {{- '\n<tool_response>\n' }}
            {{- message.content }}
            {{- '\n</tool_response>' }}
            {%- if loop.last or (messages[loop.index0 + 1].role != "tool") %}
            {{- '<|im_end|>\n' }}
            {%- endif %}
            {%- endif %}
            {%- endfor %}
            {%- if add_generation_prompt %}
            {{- '<|im_start|>assistant\n' }}
            {%- if enable_thinking is defined and enable_thinking is false %}
            {{- '<think>\n\n</think>\n\n' }}
            {%- endif %}
            {%- endif %}
        "#;

        let source = template
            .lines()
            .map(|line| line.trim())
            .collect::<Vec<&str>>()
            .join("");

        let msgs: Vec<Message> = vec![
            Message {
                name: None,
                tool_call_id: None,
                role: "system".to_string(),
                body: MessageBody::Content {
                    content: MessageContent::SingleText(
                        "Youre a helpful assistant! Answer the users question best you can."
                            .to_string(),
                    ),
                },
            },
            Message {
                name: None,
                tool_call_id: None,
                role: "assistant".to_string(),
                body: MessageBody::Content {
                    content: MessageContent::SingleText(
                        "<think> </think> What is the weather like in Brooklyn, New York?".to_string(),
                    ),
                },
            },
            Message {
                name: None,
                tool_call_id: None,
                role: "user".to_string(),
                body: MessageBody::Content {
                    content: MessageContent::MultipleChunks(vec![
                        MessageChunk::Text {
                            text: "I'm already using this supplement ".to_string(),
                        },
                        MessageChunk::ImageUrl {
                            image_url: Url {
                                url:  "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/IMG_3018.JPG".to_string()
                            },
                        },
                        MessageChunk::Text {
                            text: "and I want to use this one too ".to_string()
                        },
                        MessageChunk::ImageUrl {
                            image_url: Url {
                                url: "https://huggingface.co/datasets/merve/vlm_test_images/resolve/main/IMG_3015.jpg".to_string()
                            },
                        },
                        MessageChunk::Text {
                            text: " what are cautions?".to_string()
                        },
                    ]),
                },
            },

        ];

        let ct = ChatTemplate::new(source, None, None);

        let result = ct.apply(msgs, None, None).unwrap();

        println!("{}", result);
    }
}
