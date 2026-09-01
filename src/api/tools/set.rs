//! Reusable bundles of tools.

use std::sync::Arc;

use super::Tool;

/// Converts anything tool-shaped into a registrable handle.
///
/// Lets registration accept a concrete tool, a boxed one, or the
/// `Arc<dyn Tool>` an MCP or plugin integration hands back, without the caller
/// wrapping each one.
pub trait IntoTool {
    fn into_tool(self) -> Arc<dyn Tool>;
}

impl<T: Tool + 'static> IntoTool for T {
    fn into_tool(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

impl IntoTool for Arc<dyn Tool> {
    fn into_tool(self) -> Arc<dyn Tool> {
        self
    }
}

/// A named bundle of tools, composed once and registered anywhere.
///
/// The point is reuse: a `filesystem` set is resident in a coding agent and
/// deferred in a general assistant, without either agent restating its
/// contents. Loading is decided at registration, not here — deferredness is a
/// property of the agent, not of the tool.
///
/// ```no_run
/// # use pio_gen2::{FunctionTool, ToolSet};
/// # fn t(n: &str) -> FunctionTool<()> { unimplemented!() }
/// let filesystem = ToolSet::new()
///     .add(t("read_file"))
///     .add(t("write_file"))
///     .add(t("list_dir"));
/// ```
#[derive(Default, Clone)]
pub struct ToolSet {
    tools: Vec<Arc<dyn Tool>>,
}

impl ToolSet {
    /// An empty set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add one tool.
    // Named `add` because that is what it does to a set; the std `Add` trait
    // (which would mean set union) is not what a caller wants here.
    #[allow(clippy::should_implement_trait)]
    #[must_use]
    pub fn add(mut self, tool: impl IntoTool) -> Self {
        self.tools.push(tool.into_tool());
        self
    }

    /// Add several — an MCP server's whole surface, say.
    #[must_use]
    pub fn extend(mut self, tools: impl IntoIterator<Item = impl IntoTool>) -> Self {
        self.tools
            .extend(tools.into_iter().map(IntoTool::into_tool));
        self
    }

    /// How many tools the set holds.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Whether the set is empty.
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }

    /// The tools, by name.
    pub fn names(&self) -> Vec<&str> {
        self.tools.iter().map(|t| t.spec().name.as_str()).collect()
    }
}

impl IntoIterator for ToolSet {
    type Item = Arc<dyn Tool>;
    type IntoIter = std::vec::IntoIter<Arc<dyn Tool>>;

    fn into_iter(self) -> Self::IntoIter {
        self.tools.into_iter()
    }
}

impl FromIterator<Arc<dyn Tool>> for ToolSet {
    fn from_iter<I: IntoIterator<Item = Arc<dyn Tool>>>(iter: I) -> Self {
        Self {
            tools: iter.into_iter().collect(),
        }
    }
}

impl std::fmt::Debug for ToolSet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ToolSet")
            .field("tools", &self.names())
            .finish()
    }
}
