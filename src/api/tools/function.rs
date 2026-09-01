//! [`FunctionTool`] — a tool from a typed async closure.

use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use async_trait::async_trait;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;

use super::{ExecutionPolicy, Risk, Tool, ToolContext, ToolError, ToolOutput, ToolSpec};

type Handler<A> = Box<
    dyn for<'a> Fn(
            &'a ToolContext,
            A,
        )
            -> Pin<Box<dyn Future<Output = Result<ToolOutput, ToolError>> + Send + 'a>>
        + Send
        + Sync,
>;

/// A tool built from a handler that takes a typed argument struct.
///
/// The schema the model sees is derived from `A`, and `A` is what the handler
/// receives — so the declared arguments and the code reading them come from one
/// declaration and cannot drift.
///
/// ```no_run
/// # use gen2::{FunctionTool, ToolOutput};
/// # use gen2::schemars::JsonSchema;
/// # use serde::Deserialize;
/// #[derive(Deserialize, JsonSchema)]
/// struct WeatherArgs {
///     /// City to look up.
///     city: String,
/// }
///
/// let weather = FunctionTool::new(
///     "get_weather",
///     "Current weather for a city",
///     |_ctx, args: WeatherArgs| async move {
///         Ok(ToolOutput::from(format!("18C in {}", args.city)))
///     },
/// );
/// ```
pub struct FunctionTool<A> {
    spec: ToolSpec,
    handler: Handler<A>,
    policy: ExecutionPolicy,
    risk: Risk,
    _args: PhantomData<fn(A)>,
}

impl<A> FunctionTool<A>
where
    A: JsonSchema + DeserializeOwned + Send + 'static,
{
    /// Build a tool. The argument schema comes from `A`.
    pub fn new<F, Fut>(name: impl Into<String>, description: impl Into<String>, f: F) -> Self
    where
        F: for<'a> Fn(&'a ToolContext, A) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<ToolOutput, ToolError>> + Send + 'static,
    {
        let schema = serde_json::to_value(schemars::schema_for!(A))
            .unwrap_or_else(|_| serde_json::json!({"type": "object"}));

        Self {
            spec: ToolSpec::new(name, description, schema),
            handler: Box::new(move |ctx, args| Box::pin(f(ctx, args))),
            policy: ExecutionPolicy::default(),
            risk: Risk::Safe,
            _args: PhantomData,
        }
    }

    /// Declare how this tool may be scheduled.
    #[must_use]
    pub fn with_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// Declare that this tool needs approval under
    /// [`ApprovalMode::AskOnRisky`](crate::ApprovalMode::AskOnRisky).
    ///
    /// ```
    /// # use gen2::{FunctionTool, ToolOutput};
    /// # use schemars::JsonSchema;
    /// # #[derive(serde::Deserialize, JsonSchema)]
    /// # struct Path { path: String }
    /// let delete = FunctionTool::new("delete_file", "Delete a file", |_c, a: Path| async move {
    ///     Ok(ToolOutput::from(format!("deleted {}", a.path)))
    /// })
    /// .risky();
    /// ```
    #[must_use]
    pub fn risky(mut self) -> Self {
        self.risk = Risk::Risky;
        self
    }
}

#[async_trait]
impl<A> Tool for FunctionTool<A>
where
    A: JsonSchema + DeserializeOwned + Send + Sync + 'static,
{
    fn spec(&self) -> &ToolSpec {
        &self.spec
    }

    fn risk(&self) -> Risk {
        self.risk
    }

    async fn call(
        &self,
        ctx: &ToolContext,
        args: serde_json::Value,
    ) -> Result<ToolOutput, ToolError> {
        // Deserialization is where a model's malformed arguments surface. Report
        // it as InvalidArguments so the loop hands it back for correction
        // rather than treating it as a dead tool.
        let typed: A =
            serde_json::from_value(args).map_err(|e| ToolError::InvalidArguments(e.to_string()))?;
        (self.handler)(ctx, typed).await
    }

    fn execution_policy(&self) -> ExecutionPolicy {
        self.policy
    }
}

impl<A> std::fmt::Debug for FunctionTool<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FunctionTool")
            .field("name", &self.spec.name)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Deserialize, JsonSchema)]
    struct WeatherArgs {
        /// City to look up.
        city: String,
    }

    fn weather() -> FunctionTool<WeatherArgs> {
        FunctionTool::new(
            "get_weather",
            "Current weather",
            |_ctx, a: WeatherArgs| async move { Ok(ToolOutput::from(format!("18C in {}", a.city))) },
        )
    }

    #[test]
    fn the_schema_comes_from_the_argument_type() {
        // This is the whole point: nobody wrote this schema by hand, so it
        // cannot disagree with the struct the handler deserializes.
        let spec = weather().spec().clone();
        let props = &spec.input_schema["properties"];
        assert!(props.get("city").is_some(), "got {}", spec.input_schema);
        assert_eq!(
            spec.input_schema["required"][0], "city",
            "a non-Option field is required"
        );
        // The doc comment travels into the schema, so search can index it.
        assert!(spec.searchable_text().contains("City to look up"));
    }

    #[tokio::test]
    async fn a_call_deserializes_into_the_typed_argument() {
        let ctx = ToolContext::new("s1");
        let out = weather()
            .call(&ctx, serde_json::json!({"city": "Paris"}))
            .await
            .unwrap();
        assert_eq!(out.to_model_text(), "18C in Paris");
    }

    #[tokio::test]
    async fn malformed_arguments_are_reported_as_model_actionable() {
        // A small model omitting a required field must get a correctable error,
        // not look like a broken tool.
        let ctx = ToolContext::new("s1");
        let err = weather()
            .call(&ctx, serde_json::json!({"town": "Paris"}))
            .await
            .unwrap_err();
        assert!(matches!(err, ToolError::InvalidArguments(_)));
        assert!(err.is_model_actionable());
    }

    #[test]
    fn a_policy_can_be_declared_per_tool() {
        let t = weather().with_policy(ExecutionPolicy::gpu_bound());
        assert!(t.execution_policy().blocks_inference);
    }
}
