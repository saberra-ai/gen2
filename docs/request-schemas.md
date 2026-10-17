# Request-owned constrained schemas

`RequestSchema<T>` freezes a schemars schema and narrows explicit scalar string
schemas using JSON Pointer into that document. It does not expose mutation,
replace types, follow references implicitly or accept raw regex/grammar fragments.
Existing enum/const domains are intersected; sibling constraints remain. Empty
sets become false schemas, allowing existing enclosing null/empty-list branches
without inventing a permitted ID. Distinct owner/due domains need distinct schema
definitions rather than narrowing one shared definition twice.

Pointers must address schema-valued positions; literal `const`, `default`, `enum`
and example data cannot be refined. Referenced targets must also be schemas.
This initial API refuses negation, conditionals, `oneOf`, `contains`, dependent/
unevaluated applicators, content schemas and alternate identifier/anchor scopes
anywhere in the schema, including definitions. Narrowing those positions can
widen the root language or affect only annotations. Positive `anyOf`/`allOf`,
ordinary object/array/string constraints and root-local pointer references remain.
For empty intersections, references proven impossible through required object
fields, positive-length homogeneous arrays or union/intersection branches become
false at their use sites. This preserves the accepted JSON language while avoiding
llguidance's deferred compilation of impossible definitions. Definitions and
unproven branches stay intact. Cycles are treated conservatively. Traversal is
limited to16,384 schema positions/128 levels; emptiness analysis to100,000 steps
and128 reference levels. Exhausted work refuses preparation before inference.

```rust,ignore
let schema = gen2::RequestSchema::<Draft>::new()?
    .string_enum_at("/$defs/Citation/properties/id", allowed_ids)?;
let system = format!("Use only supplied evidence. Schema: {}", schema.as_json());
let mut session = gen2::Session::new().with_system(system);
let turn = model.turn(&mut session).user(evidence)
    .max_tokens(512).request_schema(&schema)?.full_context();
let cancellation = turn.canceller();
let response = turn.run()?;
// Inspect Stop/cancellation/context receipts, decode as Draft, and validate
// source membership, revision, entailment and application authority natively.
```

The same immutable JSON value goes to instructions and grammar. Each turn owns a
clone; there is no global/thread-local domain state or type-only grammar cache.
`as_json` is read-only. Debug/error text does not include enum values or schemas.
Bindings are bounded to4096 input values per field,1024 bytes per value,64KiB
combined enum input and128 fields; the serialized schema is at most2MiB. These
construction limits do not imply every maximum-size schema fits a model's context
or a mobile memory tier. Native Runtime admission and exact context checks remain.

`Turn::request_schema` requires an explicit positive output budget, `full_context`,
absent/suppressed tools and the qualified local llama.cpp strict path. Unsupported
backends refuse before message staging/token pull; there is no prompt-only
fallback. Weights may be restored by ordinary residency before backend validation.
Missing strict controls refuse before residency restoration. Schema construction
errors do not invoke a model. Native grammar construction/mask failures remain
fail-closed. Later schema setters/`structured` cannot silently replace a prepared
request; existing static `json_schema<T>` behavior otherwise remains compatible.

Cancellation, full response metadata, native token-capacity checks, shared Runtime
admission, no strict warm restore/append/speculation, and actual cleanup ownership
remain unchanged. A cancelled, truncated or malformed object is never a completed
artifact. A grammar-constrained ID can still select irrelevant evidence; string
membership is not proof of factual support, ownership, dates or chronology.

Verification uses public RequestSchema/Runtime/Model/Turn boundaries, then the actual
llguidance matcher through a pinned local model. Cover readonly request isolation,
intersections, false/nullable branches, Unicode/JSON escaping, reference/pointer and
size refusal, schema overwrite conflicts, unsupported strict backends, actual
invalid-generation pressure and native normal-stop/context receipts. Run actual
Pnola nested action/context schemas before product qualification; a simple object
test cannot establish every nested empty-domain case. Keep earlier failures.
