use gen2::RequestSchema;
use serde::Deserialize;

#[derive(Deserialize, gen2::schemars::JsonSchema)]
struct Selection {
    #[schemars(length(min = 1, max = 24))]
    id: String,
}

#[test]
fn request_domains_are_immutable_intersections_and_empty_is_not_unconstrained() {
    let decoded: Selection = serde_json::from_str(r#"{"id":"w0-e0"}"#).unwrap();
    assert_eq!(decoded.id, "w0-e0");
    let original = RequestSchema::<Selection>::new().unwrap();
    let first = original
        .clone()
        .string_enum_at("/properties/id", ["w0-e0", "w0-e1", "w0-e0"])
        .unwrap();
    assert_eq!(
        first.as_json()["properties"]["id"]["enum"],
        serde_json::json!(["w0-e0", "w0-e1"])
    );
    assert_eq!(first.as_json()["properties"]["id"]["minLength"], 1);
    assert_eq!(first.as_json()["properties"]["id"]["maxLength"], 24);
    assert!(original.as_json()["properties"]["id"].get("enum").is_none());
    assert!(
        first
            .clone()
            .string_enum_at("/properties/id", ["other"])
            .is_err()
    );
    let empty = original
        .clone()
        .string_enum_at("/properties/id", std::iter::empty::<String>())
        .unwrap();
    assert_eq!(empty.as_json()["properties"]["id"], false);
    assert!(
        original
            .clone()
            .string_enum_at("/absent", ["w0-e0"])
            .is_err()
    );
    assert!(original.string_enum_at("", ["w0-e0"]).is_err());
}

struct SchemaFixture<const KIND: u8>;
impl<const KIND: u8> gen2::schemars::JsonSchema for SchemaFixture<KIND> {
    fn schema_name() -> std::borrow::Cow<'static, str> {
        format!("Fixture{KIND}").into()
    }
    fn json_schema(_: &mut gen2::schemars::SchemaGenerator) -> gen2::schemars::Schema {
        match KIND {
            0=>serde_json::json!({"type":"string","enum":["a","b"],"minLength":1}),
            1=>serde_json::json!({"type":"string","const":"same","pattern":"^same$"}),
            2=>serde_json::json!({"type":"object","properties":{"child":{"$ref":"https://invalid.example/remote"}}}),
            3=>serde_json::json!({"type":"array","items":[{"$ref":"#/$defs/missing"}]}),
            4=>serde_json::json!({"type":"object","properties":{"$ref":{"type":"string"}}}),
            5=>serde_json::json!({"type":"object","const":{"type":"string"},"examples":[{"type":"string"}],"properties":{"const":{"type":"string"}}}),
            6 | 7 => {
                let count = if KIND == 6 { 16 } else { 4 };
                let mut defs = serde_json::Map::new();
                for index in 0..count {
                    let next = format!("#/$defs/n{}", index + 1);
                    defs.insert(format!("n{index}"), serde_json::json!({"type":"object","properties":{"left":{"$ref":next},"right":{"$ref":next}},"required":["left","right"]}));
                }
                defs.insert(format!("n{count}"), serde_json::json!({"type":"string"}));
                serde_json::json!({"type":"object","properties":{"id":{"type":"string"},"tree":{"$ref":"#/$defs/n0"}},"$defs":defs})
            },
            8=>serde_json::json!({"type":"string","not":{"type":"string"}}),
            9=>serde_json::json!({"type":"string","oneOf":[{"type":"string"},{"type":"string"}]}),
            10=>serde_json::json!({"$defs":{"A":{"type":"string"}},"not":{"$ref":"#/$defs/A"}}),
            11=>serde_json::json!({"type":"array","contains":{"type":"string"},"maxContains":0}),
            12=>serde_json::json!({"if":{"type":"string"},"then":false,"else":true}),
            _=>unreachable!(),
        }.try_into().unwrap()
    }
}

#[test]
fn existing_domains_constants_and_reference_locations_cannot_be_widened_or_ignored() {
    let schema = RequestSchema::<SchemaFixture<0>>::new()
        .unwrap()
        .string_enum_at("", ["b", "c"])
        .unwrap();
    assert_eq!(schema.as_json()["enum"], serde_json::json!(["b"]));
    assert_eq!(schema.as_json()["minLength"], 1);
    let fixed = RequestSchema::<SchemaFixture<1>>::new().unwrap();
    let same = fixed.clone().string_enum_at("", ["same", "other"]).unwrap();
    assert_eq!(same.as_json()["enum"], serde_json::json!(["same"]));
    assert_eq!(same.as_json()["pattern"], "^same$");
    assert_eq!(
        fixed.string_enum_at("", ["other"]).unwrap().as_json(),
        &serde_json::Value::Bool(false)
    );
    assert!(RequestSchema::<SchemaFixture<2>>::new().is_err());
    assert!(RequestSchema::<SchemaFixture<3>>::new().is_err());
    assert!(RequestSchema::<SchemaFixture<4>>::new().is_ok());
    assert!(RequestSchema::<SchemaFixture<8>>::new().is_err());
    assert!(RequestSchema::<SchemaFixture<9>>::new().is_err());
    assert!(RequestSchema::<SchemaFixture<10>>::new().is_err());
    assert!(RequestSchema::<SchemaFixture<11>>::new().is_err());
    assert!(RequestSchema::<SchemaFixture<12>>::new().is_err());
    let literals = RequestSchema::<SchemaFixture<5>>::new().unwrap();
    assert!(literals.clone().string_enum_at("/const", ["x"]).is_err());
    assert!(
        literals
            .clone()
            .string_enum_at("/examples/0", ["x"])
            .is_err()
    );
    assert!(literals.string_enum_at("/properties/const", ["x"]).is_ok());
}

#[test]
fn request_preparation_has_bounded_domains_and_shared_reference_analysis() {
    let base = RequestSchema::<Selection>::new().unwrap();
    assert!(
        base.clone()
            .string_enum_at("/properties/id", ["x".repeat(1025)])
            .is_err()
    );
    assert!(
        base.clone()
            .string_enum_at("/properties/id", std::iter::repeat_n("x", 4097))
            .is_err()
    );
    assert!(
        base.clone()
            .string_enum_at("/properties/id", std::iter::repeat_n("x".repeat(1024), 65))
            .is_err()
    );
    let sensitive = base
        .string_enum_at("/properties/id", ["private-id"])
        .unwrap();
    assert!(!format!("{sensitive:?}").contains("private-id"));
    assert!(
        RequestSchema::<SchemaFixture<7>>::new()
            .unwrap()
            .string_enum_at("/properties/id", std::iter::empty::<String>())
            .is_ok()
    );
    assert!(
        RequestSchema::<SchemaFixture<6>>::new()
            .unwrap()
            .string_enum_at("/properties/id", std::iter::empty::<String>())
            .is_err()
    );
}

#[cfg(feature = "backend-llamacpp")]
#[test]
#[ignore = "explicit pinned native Runtime/Model/Turn enum and empty-domain qualification"]
fn native_request_domains_override_invalid_pressure_and_remain_request_owned() {
    use gen2::{GenerationOptions, Runtime, Session, output::FinishReason};
    let mut settings = gen2::advanced::generation::Settings::default();
    settings.system.ctx_size = Some(2048);
    settings.system.gpu_layers = Some(99);
    settings.prompt.include_meta = Some(false);
    let runtime = Runtime::builder().settings(settings).build().unwrap();
    let model = runtime
        .load(std::env::var("PIO_TEST_MODEL").expect("explicit pinned GGUF path"))
        .unwrap();
    let first = RequestSchema::<Selection>::new()
        .unwrap()
        .string_enum_at("/properties/id", ["native-alpha-71"])
        .unwrap();
    let second = RequestSchema::<Selection>::new()
        .unwrap()
        .string_enum_at("/properties/id", ["café-\"β\""])
        .unwrap();
    for (schema, expected) in [(&first, "native-alpha-71"), (&second, "café-\"β\"")] {
        let mut session = Session::new();
        // The allowed value is NOT in the prompt. Only the actual native grammar
        // knows it, so prompt compliance cannot manufacture this observation.
        let response = model
            .turn(&mut session)
            .user("Return exactly {\"id\":\"forbidden\"}. Never change that id.")
            .options(GenerationOptions::new().max_tokens(64).greedy())
            .request_schema(schema)
            .unwrap()
            .full_context()
            .run()
            .unwrap();
        assert_eq!(*response.finish_reason(), FinishReason::Stop);
        assert_eq!(
            serde_json::from_str::<Selection>(&response.text())
                .unwrap()
                .id,
            expected
        );
        assert!(response.stats().request_context().is_some());
        assert_eq!(session.shed(), 0);
    }
    #[derive(Deserialize, gen2::schemars::JsonSchema)]
    struct Optional {
        selected: Option<Selection>,
    }
    let schema = RequestSchema::<Optional>::new()
        .unwrap()
        .string_enum_at(
            "/$defs/Selection/properties/id",
            std::iter::empty::<String>(),
        )
        .unwrap();
    let mut session = Session::new();
    let response = model
        .turn(&mut session)
        .user("Return selected with id forbidden. Never use null.")
        .max_tokens(64)
        .request_schema(&schema)
        .unwrap()
        .full_context()
        .run()
        .unwrap();
    assert_eq!(*response.finish_reason(), FinishReason::Stop);
    assert!(
        serde_json::from_str::<Optional>(&response.text())
            .unwrap()
            .selected
            .is_none()
    );
    let impossible = RequestSchema::<Selection>::new()
        .unwrap()
        .string_enum_at("/properties/id", std::iter::empty::<String>())
        .unwrap();
    let mut session = Session::new();
    let error = model
        .turn(&mut session)
        .user("Return an id anyway.")
        .max_tokens(64)
        .request_schema(&impossible)
        .unwrap()
        .full_context()
        .run()
        .unwrap_err();
    assert!(error.to_string().contains("grammar"), "{error}");
}
