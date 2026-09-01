//! A pinned retrieval corpus, so ranking quality cannot degrade quietly.
//!
//! Everything else about tool search has a test: that deferred specs stay out
//! of the prompt, that hydration is idempotent, that Hybrid degrades to
//! lexical without an embedder. None of that notices if the ranking itself
//! gets worse. Change the BM25 tokeniser, adjust an RRF weight, add a
//! stopword, and every existing test still passes while the model stops
//! finding the tool it needs.
//!
//! So: sixty tools, twenty queries, and the tool that should win named for
//! each. The assertion is deliberately loose — the target must appear in the
//! top three, not first — because pinning exact ranks would fail on
//! improvements as readily as on regressions, and a test that cries wolf gets
//! deleted. What it does catch is a target falling out of contention
//! entirely, which is what every one of those changes actually does when it
//! goes wrong.

use std::sync::Arc;

use schemars::JsonSchema;
use serde::Deserialize;

use super::registry::ToolRegistry;
use super::spec::ToolLoading;
use super::{FunctionTool, Tool, ToolOutput, ToolSearch};

/// Sixty plausible tools across seven domains, so a query has to discriminate
/// against neighbours rather than against noise. Several pairs are
/// deliberately close (`git_commit` and `git_push`, `docker_build` and
/// `docker_run`) since near-misses are where a tokeniser change shows up.
const CORPUS: &[(&str, &str)] = &[
    // Version control
    (
        "git_commit",
        "Record staged changes to the repository history",
    ),
    ("git_push", "Upload local commits to a remote repository"),
    ("git_rebase", "Replay commits onto another base branch"),
    (
        "git_bisect",
        "Find the commit that introduced a bug by binary search",
    ),
    (
        "git_stash",
        "Shelve uncommitted changes to work on something else",
    ),
    (
        "github_create_pull_request",
        "Open a pull request for review",
    ),
    (
        "github_review_comment",
        "Leave a review comment on a pull request",
    ),
    (
        "github_merge_branch",
        "Merge an approved pull request into main",
    ),
    // Filesystem
    ("read_file", "Read the contents of a file from disk"),
    ("write_file", "Write contents to a file on disk"),
    ("append_file", "Add lines to the end of an existing file"),
    ("delete_file", "Remove a file from disk permanently"),
    ("list_directory", "List the entries inside a directory"),
    ("move_file", "Rename or relocate a file"),
    ("find_files", "Search for files matching a glob pattern"),
    (
        "grep_contents",
        "Search inside files for a regular expression",
    ),
    // Containers and clusters
    ("docker_build", "Build a container image from a Dockerfile"),
    ("docker_run", "Start a container from an image"),
    (
        "docker_push_image",
        "Upload a container image to a registry",
    ),
    ("kubectl_apply", "Apply a manifest to a Kubernetes cluster"),
    ("kubectl_logs", "Fetch logs from a running pod"),
    ("kubectl_scale", "Change the replica count of a deployment"),
    ("helm_install", "Install a Helm chart into a cluster"),
    (
        "terraform_plan",
        "Preview infrastructure changes before applying",
    ),
    // Images and media
    ("resize_image", "Change the pixel dimensions of an image"),
    ("crop_image", "Cut a rectangular region out of an image"),
    (
        "convert_image_format",
        "Convert an image between PNG, JPEG and WebP",
    ),
    (
        "compress_image",
        "Reduce an image file size with lossy encoding",
    ),
    (
        "extract_video_frames",
        "Pull still frames out of a video file",
    ),
    ("transcode_video", "Re-encode a video into another codec"),
    (
        "transcribe_audio",
        "Produce a text transcript from spoken audio",
    ),
    (
        "normalize_audio_levels",
        "Even out the loudness of an audio track",
    ),
    // Data and queries
    ("run_sql_query", "Execute a SQL query against a database"),
    (
        "explain_query_plan",
        "Show how the database will execute a query",
    ),
    ("create_table", "Define a new table in the database schema"),
    ("run_database_migration", "Apply a pending schema migration"),
    ("export_csv", "Write query results out as a CSV file"),
    ("load_parquet", "Read a Parquet file into a dataframe"),
    (
        "aggregate_metrics",
        "Roll up raw events into summary statistics",
    ),
    (
        "detect_outliers",
        "Flag values far outside the expected distribution",
    ),
    // Network and web
    ("http_get", "Fetch a URL over HTTP and return the body"),
    ("http_post_json", "Send a JSON payload to an HTTP endpoint"),
    ("download_file", "Save a remote file to local disk"),
    ("scrape_page_text", "Extract readable text from a web page"),
    (
        "check_certificate_expiry",
        "Report when a TLS certificate expires",
    ),
    ("resolve_dns", "Look up the DNS records for a hostname"),
    ("ping_host", "Check whether a host responds to ICMP"),
    (
        "port_scan",
        "Report which TCP ports on a host accept connections",
    ),
    // Communication and scheduling
    ("send_email", "Compose and send an email message"),
    ("send_slack_message", "Post a message to a Slack channel"),
    ("create_calendar_event", "Add an event to a calendar"),
    ("find_meeting_slot", "Find a time when everyone is free"),
    ("create_issue", "File a new issue in the tracker"),
    ("assign_issue", "Give an issue to a person to work on"),
    ("close_issue", "Mark an issue as resolved"),
    (
        "post_status_update",
        "Publish a short status note to the team",
    ),
    // Build and release
    ("run_tests", "Execute the project's automated test suite"),
    (
        "run_linter",
        "Check the source for style and correctness warnings",
    ),
    (
        "publish_package",
        "Upload a release to the package registry",
    ),
    ("tag_release", "Create a version tag for a release"),
];

/// Queries phrased the way a model would ask, paired with the tool that must
/// come back. None of them repeats its target's wording exactly, because a
/// search that only works on verbatim names is not doing anything.
const QUERIES: &[(&str, &str)] = &[
    ("open a pull request", "github_create_pull_request"),
    ("resize this image", "resize_image"),
    ("apply a deployment yaml to the cluster", "kubectl_apply"),
    ("what did the pod print", "kubectl_logs"),
    ("save these changes to version control", "git_commit"),
    ("send my commits to the remote", "git_push"),
    ("which commit broke this", "git_bisect"),
    ("read the contents of a file", "read_file"),
    ("remove a file permanently", "delete_file"),
    ("look inside files for a pattern", "grep_contents"),
    ("build a container image", "docker_build"),
    ("turn speech into text", "transcribe_audio"),
    ("run a query against the database", "run_sql_query"),
    ("apply a pending schema migration", "run_database_migration"),
    ("write the results out as csv", "export_csv"),
    ("fetch a url over http", "http_get"),
    (
        "when does the tls certificate expire",
        "check_certificate_expiry",
    ),
    ("post a message to a slack channel", "send_slack_message"),
    ("find a time when everyone is free", "find_meeting_slot"),
    ("execute the automated test suite", "run_tests"),
];

/// How far down the ranking the target may be before this is a regression.
///
/// Three, not one. A search that puts the right tool second is working; a test
/// that demands first place fails on genuine improvements and gets deleted the
/// third time it does.
const ACCEPTABLE_RANK: usize = 3;

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn listed(name: &str, description: &str) -> Arc<dyn Tool> {
    Arc::new(FunctionTool::new(
        name,
        description,
        |_c, _a: NoArgs| async move { Ok(ToolOutput::from("unused")) },
    ))
}

fn registry_from(
    entries: impl Iterator<Item = (&'static str, &'static str)>,
    search: ToolSearch,
) -> ToolRegistry {
    let entries = entries
        .map(|(name, description)| (listed(name, description), ToolLoading::Deferred))
        .collect();
    ToolRegistry::build(entries, Some(search)).expect("the corpus is a valid tool set")
}

fn corpus_registry(search: ToolSearch) -> ToolRegistry {
    registry_from(CORPUS.iter().copied(), search)
}

#[test]
fn lexical_search_finds_the_right_tool_among_sixty() {
    let registry = corpus_registry(ToolSearch::Bm25);
    let mut missed = Vec::new();

    for (query, expected) in QUERIES {
        let found: Vec<String> = registry
            .search(query, ACCEPTABLE_RANK, None)
            .into_iter()
            .map(|s| s.name)
            .collect();
        if !found.iter().any(|n| n == expected) {
            missed.push(format!("{query:?} wanted {expected:?}, got {found:?}"));
        }
    }

    assert!(
        missed.is_empty(),
        "{} of {} queries lost their tool out of the top {ACCEPTABLE_RANK}:\n  {}",
        missed.len(),
        QUERIES.len(),
        missed.join("\n  "),
    );
}

#[test]
fn hybrid_without_an_embedder_ranks_exactly_as_lexical_does() {
    // The documented degradation path. If these ever diverge, Hybrid has
    // started doing something with an empty semantic half rather than falling
    // back to the lexical one.
    let lexical = corpus_registry(ToolSearch::Bm25);
    let hybrid = corpus_registry(ToolSearch::Hybrid);

    for (query, _) in QUERIES {
        let l: Vec<String> = lexical
            .search(query, 5, None)
            .into_iter()
            .map(|s| s.name)
            .collect();
        let h: Vec<String> = hybrid
            .search(query, 5, None)
            .into_iter()
            .map(|s| s.name)
            .collect();
        assert_eq!(
            l, h,
            "without an embedder, Hybrid must be lexical for {query:?}"
        );
    }
}

#[test]
fn a_query_matching_nothing_returns_nothing_rather_than_filler() {
    // Handing the model three irrelevant tools is worse than handing it none:
    // it will call one.
    let registry = corpus_registry(ToolSearch::Bm25);
    let found = registry.search("xyzzy plugh frobnicate", 5, None);
    assert!(
        found.is_empty(),
        "a query with no lexical overlap should match nothing, got {:?}",
        found.iter().map(|s| &s.name).collect::<Vec<_>>(),
    );
}

#[test]
fn common_words_alone_do_not_pull_a_tool_into_the_results() {
    // The stopword regression, concretely. "a file from a disk" once matched
    // "Apply a manifest" on the word "a".
    let registry = corpus_registry(ToolSearch::Bm25);
    let found: Vec<String> = registry
        .search("a the to of and from with", 5, None)
        .into_iter()
        .map(|s| s.name)
        .collect();
    assert!(
        found.is_empty(),
        "stopwords carry no signal but matched {found:?}"
    );
}

#[test]
fn the_ranking_does_not_depend_on_registration_order() {
    // Same tools, reversed. A ranker that leaks insertion order is scoring
    // something other than the query.
    let forward = corpus_registry(ToolSearch::Bm25);
    let backward = registry_from(CORPUS.iter().rev().copied(), ToolSearch::Bm25);

    for (query, expected) in QUERIES {
        let hit = |r: &ToolRegistry| {
            r.search(query, ACCEPTABLE_RANK, None)
                .into_iter()
                .any(|s| s.name == *expected)
        };
        assert_eq!(
            hit(&forward),
            hit(&backward),
            "reversing registration order changed whether {query:?} found {expected:?}",
        );
    }
}

#[test]
fn searching_returns_no_more_than_the_limit() {
    let registry = corpus_registry(ToolSearch::Bm25);
    for limit in [1, 2, 5, 10] {
        assert!(
            registry.search("file", limit, None).len() <= limit,
            "a limit of {limit} must be a limit"
        );
    }
}

/// Queries whose target currently ranks first.
///
/// Pinned as a number rather than a list, because which four fall short is a
/// property of the ranker rather than a contract. Sixteen is what the ranker
/// does today: a drop is a regression worth investigating, and an improvement
/// should raise this line so the gain cannot be lost again silently.
const EXPECTED_TOP_HITS: usize = 16;

#[test]
fn the_top_result_is_right_as_often_as_it_was() {
    let registry = corpus_registry(ToolSearch::Bm25);
    let (hits, misses): (Vec<_>, Vec<_>) = QUERIES.iter().partition(|(query, expected)| {
        registry
            .search(query, 1, None)
            .first()
            .is_some_and(|s| s.name == **expected)
    });

    assert_eq!(
        hits.len(),
        EXPECTED_TOP_HITS,
        "rank-1 accuracy moved from {EXPECTED_TOP_HITS} to {} of {}. If it went down, \
         something in tokenisation or scoring regressed. If it went up, raise \
         EXPECTED_TOP_HITS so the improvement is held. Currently missing: {:?}",
        hits.len(),
        QUERIES.len(),
        misses.iter().map(|(q, _)| q).collect::<Vec<_>>(),
    );
}
