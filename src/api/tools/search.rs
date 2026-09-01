//! Finding a tool among many, without putting every schema in the prompt.

use std::collections::HashMap;

use super::ToolSpec;

/// How deferred tools are found.
///
/// Neither strategy dominates. Lexical matching wins on exact terminology —
/// `github_create_pull_request`, `kubectl`, an argument called `invoice_id` —
/// which is most of what a model actually types when it knows roughly what it
/// wants. Semantic matching wins on intent: "something that can resize an
/// image" shares no words with `thumbnail_generate`. Hybrid runs both and
/// fuses, which is why it's the default.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum ToolSearch {
    /// Lexical BM25 over names, descriptions, and argument names.
    Bm25,
    /// Embedding similarity. Requires an embedder on the engine.
    Semantic,
    /// Both, fused with reciprocal rank fusion.
    #[default]
    Hybrid,
}

impl ToolSearch {
    /// Whether this strategy needs an embedding model loaded.
    pub fn needs_embedder(&self) -> bool {
        matches!(self, Self::Semantic | Self::Hybrid)
    }
}

/// Lexical index over tool specs.
///
/// Okapi BM25 with the usual parameters. Small corpora — tens to low hundreds
/// of tools — so this is built eagerly and searched by scan.
// Consumed by the agent builder, which lands next; the registry below is the
// only caller and is itself not yet wired to an agent.
#[allow(dead_code)]
#[derive(Debug, Default)]
pub(crate) struct Bm25Index {
    /// Per-document term frequencies, parallel to `names`.
    docs: Vec<HashMap<String, usize>>,
    names: Vec<String>,
    doc_len: Vec<usize>,
    /// How many documents contain each term.
    doc_freq: HashMap<String, usize>,
    avg_len: f32,
}

const K1: f32 = 1.2;
const B: f32 = 0.75;

#[allow(dead_code)]
impl Bm25Index {
    pub(crate) fn build(specs: &[ToolSpec]) -> Self {
        let mut idx = Self::default();
        for spec in specs {
            let terms = tokenize(&spec.searchable_text());
            let mut tf: HashMap<String, usize> = HashMap::new();
            for t in &terms {
                *tf.entry(t.clone()).or_default() += 1;
            }
            for t in tf.keys() {
                *idx.doc_freq.entry(t.clone()).or_default() += 1;
            }
            idx.doc_len.push(terms.len());
            idx.docs.push(tf);
            idx.names.push(spec.name.clone());
        }
        let total: usize = idx.doc_len.iter().sum();
        idx.avg_len = if idx.docs.is_empty() {
            0.0
        } else {
            total as f32 / idx.docs.len() as f32
        };
        idx
    }

    /// Tool names ranked by relevance, best first.
    pub(crate) fn search(&self, query: &str, limit: usize) -> Vec<String> {
        let terms = tokenize(query);
        let n = self.docs.len() as f32;

        let mut scored: Vec<(f32, usize)> = (0..self.docs.len())
            .map(|i| {
                let score = terms
                    .iter()
                    .filter_map(|term| {
                        let tf = *self.docs[i].get(term)? as f32;
                        let df = *self.doc_freq.get(term)? as f32;
                        // Standard BM25 idf, +1 inside the log so a term in
                        // every document scores 0 rather than negative.
                        let idf = (((n - df + 0.5) / (df + 0.5)) + 1.0).ln();
                        let len_norm =
                            1.0 - B + B * (self.doc_len[i] as f32 / self.avg_len.max(1.0));
                        Some(idf * (tf * (K1 + 1.0)) / (tf + K1 * len_norm))
                    })
                    .sum::<f32>();
                (score, i)
            })
            .filter(|(s, _)| *s > 0.0)
            .collect();

        // Ties break on the name, not on registration order. Without it,
        // registering an unrelated tool can push a needed one out of the top
        // k, and the same corpus ranks differently depending on how it was
        // assembled.
        scored.sort_by(|a, b| {
            b.0.total_cmp(&a.0)
                .then_with(|| self.names[a.1].cmp(&self.names[b.1]))
        });
        scored
            .into_iter()
            .take(limit)
            .map(|(_, i)| self.names[i].clone())
            .collect()
    }
}

/// Function words carry no signal about which tool is wanted.
///
/// BM25's IDF normally handles these, but a tool registry is a tiny corpus —
/// with a handful of documents, a stopword appearing in one of them scores as
/// though it were meaningful, and a query like "read a file from disk" matches
/// "Apply a manifest" on the word "a".
const STOPWORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "into", "is", "it", "of",
    "on", "or", "that", "the", "then", "this", "to", "with",
];

/// Lowercase alphanumeric terms, splitting snake_case and camelCase.
///
/// Tool names carry meaning in their structure — `github_create_pull_request`
/// should match a query for "pull request", which it can't as one opaque token.
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut prev_lower = false;

    for ch in text.chars() {
        if ch.is_alphanumeric() {
            // camelCase boundary: a capital after a lowercase starts a term.
            if ch.is_uppercase() && prev_lower && !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
            cur.extend(ch.to_lowercase());
            prev_lower = ch.is_lowercase() || ch.is_numeric();
        } else if !cur.is_empty() {
            out.push(std::mem::take(&mut cur));
            prev_lower = false;
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out.retain(|t| !STOPWORDS.contains(&t.as_str()));
    out
}

/// Reciprocal rank fusion of several ranked lists.
///
/// Scores are not comparable across BM25 and cosine similarity, so fusing on
/// rank rather than score is what makes hybrid search sound. `k` damps the
/// influence of top ranks; 60 is the value from the original paper.
pub(crate) fn reciprocal_rank_fusion(rankings: &[Vec<String>], limit: usize) -> Vec<String> {
    const K: f32 = 60.0;
    let mut scores: HashMap<&str, f32> = HashMap::new();

    for ranking in rankings {
        for (rank, name) in ranking.iter().enumerate() {
            *scores.entry(name.as_str()).or_default() += 1.0 / (K + rank as f32 + 1.0);
        }
    }

    let mut fused: Vec<(&str, f32)> = scores.into_iter().collect();
    // Ties broken by name so fusion is deterministic across runs.
    fused.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(b.0)));
    fused
        .into_iter()
        .take(limit)
        .map(|(n, _)| n.to_string())
        .collect()
}

/// Cosine similarity ranking over precomputed embeddings.
pub(crate) fn rank_by_similarity(
    query: &[f32],
    corpus: &[(String, Vec<f32>)],
    limit: usize,
) -> Vec<String> {
    let mut scored: Vec<(f32, &str)> = corpus
        .iter()
        .map(|(name, vec)| (cosine(query, vec), name.as_str()))
        .collect();
    // Ties break on the name, as they do in the lexical ranker and the fusion
    // below, so an identical corpus always ranks identically.
    scored.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(b.1)));
    scored
        .into_iter()
        .take(limit)
        .map(|(_, n)| n.to_string())
        .collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus() -> Vec<ToolSpec> {
        vec![
            ToolSpec::new(
                "github_create_pull_request",
                "Open a pull request against a branch",
                serde_json::json!({"type":"object","properties":{
                    "repository":{"type":"string","description":"owner/name"}}}),
            ),
            ToolSpec::new(
                "thumbnail_generate",
                "Shrink a picture to a smaller preview",
                serde_json::json!({"type":"object","properties":{
                    "path":{"type":"string"}}}),
            ),
            ToolSpec::new(
                "kubectl_apply",
                "Apply a manifest to a cluster",
                serde_json::json!({"type":"object","properties":{
                    "namespace":{"type":"string"}}}),
            ),
        ]
    }

    #[test]
    fn tokenize_drops_function_words() {
        // On a corpus this small, an unfiltered "a" outranks real signal.
        assert_eq!(
            tokenize("read a file from the disk"),
            ["read", "file", "disk"]
        );
        assert!(tokenize("the and of").is_empty());
    }

    #[test]
    fn a_query_of_only_function_words_matches_nothing() {
        let idx = Bm25Index::build(&corpus());
        assert!(idx.search("a the of", 3).is_empty());
    }

    #[test]
    fn tokenize_splits_snake_and_camel_case() {
        // A tool name is only findable if its parts are terms.
        assert_eq!(
            tokenize("github_create_pull_request"),
            ["github", "create", "pull", "request"]
        );
        assert_eq!(tokenize("createPullRequest"), ["create", "pull", "request"]);
        assert_eq!(tokenize("Apply a manifest!"), ["apply", "manifest"]);
    }

    #[test]
    fn bm25_finds_a_tool_by_the_words_in_its_name() {
        let idx = Bm25Index::build(&corpus());
        let hits = idx.search("pull request", 3);
        assert_eq!(
            hits.first().map(String::as_str),
            Some("github_create_pull_request")
        );
    }

    #[test]
    fn bm25_matches_exact_terminology_a_description_never_paraphrases() {
        // This is what lexical search is for: "kubectl" appears nowhere in any
        // prose description, only in the name.
        let idx = Bm25Index::build(&corpus());
        assert_eq!(
            idx.search("kubectl", 3).first().map(String::as_str),
            Some("kubectl_apply")
        );
    }

    #[test]
    fn bm25_matches_argument_names() {
        let idx = Bm25Index::build(&corpus());
        assert_eq!(
            idx.search("namespace", 3).first().map(String::as_str),
            Some("kubectl_apply")
        );
    }

    #[test]
    fn bm25_returns_nothing_rather_than_everything_for_an_unmatched_query() {
        let idx = Bm25Index::build(&corpus());
        assert!(idx.search("quantum entanglement", 3).is_empty());
    }

    #[test]
    fn rrf_promotes_what_both_rankers_agree_on() {
        // `b` is second in each list but the only one both rank, so it should
        // beat items that placed first in just one.
        let lexical = vec!["a".to_string(), "b".to_string()];
        let semantic = vec!["c".to_string(), "b".to_string()];
        let fused = reciprocal_rank_fusion(&[lexical, semantic], 3);
        assert_eq!(fused[0], "b", "agreement outranks a single first place");
    }

    #[test]
    fn rrf_is_deterministic_when_scores_tie() {
        let one = vec!["z".to_string(), "a".to_string()];
        let two = vec!["z".to_string(), "a".to_string()];
        assert_eq!(
            reciprocal_rank_fusion(&[one.clone(), two.clone()], 2),
            reciprocal_rank_fusion(&[one, two], 2)
        );
    }

    #[test]
    fn similarity_ranking_orders_by_cosine() {
        let corpus = vec![
            ("far".to_string(), vec![0.0, 1.0]),
            ("near".to_string(), vec![1.0, 0.0]),
        ];
        assert_eq!(rank_by_similarity(&[1.0, 0.0], &corpus, 2)[0], "near");
    }

    #[test]
    fn hybrid_is_the_default_and_needs_an_embedder() {
        assert_eq!(ToolSearch::default(), ToolSearch::Hybrid);
        assert!(ToolSearch::Hybrid.needs_embedder());
        assert!(!ToolSearch::Bm25.needs_embedder());
    }
}
