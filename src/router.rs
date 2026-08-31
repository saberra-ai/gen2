//! Inference router — Phase D week 18.
//!
//! Given a generation request ("I want to run model X with parameters Y"),
//! pick which device in the flock actually runs it. Local-first: if the
//! current device can host the requested model, it always wins (privacy +
//! zero network latency). Otherwise we route to a paired peer with
//! enough RAM, the model cached, and a healthy recent heartbeat.
//!
//! The router is a pure function over `(request, local capability, peer
//! list)` — no async, no I/O. Transport (streaming tokens back from the
//! chosen device) lives in `crate::p2p`; this module only picks.
//!
//! **No cloud, no account.** The peer list is populated by
//! [`crate::flock::discovery`] (LAN mDNS) and optionally iroh's relay-
//! based WAN discovery. Identity + authorization is ed25519 + paired
//! pubkeys. The router itself doesn't touch identity — it trusts the
//! caller to only supply already-authenticated peers.

use crate::zoo::{ModelZoo, PlatformBundle, current_platform_id};

/// What a peer device advertises about itself: who it is, what hardware it
/// has, and which models it already holds. This is the router's whole view of
/// the flock — it never touches transport or identity, and trusts the caller
/// to supply only already-authenticated peers.
///
/// Field-for-field the record a host's discovery layer broadcasts (pio-core's
/// `flock::discovery`), so a host converts by moving fields across rather than
/// reshaping. It lives here because placement is what it exists to inform.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeerAdvertisement {
    /// Stable device id (the peer's DID).
    pub did: String,
    /// base64url-encoded ed25519 verifying key.
    pub pubkey: String,
    /// Human-readable device name, for UI ("route ran on *Server*").
    pub name: String,
    /// Platform id in zoo terms — see [`current_platform_id`].
    pub platform: String,
    pub cores: u32,
    pub ram_mb: u32,
    pub vram_mb: u32,
    /// GPU backend label: `"metal"`, `"cuda"`, `"cpu"`, …
    pub gpu: String,
    /// Canonical zoo ids this peer already has cached.
    pub models: Vec<String>,
    /// Fingerprint of the peer's search index, for staleness checks.
    pub idx_fp: String,
    /// Advertising device's protocol version.
    pub ver: String,
}

/// What the caller wants to run. Kept tiny and allocation-free so the
/// hot path in `gen2::controller` can build it per-request.
#[derive(Debug, Clone)]
pub struct RouteRequest<'a> {
    /// Canonical model id from the zoo. `"gemma-4"`, `"llama-3.1-8b"`, …
    pub model_id: &'a str,
    /// User-set priority. `Urgent` skips remote fallback when local can
    /// run at all (e.g. "don't make me wait for a network hop just
    /// because the server has more VRAM"). Default `Normal`.
    pub priority: RoutePriority,
    /// Hard cap on how many tokens we intend to generate. Advisory to the
    /// router for peer-load prediction only.
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RoutePriority {
    #[default]
    Normal,
    Urgent,
}

/// Snapshot of this device's local capability — what the router needs
/// to decide "can I run this model right here?". Distinct from the
/// capability advertised to peers in [`crate::flock::discovery::CapabilitySnapshot`]
/// because here we also expose "models downloaded to local disk" while
/// peer adverts only list models this device is willing to serve.
#[derive(Debug, Clone, Default)]
pub struct LocalCapability {
    pub ram_mb: u32,
    pub vram_mb: u32,
    pub cached_models: Vec<String>,
}

/// The routing decision. `reason` is human-friendly — surfaces to the
/// "Running on: …" UI chip and to debug panels.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RouteDecision<'peer> {
    /// Run on this device.
    Local {
        bundle_backend: String,
        reason: &'static str,
    },
    /// Route to a remote peer. `peer` is a borrowed slot out of the
    /// caller's peer list — the caller then opens (or reuses) a p2p
    /// session with that peer's pubkey and streams the request over.
    Remote {
        peer: &'peer PeerAdvertisement,
        reason: &'static str,
    },
    /// Nobody in the flock can run this right now. Caller falls open to
    /// a smaller local model with a UI notice.
    Fallback { reason: &'static str },
}

/// Pick a target device for `request`. Pure function — no I/O, no
/// syscalls beyond `current_platform_id()` which is compile-time `cfg`.
///
/// Algorithm (described in the end-state plan, week 18):
///
/// 1. **Local-first**. If the bundled zoo has a platform bundle for this
///    device AND the device has the model cached AND RAM fits, choose
///    local. `Urgent` priority stops here even if a beefier remote is
///    available — user wants minimum latency, not best model.
/// 2. **Remote by capability.** Iterate paired peers. Keep only peers
///    whose advert lists the requested `model_id` AND has enough
///    `ram_mb`. Tiebreak by bigger VRAM (implies faster inference), then
///    by most recent presence.
/// 3. **Download-then-local for Normal priority.** If local can
///    physically host (RAM fits) but the model isn't cached, prefer
///    downloading and running local over going remote — privacy wins.
///    Caller is expected to interpret `Local { … }` with a model that
///    isn't cached as "kick off download first."
/// 4. **Fail-open.** Nothing fits → `Fallback`. Caller picks a smaller
///    model and tells the user.
pub fn route<'a>(
    request: &RouteRequest<'_>,
    local: &LocalCapability,
    peers: &'a [PeerAdvertisement],
    zoo: &ModelZoo,
) -> RouteDecision<'a> {
    // 1. Local-first
    let local_bundle = local_bundle_for(request.model_id, zoo);
    let local_fits = local_bundle
        .map(|b| local.ram_mb >= b.min_ram_mb)
        .unwrap_or(false);
    let local_cached = local.cached_models.iter().any(|m| m == request.model_id);

    if local_fits && local_cached {
        return RouteDecision::Local {
            bundle_backend: local_bundle.map(|b| b.backend.clone()).unwrap_or_default(),
            reason: "local device has model cached and enough RAM",
        };
    }

    if request.priority == RoutePriority::Urgent && local_fits {
        return RouteDecision::Local {
            bundle_backend: local_bundle.map(|b| b.backend.clone()).unwrap_or_default(),
            reason: "urgent — forcing local even without cached model",
        };
    }

    // 2. Remote by capability
    if let Some(peer) = best_remote_peer(request, peers) {
        return RouteDecision::Remote {
            peer,
            reason: "remote peer has model + capacity",
        };
    }

    // 3. Download-then-local for Normal priority
    if local_fits {
        return RouteDecision::Local {
            bundle_backend: local_bundle.map(|b| b.backend.clone()).unwrap_or_default(),
            reason: "no remote available — will download model locally then run",
        };
    }

    // 4. Fail-open
    RouteDecision::Fallback {
        reason: "no device can host this model — falling back to smaller model",
    }
}

fn local_bundle_for<'z>(model_id: &str, zoo: &'z ModelZoo) -> Option<&'z PlatformBundle> {
    let entry = zoo.get(model_id)?;
    entry.platforms.get(current_platform_id())
}

/// Rank and pick the best candidate peer. Returns `None` when no peer
/// has the model and enough RAM. Stable: equal-ranked peers break ties
/// on device_id (lexicographic) so the same inputs always produce the
/// same pick.
fn best_remote_peer<'a>(
    request: &RouteRequest<'_>,
    peers: &'a [PeerAdvertisement],
) -> Option<&'a PeerAdvertisement> {
    let mut best: Option<&'a PeerAdvertisement> = None;
    for peer in peers {
        if !peer_can_host(peer, request.model_id) {
            continue;
        }
        best = match best {
            None => Some(peer),
            Some(cur) => Some(if peer_ranks_higher(peer, cur) {
                peer
            } else {
                cur
            }),
        };
    }
    best
}

fn peer_can_host(peer: &PeerAdvertisement, model_id: &str) -> bool {
    peer.models.iter().any(|m| m == model_id)
}

/// Capacity heuristic. Higher VRAM wins first (GPU inference is faster
/// on dedicated video memory); then higher RAM (unified memory on
/// Apple Silicon wins when VRAM is 0 on both sides); then device_id
/// for stable sort.
fn peer_ranks_higher(a: &PeerAdvertisement, b: &PeerAdvertisement) -> bool {
    if a.vram_mb != b.vram_mb {
        return a.vram_mb > b.vram_mb;
    }
    if a.ram_mb != b.ram_mb {
        return a.ram_mb > b.ram_mb;
    }
    a.did < b.did
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_peer(name: &str, ram_mb: u32, vram_mb: u32, models: &[&str]) -> PeerAdvertisement {
        PeerAdvertisement {
            // Distinct per peer: `peer_ranks_higher` breaks exact hardware ties
            // on `did`, so sharing one would make ranking order arbitrary.
            did: format!("did:pio:{name}"),
            pubkey: "test-pubkey".into(),
            name: name.into(),
            platform: current_platform_id().to_string(),
            cores: 8,
            ram_mb,
            vram_mb,
            gpu: if vram_mb > 0 {
                "cuda".into()
            } else {
                "cpu".into()
            },
            models: models.iter().map(|s| s.to_string()).collect(),
            idx_fp: "deadbeef".into(),
            ver: "1".into(),
        }
    }

    fn zoo() -> ModelZoo {
        ModelZoo::bundled()
    }

    #[test]
    fn local_first_when_cached_and_fits() {
        let local = LocalCapability {
            ram_mb: 32 * 1024,
            vram_mb: 0,
            cached_models: vec!["gemma-4".into()],
        };
        let peers = vec![fake_peer("Server", 128 * 1024, 24 * 1024, &["gemma-4"])];
        let decision = route(
            &RouteRequest {
                model_id: "gemma-4",
                priority: RoutePriority::Normal,
                max_tokens: None,
            },
            &local,
            &peers,
            &zoo(),
        );
        assert!(
            matches!(decision, RouteDecision::Local { .. }),
            "local must win when cached"
        );
    }

    #[test]
    fn remote_wins_when_local_lacks_cached_model() {
        let local = LocalCapability {
            ram_mb: 32 * 1024,
            vram_mb: 0,
            cached_models: vec![],
        };
        let peers = vec![fake_peer("Server", 128 * 1024, 24 * 1024, &["gemma-4"])];
        let decision = route(
            &RouteRequest {
                model_id: "gemma-4",
                priority: RoutePriority::Normal,
                max_tokens: None,
            },
            &local,
            &peers,
            &zoo(),
        );
        match decision {
            RouteDecision::Remote { peer, .. } => assert_eq!(peer.name, "Server"),
            other => panic!("expected remote, got {other:?}"),
        }
    }

    #[test]
    fn urgent_pins_local_even_without_cache() {
        let local = LocalCapability {
            ram_mb: 32 * 1024,
            vram_mb: 0,
            cached_models: vec![],
        };
        let peers = vec![fake_peer("Server", 128 * 1024, 24 * 1024, &["gemma-4"])];
        let decision = route(
            &RouteRequest {
                model_id: "gemma-4",
                priority: RoutePriority::Urgent,
                max_tokens: None,
            },
            &local,
            &peers,
            &zoo(),
        );
        assert!(
            matches!(decision, RouteDecision::Local { .. }),
            "urgent must force local"
        );
    }

    #[test]
    fn fallback_when_local_too_small_and_no_peer_has_model() {
        let local = LocalCapability {
            ram_mb: 512, // way under any Gemma-4 bundle's min_ram_mb
            vram_mb: 0,
            cached_models: vec![],
        };
        let peers = vec![fake_peer("Phone", 6 * 1024, 0, &["qwen-2.5-1b"])];
        let decision = route(
            &RouteRequest {
                model_id: "gemma-4",
                priority: RoutePriority::Normal,
                max_tokens: None,
            },
            &local,
            &peers,
            &zoo(),
        );
        assert!(
            matches!(decision, RouteDecision::Fallback { .. }),
            "nobody can host → fallback"
        );
    }

    #[test]
    fn prefers_peer_with_more_vram() {
        let local = LocalCapability {
            ram_mb: 8 * 1024,
            vram_mb: 0,
            cached_models: vec![],
        };
        let peers = vec![
            fake_peer("SmallServer", 32 * 1024, 8 * 1024, &["gemma-4"]),
            fake_peer("BigServer", 64 * 1024, 24 * 1024, &["gemma-4"]),
        ];
        let decision = route(
            &RouteRequest {
                model_id: "gemma-4",
                priority: RoutePriority::Normal,
                max_tokens: None,
            },
            &local,
            &peers,
            &zoo(),
        );
        match decision {
            RouteDecision::Remote { peer, .. } => {
                assert_eq!(peer.name, "BigServer", "higher VRAM wins");
            }
            other => panic!("expected remote, got {other:?}"),
        }
    }

    #[test]
    fn peer_without_model_is_skipped() {
        let local = LocalCapability::default();
        let peers = vec![
            fake_peer("Server-A", 64 * 1024, 24 * 1024, &["llama-3.1-8b"]),
            fake_peer("Server-B", 32 * 1024, 0, &["gemma-4"]),
        ];
        let decision = route(
            &RouteRequest {
                model_id: "gemma-4",
                priority: RoutePriority::Normal,
                max_tokens: None,
            },
            &local,
            &peers,
            &zoo(),
        );
        match decision {
            RouteDecision::Remote { peer, .. } => {
                assert_eq!(peer.name, "Server-B", "only peer with model wins");
            }
            other => panic!("expected remote with Server-B, got {other:?}"),
        }
    }

    #[test]
    fn download_then_local_when_fits_but_not_cached_and_no_peer() {
        let local = LocalCapability {
            ram_mb: 32 * 1024,
            vram_mb: 0,
            cached_models: vec![],
        };
        let peers: Vec<PeerAdvertisement> = vec![];
        let decision = route(
            &RouteRequest {
                model_id: "gemma-4",
                priority: RoutePriority::Normal,
                max_tokens: None,
            },
            &local,
            &peers,
            &zoo(),
        );
        match decision {
            RouteDecision::Local { reason, .. } => {
                assert!(
                    reason.contains("download"),
                    "reason should mention download: {reason}"
                );
            }
            other => panic!("expected local-with-download, got {other:?}"),
        }
    }

    #[test]
    fn tie_break_on_device_id_is_stable() {
        let local = LocalCapability::default();
        // Two peers with identical capability; best_remote_peer must
        // always pick the same one given the same inputs.
        let a = fake_peer("PeerA", 32 * 1024, 8 * 1024, &["gemma-4"]);
        let b = fake_peer("PeerB", 32 * 1024, 8 * 1024, &["gemma-4"]);
        let peers = vec![a.clone(), b.clone()];
        let d1 = route(
            &RouteRequest {
                model_id: "gemma-4",
                priority: RoutePriority::Normal,
                max_tokens: None,
            },
            &local,
            &peers,
            &zoo(),
        );
        let d2 = route(
            &RouteRequest {
                model_id: "gemma-4",
                priority: RoutePriority::Normal,
                max_tokens: None,
            },
            &local,
            &peers,
            &zoo(),
        );
        assert_eq!(d1, d2, "same inputs must produce same pick");
    }

    #[test]
    fn unknown_model_falls_back() {
        let local = LocalCapability {
            ram_mb: 32 * 1024,
            vram_mb: 0,
            cached_models: vec![],
        };
        let decision = route(
            &RouteRequest {
                model_id: "no-such-model-xyz",
                priority: RoutePriority::Normal,
                max_tokens: None,
            },
            &local,
            &[],
            &zoo(),
        );
        assert!(matches!(decision, RouteDecision::Fallback { .. }));
    }
}
