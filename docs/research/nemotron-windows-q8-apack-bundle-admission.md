# Local admission of the independent A-packing replay

The repaired independent receiver atcc655ab9 completed successfully in
run39080172371/job117535343984. The original Windows experiment was not rerun.
This status establishes that its frozen saved-artifact pipeline completed;
local delivery and raw numerical/timing interpretation remain separate.

The terminal API preparation fixes result artifact12118043328,17,392,754bytes,
SHA256`0a31a38d6e136bdf48a66b7b106745ab1c175e601e6f44b2af438cbe207bcd16`.
Raw run/job/artifact API records are retained in the
[preparation collection](../../benches/results/nemotron-diarization/windows-x86_64/2026-10-17-q8-apack-bundle-admission-preparation/collection.json).
Fresh API records must still admit these exact successful source/run/job and
artifact identities before download. The reviewed receiver and both transport
protocols are committed and byte-bound before artifact access.

This distinct return route freezes a4GiB local free-space floor,128MiB ZIP and
bundle bounds,192MiB extraction bound and1,000 delivery entries. The original
8GiB full native-archive fetch floor remains unchanged and was enforced on the
fresh receiver. The new floor budgets the bounded ZIP/extraction/quarantine and
two at-most1GiB materialized collections with remaining headroom. Returned
individual Git blobs are capped at128MiB; original native256MiB member admission
is unchanged. No local full native ZIP or source-tree copy is required.

Complete ZIP metadata is checked before extraction. Only declared root files
and command-output members are allowed. Traversal, noncanonical/ambiguous paths,
symlinks, duplicate/case-colliding paths, encryption, unknown files and expansion
overflow reject the package. All data stays in a fresh external task directory.
Duplicate JSON keys reject metadata. The transferred bundle must match its
retained descriptor and exact original native source/run/job/archive provenance.

The Git bundle has one SHA1 HEAD and exactly the fixed verifier-base prerequisite.
Git verifies it and unbundles it into a fresh bare quarantine with read-only
alternates pointing at existing source objects. Lazy fetch, global/system Git
configuration, hooks and automatic maintenance are disabled for this route.
No remote, source ref, source index or source working file is written. Saved
initial/final HEAD, refs and index checks additionally verify this invariant.

The result commit must have precisely the verifier base as its sole parent.
Both result roots must be absent at that base. Every change must be an addition
under the two frozen roots, and every tree member a unique regular100644 blob.
Complete raw tree diffs include directory nodes, so empty trees cannot hide
outside the result roots. Only necessary ancestor-tree updates are allowed;
empty added trees, symlinks/gitlinks, changes elsewhere and oversized blobs/collections reject it.
Git tree paths/sizes are checked before materialization. Only selected blobs are
streamed through bounded owned process groups; the alternate object store is
never enumerated or copied. Per-command output, deadline, signal cleanup and
process reports use the already reviewed common helper.

Materialized collections must exactly match their manifests and sender
member/size/manifest descriptors, including all retained raw vectors. Transport
admission leaves raw numerical, half/packing, timing, observer and static/source
interpretation pending. No imported Python/native executable runs, no feature
branch is advanced automatically and no main promotion occurs.

At preparation this local receiver has not downloaded or accepted the bundle.
Five focused tests exercise exact API identity, strict JSON/path/ZIP admission,
bundle provenance, actual incremental Git transfer and quarantine isolation.
Real temporary repositories additionally reject an outside path, symlink,
corrupted result and an outside empty tree while preserving the source HEAD/index and absence of the
result commit in its object store. Every original quality, human-review,
desktop, fresh-machine, platform, performance and release gate remains open.
