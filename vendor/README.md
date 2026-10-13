# llama.cpp dependency pin

The two binding packages are copied from crates.io 0.1.156. See
`../THIRD_PARTY_NOTICES` and `LICENSE-MIT.llama-cpp-rs` for attribution.
The native source is an unmodified Git submodule at
`847f447c310a3784d623a953e6f0bb07f04c0f27`.

Local binding changes: dependency paths; DFlash2 initialization; target/draft
destruction order; common_batch, common_json and MTMD helper API adaptation;
linking the native vendor-hash archive required by MTMD.

Initialize with `git submodule update --init --recursive`. See
`../docs/plans/dflash2-inprocess.md` for configuration and qualification.
This is a workspace-native experiment. Cargo registry publication must not
silently substitute the unpatched 0.1.156 packages for these path dependencies.
