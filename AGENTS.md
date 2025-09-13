# Repository Guidelines

## Project Structure & Module Organization
- `src/lib.rs` — Rust FFI wrappers and helpers.
- `src/bin/` — example binaries: `devicelist`, `capture_preview`, `capture_preview_wgpu`, `capture_preview_screen`.
- `shim/shim.cpp` — C++17 shim exposing a C ABI to DeckLink.
- `include/` — DeckLink SDK headers and `DeckLinkAPIDispatch.cpp`.
- `build.rs` — compiles the shim and links `DeckLinkAPI.framework` (macOS).

## Build, Test, and Development Commands
- Setup toolchain: `rustup default stable`
- Build (debug/release): `cargo build` / `cargo build --release`
- Run examples:
  - `cargo run --bin devicelist`
  - `cargo run --bin capture_preview`
  - `cargo run --bin capture_preview_wgpu`
  - `cargo run --bin capture_preview_screen`
- SDK path (optional): `DECKLINK_SDK_DIR=/path/to/DeckLinkSDK cargo build`
- Format & lint: `cargo fmt --all` and `cargo clippy --all-targets -- -D warnings`

## Coding Style & Naming Conventions
- Rust 2021 edition, 4‑space indentation.
- Naming: `snake_case` for functions/modules, `UpperCamelCase` for types, `SCREAMING_SNAKE_CASE` for consts.
- Keep unsafe/FFI boundaries narrow; document `unsafe` blocks.
- C++ shim: C++17, match existing style; prefer RAII, avoid owning raw pointers, guard shared state with mutex/atomics.

## Testing Guidelines
- Unit tests in-module: `#[cfg(test)] mod tests { ... }` or integration tests under `tests/` (e.g., `tests/devices_test.rs`).
- Use `cargo test` to run. Gate hardware-dependent tests with `#[cfg(target_os = "macos")]` and feature flags if needed.
- Favor deterministic, fast tests; add smoke tests around FFI that don’t require a live signal when possible.

## Commit & Pull Request Guidelines
- Commit style: short present‑tense subject; prefix type when helpful (e.g., `feat:`, `fix:`, `docs:`). Existing history uses `init:`.
- PRs must include: summary, rationale, testing steps (commands), platform info (macOS version), and screenshots/logs for preview apps.
- Keep changes scoped; separate refactors from functional changes.

## Security & Configuration Tips
- macOS required: links `DeckLinkAPI.framework` from `/Library/Frameworks`.
- Do not commit SDK binaries or proprietary assets. Use `DECKLINK_SDK_DIR` for local SDK paths.
- Validate inputs at FFI boundaries; avoid panics across FFI. Document any required environment variables.
