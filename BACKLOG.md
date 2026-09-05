# Backlog

Consumed by the ContainerHub agentic loop (`shared/agentic-loop/`) when this
repo adopts it. The file was previously zero bytes, which reads as "the
protocol exists and the backlog is empty" — it was neither.

## Protocol

- `- [ ]` actionable — the planner may pick it up
- `- [b]` blocked — skipped, and excluded from the pending count, so a
  backlog containing only blocked items still lets the planner run again
- `- [x]` completed — pruned on sight; the history lives in git

## Open

- [b] Instanced normals shade differently from the equivalent node transform.
      `a_non_uniform_instance_scale_shades_like_the_same_node_scale` fails with
      987 differing pixels against a threshold of 40; confirmed pre-existing and
      deterministic. Blocked here: `src/shaders/*.wgsl` are generated artifacts
      and the `forward.slang` source lives in the C++ engine repo.
- [b] The ARM lane cannot go green — `:latest-cross` resolves to an amd64-only
      manifest list. Blocked on ContainerHub
      (`build-runtime-manifest.sh --repair --push-manifest`).
- [ ] Give CI a GPU adapter (software is enough: `mesa-vulkan-drivers` plus
      `KATAGLYPHIS_REQUIRE_GPU=1`) so the ~40 headless golden tests stop
      silently skipping and reporting as passed.

- [ ] The Ubuntu lane can only produce a tarball. ContainerHub's
      `package_archive.sh` writes the tar and stops; its `create_deb()` was
      deleted on 2026-08-08 as unreachable, and `--flatpak-manifest`,
      `--desktop-file` and `--appdata-file` are checked for existence and then
      never read. `PACKAGE_TYPES` now says `tar`, which is honest but narrow.
      Restoring deb/AppImage/Flatpak means changing ContainerHub, not this repo
      -- the packaging/flatpak/ files here are ready and unused.

## Not adopted yet

The loop itself — config, runner wrappers, `scripts/AgenticLoop/` — is not set
up. Copy-and-edit templates live in ContainerHub's
`shared/agentic-loop/templates/`; a consumer supplies this file, a config JSON,
thin runner wrappers, and optionally per-engine system prompts.
