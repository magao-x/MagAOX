# xInstGraph F04: validate graph node handlers before startup

## Objective

Address F04 in the [xInstGraph review](xInstGraph-review.md). The app must not enter `READY` or publish a graph when a configured node has an unknown handler type or a parsed graph node lacks a configured handler. Errors must identify the configuration section and offending `type` value, or the graph node that has no handler.

## Evidence and scope

- `xInstGraph::loadConfigImpl()` reads unused configuration sections, skips sections without `type`, and recognizes only `indiProp`, `pwrOnOff`, `fsm`, `stdMotion`, and `static` (`apps/xInstGraph/xInstGraph.hpp`). An unrecognized value leaves `xn` null and is silently skipped. The function does not compare the resulting `m_nodes` handler map with `m_graph.nodes()`. `appStartup()` can then publish and set `READY`.
- The draw.io parser exposes parsed nodes through `instGraph::nodes()`. A `static` handler is the explicit way to represent a node without INDI updates, so it counts as a configured handler. The existing `xigNode` constructor already rejects a handler section whose name is absent from the graph, but its exception is late and does not identify the configuration problem as clearly.
- F01/F03 and F02 already stage and publish output with ownership checks. F04 should use those paths unchanged. No instGraph library API or installed-library change is needed.

## Validation contract

1. Keep the five handler type names exact and case sensitive. Reject empty, misspelled, or otherwise unsupported values. Include the section name and literal value in the diagnostic; for an empty value, say that `type` is empty. Do not silently treat an unknown value as `static`.
2. Every node in `m_graph.nodes()` must have one configuration section with a supported `type`. Distinguish a missing section from a section that exists but lacks `type` where possible, and name the graph node in the error. A `static` handler satisfies this requirement even though it may subscribe to no INDI property.
3. A section with a `type` must name a parsed graph node. Reject a stray or misspelled section with its section name and value. Continue to ignore unrelated application sections that have no `type` and do not match a graph node.
4. Validate the complete section-to-graph mapping before `createStage()`, allocating node handlers, registering callbacks, hiding puts/links, or publishing the output. On validation failure, `loadConfigImpl()` returns failure, `loadConfig()` sets shutdown, and normal startup never reaches `READY`. The input and any pre-existing output remain unchanged, and no staging file remains.

## Implementation plan

1. In `apps/xInstGraph/xInstGraph.hpp`, after loading the XML and checking the output path, collect candidate node sections from `unusedSections()`. Check the return values of `unusedSections()` and each `configUnused()` read used for validation. Build a validated section/name-to-type inventory while preserving the section order used for handler construction.
2. In that pass, reject unsupported or empty types and typed sections absent from `m_graph.nodes()`. Compare the validated inventory with every parsed graph node to find missing sections or missing `type` keys. Report the first failure through the existing `software_error` path with enough context to fix the config file. Check the result of handler-map insertion when constructing nodes so a validated node cannot be silently omitted.
3. Create staging and construct handlers only after validation succeeds. Keep existing handler-specific `loadConfig()` calls and output publication behavior. Retain concrete-type ownership of each newly constructed handler until its configuration and map insertion succeed; `xigNode` has no virtual destructor, so do not introduce deletion through a base pointer. Keep broader handler-map ownership work under F10.
4. Update `apps/xInstGraph/README.md` to state that every draw.io graph node needs a matching section and one of the five exact types, including `static` for intentionally fixed nodes. State that a typed section absent from the graph is an error.
5. Follow the MagAOX `AGENTS.md` documentation pass for every changed C++ file, preserve the header-only app pattern, run `clang-format`, and commit functional changes before documentation-only cleanup.

## Acceptance tests

- Extend `apps/xInstGraph/tests/xInstGraph_test.cpp` using its isolated draw.io/config fixtures. Cover an unknown value, an empty value, and a case-mismatched value. Assert startup is rejected, diagnostics include section and value, no output or staging file is created, and source bytes are unchanged.
- Cover a graph node with no section and one whose section lacks `type`; each must fail with the missing node named. Add a typed config section for a nonexistent graph node and verify its diagnostic. An unrelated section without `type` must remain permitted.
- Retain a positive graph containing all five supported handlers, including `static`, and verify initial publication and `READY` still work. Keep the F01/F03 input/output alias and clobber tests and the F02 callback publication tests passing.
- Build `apps/xInstGraph`, run the focused Catch2 app test via `make -C tests -f Makefile.one t=../apps/xInstGraph/tests/xInstGraph_test.cpp` with the matching installed instGraph library, and check Doxygen links for new test cases. Review the diff and worktree before committing.

## Deployment and follow-up boundary

This makes incomplete operational configurations fail startup where they previously appeared ready. Before deploying, compare each production draw.io node ID with its config section and assign `type=static` explicitly for fixed nodes. Handler-specific key validation remains with each handler; F05 and other configuration findings remain separate work.

## Execution notes (2026-09-28)

- Added a preflight validation pass before staging. It rejects unsupported or empty types, typed sections absent from the graph, and graph nodes lacking a typed section. Unrelated untyped sections remain allowed. Handler construction uses concrete-type ownership until insertion succeeds.
- Extended the app Catch2 suite for each rejection and a valid static node alongside an unrelated section. The focused suite passed 13 cases and 204 assertions against the matching instGraph library installed under `/tmp`. The app also built against that library; the system `/usr/local` instGraph install remains older and lacks the F02 API.
- HTML generated with the project Doxyfile and focused inputs contains all four new test cases and four `Referenced by` links from `validateNodeConfig()`. The project Doxyfile now defines the test-only `XINSTGRAPH_TEST_DOXYGEN_REF` macro.
