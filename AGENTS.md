Follow these code style and documentation rules exactly.

1) File-Level Documentation
  - Each header/source should have a top Doxygen file block:
  - \file
  - \brief
  - \author (if project uses it)

2) Include Guards and Includes
  - Match existing project include-guard naming convention.
  - Keep include ordering consistent with project style.
  - Do not introduce new include style unless project already uses it.

3) Function Declaration Documentation
  - Every function declaration must have a brief `///` summary.
  - Add a short `/** ... */` details block only when needed.
  - Document every parameter inline at declaration site using:
    - `type name /**< [in] description */`
  - Apply to normal methods, constructors, slots, and signals.
  - Keep return-value docs where project uses them.

4) Member Variable Documentation
  - Document non-trivial class members with `///`.
  - Describe role/ownership/state, not just type.

5) Naming and Structure
  - Use project member naming convention (e.g. `m_` prefix).
  - Keep declaration ordering/grouping stable:
  - public/protected/private
  - slots/signals grouped consistently.
  - Leave a blank line between declarations for readability.

6) Header vs Source Placement
  - Keep non-trivial definitions out of headers.
  - Move implementations to `.cpp` unless intentionally inline.

7) Editing Discipline
  - Preserve existing behavior unless explicitly requested.
  - When renaming members/APIs, update all dependent call sites.
  - Keep changes minimal and scoped.

8) Formatting and Verification
  - Run `clang-format` on touched files.
  - Ensure docs and naming are consistent after formatting.
  - Report any places where project style is ambiguous before making assumptions.

9) Doxygen Named Section Ordering
  - For classes that expose configuration via member data + accessors, keep named sections split into:
  - `... - Data` for protected/private member state
  - `...` (without `- Data`) for public access functions
  - Place the `... - Data` section before the corresponding public accessor section.

10) Header Declaration Parameter Docs
  - In headers, prefer inline parameter documentation on declarations (`type name /**< ... */`) rather than separate `\param` lists, unless there is a specific reason to deviate.

11) PR Prompt Attribution
- At the top of PR descriptions, include an explicit attribution line when work was performed with an AI agent.
- Preferred format:
  - `This work was performed by <current agent model name> in response to the prompt: "...".`
- Substitute the actual model name used for the work instead of hardcoding a specific release name.
- Include the primary user prompt verbatim (or a faithful condensed version if it is extremely long).
- Provide the PR description in a copyable md text block

12) Branch Naming (MagAOX)
  - Feature branches must be namespaced by the active username.
  - Format: `<username>/<feature-name>`
  - Example: `jrmales/gui-segfault-fixes`
  - For other users, substitute their username in place of `jrmales`.

13) Class Declarations vs Definitions
  - Do not include non-trivial function definitions inside class declarations.
  - Put function definitions below the class declaration (or in `.cpp`), preserving behavior.

14) Scoped Block Comments
  - For any `{}` block used only to control lock/mutex lifetime, annotate the opening brace as:
    - `{ //mutex scope`

15) Changed File Documentation Pass
  - When a file is touched, update documentation quality across the full changed file, not only in modified lines.

16) Keep This File Current
  - Add new standing style/documentation instructions to this `AGENTS.md` file as they are introduced.

17) Commit Branch Discipline
  - Do not commit directly to shared/integration branches (e.g. `dev`, `main`, `master`).
  - Always create/switch to a feature branch that follows rule #12 before committing.

18) dev Base-Class Macros
  - When integrating a `dev::<base-class>` helper into an app, prefer the corresponding interface macros provided by that helper header when they exist.
  - Examples include `TELEMETER_*` for `dev::telemeter` and `FRAMEGRABBER_*` for `dev::frameGrabber`.

19) Commit Separation
  - Keep changes separated into clean commits on feature branches.
  - Make functional changes first.
  - Keep relevant plan files up to date as implementation progresses when they capture engineering decisions or execution notes for the work.
  - Commit relevant plan files with the associated code changes when they serve as part of the engineering record for that work.
  - Follow with documentation-only changes.
  - Make formatting-only cleanup a separate final commit when needed.

When you finish:
- Summarize what changed.
- List affected files.
- Note any follow-up items or potential edge cases.
