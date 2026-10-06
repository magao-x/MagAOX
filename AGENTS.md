Follow these code style and documentation rules exactly.

0) Agents/LLMs should never be installed or execute on the actual instrument hardware.
  - users should run such tools on their local workstations/laptops
  - This prohibition includes exao1(AOC), exao2(RTC), exao3(ICC) and exao5 (ROC) on MagAO-X.  It includes similar-function machines in other projects.
  - This includes running commands such as getINDI and setINDI when an INDI tunnel is open to any instrument commputer
  - Agents/LLMs should instead ask the user to run any such commands / hardware steps and report the results.

1) Style, Reuse of Code, and Brevity 
  - Follow the coding style of this repo.  Specific style points are listed below but are not exhaustive.
  - Favor brevity in code
  - Reuse code whenever possible.  
  - Check for similar problems in this repo that have already been solved.  Use existing base classes when possible.  
  - Propose new base classes or common library utilities if similar problems exist in other contexts but have not been generalized
  - Use solutions in dependencies, such as mxlib, gsl, blas/lapack, cuda, whenever possible.

2) File-Level Documentation
   - Each header/source should have a top Doxygen file block:
   - \file
   - \brief
   - Do not use \author and remove it from touched files

3) Include Guards and Includes
  - Match existing project include-guard naming convention.
  - Keep include ordering consistent with project style.
  - Do not introduce new include style unless project already uses it.

4) Function Declaration Documentation
  - Every function declaration must have a brief `///` summary.
  - Add a short `/** ... */` details block only when needed.
  - Document every parameter inline at declaration site using:
    - `type name /**< [in] description */`
  - Apply to normal methods, constructors, slots, and signals.
  - Keep return-value docs where project uses them.

5) Member Variable Documentation
  - Document non-trivial class members with `///`.
  - Describe role/ownership/state, not just type.

6) Naming and Structure
  - Use project member naming convention (e.g. `m_` prefix).
  - Keep declaration ordering/grouping stable:
  - public/protected/private
  - slots/signals grouped consistently.
  - Leave a blank line between declarations for readability.

7) Header vs Source Placement
  - Keep non-trivial definitions out of headers.
  - Move implementations to `.cpp` unless intentionally inline.

8) Editing Discipline
  - Preserve existing behavior unless explicitly requested.
  - When renaming members/APIs, update all dependent call sites.
  - Keep changes minimal and scoped.

9) Formatting and Verification
  - Run `clang-format` on touched files.
  - Ensure docs and naming are consistent after formatting.
  - Report any places where project style is ambiguous before making assumptions.

10) Doxygen Named Section Ordering
  - For classes that expose configuration via member data + accessors, keep named sections split into:
  - `... - Data` for protected/private member state
  - `...` (without `- Data`) for public access functions
  - Place the `... - Data` section before the corresponding public accessor section.

11) Header Declaration Parameter Docs
  - In headers, prefer inline parameter documentation on declarations (`type name /**< ... */`) rather than separate `\param` lists, unless there is a specific reason to deviate.

12) Class Declarations vs Definitions
  - Do not include non-trivial function definitions inside class declarations.
  - Put function definitions below the class declaration (or in `.cpp`), preserving behavior.

13) Scoped Block Comments
  - For any `{}` block used only to control lock/mutex lifetime, annotate the opening brace as:
    - `{ //mutex scope`

14) Branch Naming (MagAOX)
  - Feature branches must be namespaced by the active username.
  - Format: `<username>/<feature-name>`
    - Example: `jrmales/gui-segfault-fixes`
    - For other users, substitute their username in place of `jrmales`.

15) Commit Branch Discipline
  - Do not commit directly to shared/integration branches (i.e. `dev` and `main`).
  - Always create/switch to a feature branch that follows the above rule before committing.

16) Changed File Documentation Pass
  - When a file is touched, update documentation quality across the full changed file, not only in modified lines.

17) Keep This File Current
  - Add new standing style/documentation instructions to this `AGENTS.md` file as they are introduced.

18) dev Base-Class Macros
  - When integrating a `dev::<base-class>` helper into an app, prefer the corresponding interface macros provided by that helper header when they exist.
  - Examples include `TELEMETER_*` for `dev::telemeter` and `FRAMEGRABBER_*` for `dev::frameGrabber`.

19) Application Unit Test Documentation
  - For application unit tests, place Doxygen grouping under `application_unit_test` in `tests/groups.dox`.
  - Reserve `app_unit_test` for tests of the `libMagAOX` app namespace, not end applications.
  - Prefer the structure:
  - `namespace libXWCTest { namespace <appName>Test { ... } }`
  - Add a `\defgroup <appName>_unit_test` block and `\ingroup application_unit_test`.
  - Add a brief Doxygen block for each `TEST_CASE`, not just the file header.

20) Test Doxygen Link Preservation
  - When test harness indirection, fault-injection wrappers, alternate namespaces, or protected/private access would prevent Doxygen from auto-linking the real API under test, include `tests/testXWC.hpp` and add explicit Doxygen-only references to the real symbol inside the relevant `TEST_CASE`.
  - For MagAO-X app unit tests, prefer the local `#ifdef <APP>_TEST_DOXYGEN_REF` pattern with raw member references that need not compile, and wrap those blocks in `// clang-format off` and `// clang-format on`.
  - Hide test-harness-only classes or helpers from Doxygen with `\cond ... \endcond` when they would otherwise dominate the generated links. 
  - Use this for methods/functions actually under test, especially in unit-test files that rely on wrapper namespaces, injected subclasses, or macro-based indirection.

21) App Header-Only Preference
  - For MagAOX applications, prefer header-only implementation when it matches existing app patterns and keeps the app easy to include in unit tests.
  - In the common app pattern, the `.cpp` file should contain only the main entrypoint, while the class declaration and out-of-class inline definitions live in the `.hpp`.
  - If deviating from this pattern for a specific app, preserve the local convention already established in that app or directory.

22) Commit Separation
  - Keep changes separated into clean commits on feature branches.
  - Make functional changes first.
  - Keep relevant plan files up to date as implementation progresses when they capture engineering decisions or execution notes for the work.
  - Commit relevant plan files with the associated code changes when they serve as part of the engineering record for that work.
  - Follow with documentation-only changes.
  - Make formatting-only cleanup a separate final commit when needed.

23) Commit Messages 
  - Keep commit messages brief, normally a single line to describe the commit purpose 
  - Additional detail should go in the assocated planning document (if it exists) and be committed.
  - Also include an attribution "Co-authored by  <model-name>" substituting the model name.  This should be on an additional line

24) PR Prompt Attribution
  - At the top of PR descriptions, include an explicit attribution line when work was performed with an AI agent.
  - Preferred format:
    - `This work was performed by <current agent model name> in response to the prompt: "...".`
    - Substitute the actual model name used for the work instead of hardcoding a specific release name.
  - If prompted to review and execute a planning/task file, list its path. Otherwise include the primary user prompt verbatim (or a faithful condensed version if it is extremely long). 
  - If the primary prompt referred to a planning file the planning file does not need to be summarized.
  - Provide the PR description in a copyable md text block
  - Provide a suggested PR title in a separate copyable text block

25) When you finish:
- Summarize what changed.
- List affected files.
- Note any follow-up items or potential edge cases.
