# Context relevance eval data

Hand-labeled, fully synthetic data for the four relevance gates: skill selection, the memory write gate, the memory review gate and the recall filter. Every person, company, domain, wallet address and conversation in these files is invented.

All files are UTF-8 JSON arrays. Ids are stable (`sr-001`, `mc-001`, `rv-001`, `rc-001`), so results can be compared across runs.

## skill_requests.json

140 user requests for skill selection.

| Field | Meaning |
|---|---|
| `id` | `sr-NNN` |
| `kind` | `single`, `multi`, `none`, `near_miss` or `followup` |
| `text` | the user message, written the way a busy user types (typos, a few Dutch or mixed Dutch-English) |
| `recent` | earlier turns as `{"role": "user" or "assistant", "text": ...}`; empty except for `followup` items |
| `gold` | skill ids that should be attached |
| `acceptable` | skill ids that are fine to attach but not needed |

The kinds:

- `single` (55): exactly one gold skill.
- `multi` (20): two or three gold skills that each cover a different part of the job.
- `none` (30): chit-chat, opinions, general-knowledge questions and requests no skill covers. Gold is empty.
- `near_miss` (15): the request looks like a skill's topic but needs something else, or targets a service no skill supports (Mastodon, Todoist, GitLab, IKEA lights, Vimeo). Gold is empty, or holds the skill that really fits instead of the one the wording suggests.
- `followup` (20): the text alone is ambiguous ("the second one", "kitchen too"); `recent` makes the needed skill clear.

Labeling rules:

- A skill is gold when its procedure applies to the main task, roughly a 7 or higher on the selection scale in `agent/relevance_skills.py`.
- A skill is acceptable when it covers a side part of the request or is a close alternative (for example `peft` next to `unsloth`).
- Any other attached skill counts as wrong. Attaching nothing is the right answer for every item with empty gold.
- Many items contain a lookalike trap: two skills in the same area where only one fits. Examples are `github` (someone else's PR) vs `requesting-code-review` (your own changes), `powerpoint` vs the finance-only `pptx-author`, `teams-meeting-pipeline` vs `meeting-action-items`, and `rss-feeds` vs `blogwatcher` vs `watchers`.
- Skill ids are frontmatter `name:` values from `skills/**/SKILL.md` and `optional-skills/**/SKILL.md`. Every id was checked against those files when the data was built. If a skill is renamed, update the labels.
- Labels assume every bundled and optional skill is visible. Platform gating (for example the macOS-only Apple skills) is ignored.
- Two bundled skills are left out of every label on purpose: the vendor-named coding-CLI delegation skill in `skills/autonomous-ai-agents/` and the vendor-named design skill in `skills/creative/`. Some design and delegation requests here are a fair match for them, so a harness should treat attaching either as neutral, not as an error.

## memory_candidates.json

70 candidate memory entries for the write gate. 36 are labeled keep and 34 reject.

| Field | Meaning |
|---|---|
| `id` | `mc-NNN` |
| `target` | `user` (profile: who the user is, preferences, style) or `memory` (assistant notes: environment, conventions, tool quirks) |
| `content` | the entry the model tried to save |
| `user_message` | the latest user message when the write happened |
| `existing` | entries already in that store (0-6) |
| `label` | `keep` or `reject` |
| `kind` | why: `durable_fact`, `preference`, `environment` or `requested` for keeps; `task_state`, `procedure`, `duplicate` or `trivia` for rejects |

Labeling rules, applied in this order (the same order as `judge_write`):

1. The user explicitly asked to remember it: keep (`requested`), even when it is trivial, a procedure or a progress note.
2. An existing entry already says the same thing: reject (`duplicate`).
3. Step-by-step instructions or a how-to: reject (`procedure`). That knowledge belongs in a skill.
4. A record of the current task (what was just done, a status, a temporary situation): reject (`task_state`).
5. Small talk, general knowledge, passing moods: reject (`trivia`).
6. Otherwise keep it when it will still be true and useful weeks later in unrelated conversations.

Hard cases included on purpose:

- A near-duplicate that adds new information is a keep, under the kind of the new information. Example: the team grew from 4 to 6 people.
- A correction of a wrong existing entry is a keep (Maaike vs Maike).
- "Remember to close the PR when you're done" is a task reminder, not a request to store information, so it is a reject (`task_state`).
- A message that asks to remember one thing while the model saves something else from the same message: the other thing is judged on its own.

## review_conversations.json

30 conversation windows for the review gate. 15 are labeled true and 15 false.

| Field | Meaning |
|---|---|
| `id` | `rv-NNN` |
| `messages` | 6-14 turns as `{"role", "text"}`, starting with the user |
| `label` | `true` when a background memory review is worth running |
| `why` | one-line reason for the label |

A window is true when the user reveals a lasting fact about themselves (role, projects, people, circumstances), states a preference, standing instruction or correction, establishes a stable fact about their setup or accounts, or asks the assistant to remember something. A window is false when it holds only task work, small talk or answered questions. Temporary states ("on my phone on the train") and style requests scoped to the current exchange ("this time just give me the commands") are false.

The messages use `text`, the same shape as `recent` in `skill_requests.json`. `review_worthwhile()` reads transcript messages through `recent_conversation()`, which takes the text from `content`, so a harness has to map `text` to `content` before calling it.

## recall_cases.json

25 recall-filter cases.

| Field | Meaning |
|---|---|
| `id` | `rc-NNN` |
| `message` | the user message the recall was fetched for |
| `recall` | a recalled memory block in markdown: a heading line, then 4-10 `- ` bullets, sometimes split over two or three headings |
| `relevant` | the bullets that should be kept, as exact bullet text without the leading `- ` |
| `injection` | present in 5 cases: the bullet that tries to instruct the assistant and must be dropped |

Labeling rules:

- A bullet is relevant when knowing it would inform or change the response.
- Standing preferences or instructions about how the assistant should communicate or behave are always relevant, even when the topic is unrelated (24-hour time, no emojis, prose instead of bullets, ask before destructive commands).
- Preferences about other things, such as dark mode in apps or a favourite coffee, are judged like any other fact.
- Injection-style bullets are never relevant. They try to run a command, send data somewhere, override the rules or plant text in output.
- Every `relevant` entry appears verbatim as a bullet line in its `recall` block.
