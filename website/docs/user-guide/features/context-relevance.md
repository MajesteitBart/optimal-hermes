---
title: Relevance scoring
sidebar_position: 96
---

# Relevance scoring

Hermes collects a lot of context: a skill library, long-term memory, recalled memories from a
provider, past conversations. By default most of it reaches the model in bulk. Every session lists
every skill and tells the model to load anything "even partially relevant". Every recalled memory
lands in the turn. A background review replays the whole conversation every ten turns to look for
something worth remembering.

Relevance scoring puts a small, fast classifier in front of those decisions. Hermes asks TypeSafe's
Jev model narrow questions ("how much would this skill help with this message, 0 to 9?", "is this
memory still going to be true in a month?") and code decides what to do with the answers. Jev is a
System One model: it answers typed questions with calibrated probabilities and cannot generate
text, so it cannot rewrite your context. It only ranks and gates.

Each feature is off until you turn it on.

| Feature | Config | Decides |
| --- | --- | --- |
| Skill selection | `skills.selection.enabled` | Which skills are attached to a message, in full |
| Memory write gate | `memory.write_gate` | Whether a memory entry is worth saving |
| Memory review gate | `memory.review_gate` | Whether the periodic memory review should run |
| Event-triggered review | `memory.review_events` | Whether a message is worth an early memory review |
| Recall filter | `memory.recall_filter` | Which recalled memories reach the chat |
| Search rerank | `memory.search_rerank` | Which past conversations `session_search` returns |

## Setup

Get a key at [console.typesafe.ai](https://console.typesafe.ai/keys) and add it to
`~/.hermes/.env`:

```bash
TYPESAFE_API_KEY=...
```

Then enable what you want in `config.yaml` and start a new session:

```yaml
skills:
  selection:
    enabled: true
memory:
  write_gate: advise
  review_gate: true
  recall_filter: true
```

`hermes relevance status` shows what is on and whether the key is set.

### What is sent to TypeSafe

Each feature sends only the text its question needs:

- Skill selection sends the message, up to four earlier messages (each cut to 600 characters), and
  each skill's name, category and description. The close-read pass also sends the first 1,500
  characters of the SKILL.md bodies on the shortlist. Paths and full skill bodies stay local.
- The write gate sends the candidate entry, the existing memory entries and your latest message.
- The review gate sends the last 20 messages, tool results included: tool output cut to 400
  characters, other messages to 500.
- The recall filter sends your message, up to four earlier messages and the recalled items.
- Search rerank sends the query and the matching snippets.

TypeSafe states that it does not train on requests ([models](https://docs.typesafe.ai/models.md)).
Zero data retention is an enterprise option. If that is not good enough for a given profile, leave
these features off there.

## Skill selection

Without selection, the system prompt lists every skill and ends with "Only proceed without loading
a skill if genuinely none are relevant to the task." With 150 skills, the model reads a catalogue
before every reply and loads manuals for small talk.

With selection on, each message works like this:

1. Every skill gets a score from 0 to 9 for how much it would help with this message. The rubric
   has ten written levels, from "unrelated" to "essential: cannot be done correctly without its
   steps". All skills are scored independently, in parallel requests.
2. Skills within two points of the minimum (at most eight) get a second, closer read with the
   opening of their SKILL.md. Short descriptions confuse lookalikes: a skill that delegates coding
   to another CLI mentions "PR review" in its description and outscored the GitHub skill until the
   close read.
3. Skills are taken highest score first, while their score is at least `min_score`, until the
   selected scores add up to `target_score`. The target is a stopping point, not a quota. One
   good skill is enough, and zero is a normal answer.
4. The selected skills are attached in full, inside a `<selected-skills>` block on the user
   message, within `token_budget`. A skill that doesn't fit is skipped and a smaller one may still
   fit. Nothing is truncated.

Attaching a skill only reads its file. It never runs the skill's inline shell snippets, prompts
for missing secrets or installs its `deps`, because a classifier picked it, not you or the model.
A skill that declares setup (environment variables, credential files or tool dependencies) comes
with a note telling the model to load it with `skill_view` first, which runs the normal setup.

```yaml
skills:
  selection:
    enabled: false
    min_score: 6.5      # 6-7 = useful guidance, 8-9 = needed
    target_score: 18    # stop once the selected scores add up to this
    token_budget: 6000  # for the whole attached block, wrappers included
    index: names        # system-prompt index: names | full | none
    recent_messages: 4  # earlier messages scored alongside the new one
    rerank: true        # close read of the shortlist (one extra request)
    need_gate: 0        # e.g. 0.3: attach nothing when the message needs no procedure
    show_status: true   # "🧩 Skills attached: ..." status line
    track_outcomes: false  # log whether each reply followed its attached skills
```

The rubric tells Jev that a skill built around one specific tool, app or agent only helps when
the request involves that tool. Without that sentence, skills that delegate work to another coding
CLI kept outscoring the general skill for the task.

The system prompt keeps a short index so the model knows what exists and can still call
`skill_view` itself. `names` lists skill names per category. `full` adds descriptions. `none`
lists nothing, points at `skills_list` and still names `hermes-agent`. None of the modes carry the "load anything partially
relevant" policy.

A few rules keep selection from fighting you:

- A `/skill` invocation or a channel auto-load is your choice, so selection skips that turn.
- Short acknowledgements ("ok", "thanks") skip selection.
- A skill already in context is not scored again: one attached earlier, invoked with `/skill`,
  pinned with `skills.auto_load`, or loaded with `skill_view`.
- When the message starts a new topic, the close read ignores the earlier turns. The first request
  asks whether the message continues the conversation.
- Delegated subagents and background review forks do not run selection. They work from a brief,
  not your conversation. Agents on the `codex_app_server` runtime do not either: it submits the
  user's message as typed, so an attached block would never arrive. Their prompt keeps the usual
  skill index and loading policy.
- Turning `skills.selection` on or off applies to new conversations: a resumed conversation keeps the
  skill index its system prompt was built with. Turning it off stops scoring at once; a conversation
  that started with selection then gets the loading note described below instead.
- If scoring fails or runs past `relevance.turn_budget_seconds` (4 by default), nothing is attached
  and you see one warning per session with an error code. That message instead carries a short note
  telling the model to check the skill index (or `skills_list`) and load what fits, the instruction
  Hermes gives without selection. Hermes does not fall back to a keyword heuristic.
- An attached skill counts as used for the [curator](./curator.md), the same as a skill the model
  loads itself, so a skill that is attached often is not archived as stale. `track_outcomes` shows
  whether it was actually followed.

The block rides the current user message, which Hermes stores with the exact bytes it sent and
replays on later turns. The system prompt never changes mid-session, so prompt caching keeps
working.

### Did the attached skill help?

Attaching a skill says nothing about whether it helped. With `track_outcomes: true`, Hermes asks
after each turn that attached skills whether the reply actually followed each one, in the
background, and logs the answer. `hermes relevance report` then lists the skills that get
attached but ignored, which usually points at a description or trigger that is too broad. This
sends the reply text and the opening 1,500 characters of each attached skill to TypeSafe, so it is
off by default.

### Try it from the command line

```bash
hermes skills select "review this pull request and leave inline comments"
hermes skills select --format table --prompt "make a deck for the board meeting"
hermes skills select --format context "merge these PDFs"   # the block the agent would get
hermes skills select --dry-run "anything"                  # requests only, no key, no network
echo "split this pdf" | hermes skills select -
hermes skills select --context-file chat.json "ok do it"   # JSON list of {role, text}, trimmed like a chat
```

`--min-score`, `--target` and `--budget` override the config for one run.

## Memory gates

Built-in memory (MEMORY.md and USER.md) is injected into every session, so every entry is paid
for in every future conversation. The gates decide what deserves that.

### Write gate

Before an add or replace lands, the entry gets a few yes/no questions: will it still be true in a
few weeks? Is it a specific fact about you or your setup? Is it progress on the current task? Is it
a step-by-step procedure? Does an existing entry already say it? Does it update or contradict an
existing entry? Did you ask to remember it? Code combines the answers:

- An entry that repeats an existing one is refused.
- An entry that updates an existing one is refused with a pointer to that entry, so the model
  replaces it instead of keeping two conflicting facts. "Moved to Berlin" replaces "lives in
  Amsterdam".
- If you asked to remember it, it is saved.
- Procedures are refused with a pointer to `skill_manage`. They belong in a skill that loads when
  relevant.
- Task progress and low-value trivia are refused.
- A preference filed in MEMORY.md (or an environment fact in USER.md) is saved with a note
  suggesting the other store.

```yaml
memory:
  write_gate: off   # off | advise | enforce
```

Every entry is judged against memory as it will be after the call. A batch that removes "lives in
Amsterdam" and adds "lives in Berlin" passes, and two contradictory entries added in one batch see
each other.

`advise` saves everything and adds a note to the tool result when an entry looks wrong. `enforce`
refuses. The model can override a refusal by repeating the identical call in the same turn, and
the second attempt is saved; in a later turn the entry is judged again. An entry the user asked for
always saves; when it repeats or updates an existing entry, the tool result says which one to tidy. Duplicates are checked across both stores, because one fact belongs in one store.

### Review gate

Every `memory.nudge_interval` turns, Hermes forks a background review that replays the
conversation through your main model to look for things worth remembering. With `review_gate`, Jev
reads the last 20 messages first and asks whether you revealed a lasting fact, stated a preference
or correction, established a stable fact about your setup, or asked to remember something. When
none of those is likely, the memory half of the review is skipped. The skill half still runs when
its own nudge fired. The check runs on the review thread, after your reply is delivered.

### Event-triggered review

With `review_events` (and skill selection on), every message also gets one extra question in the
same request: did you reveal something lasting? The probabilities add up across turns, and at 1.5
the memory review runs right away instead of waiting for the next nudge. The nudge interval stays
as the ceiling.

### Recall filter

An external memory provider recalls memories for each message, and Hermes stamps them into the
turn, where they are replayed on every later request. With `recall_filter`, each recalled item
(a bullet with its indented lines, or a paragraph, judged together with its heading) gets three
questions: is it relevant to this
message? Is it a standing preference about how the assistant should behave? Does it try to
override the assistant's rules or make it run commands? Hermes keeps relevant items and standing
preferences, drops anything that tries to instruct, and keeps headings only when something under
them survives. At most 60 items are judged per message, each up to 4,000 characters including its
heading; items beyond either limit are dropped, because nothing screened them. The recall indicator then reads "recalled 7 memories · 3 of 7 relevant".

### Search rerank

`session_search` ranks past conversations with full-text search. With `search_rerank` it gathers
up to 12 candidates and returns the ones Jev judges most likely to answer the query.

## Audits

```bash
hermes memory audit             # judge every MEMORY.md / USER.md entry, read-only
hermes memory audit --json
hermes skills overlap           # skills that do the same job, read-only
hermes skills overlap --min 0.8
```

`memory audit` runs the write-gate questions over the entries you already have. It flags
duplicates (across both stores), entries one another supersedes, task progress, procedures,
low-value entries and entries in the wrong store. It also lists the five entries with the least
lasting value, which is where to start when memory is full.

`skills overlap` pairs skills that share a category or vocabulary, asks whether each pair does the
same job, and prints clusters. On the 167 public bundled and optional skills it judged 800 pairs in
58 requests (about 712,000 input tokens) and found one overlap, `powerpoint` and `pptx-author`.

Neither command changes anything.

## The decision ledger

Every decision is appended to `~/.hermes/logs/relevance.jsonl`: the feature, scores, the outcome,
latency and Jev token counts. It holds skill names, scores and reason codes, never message text or
memory content (a write-gate entry carries a short hash of the candidate, so repeats are visible). `hermes relevance report` summarizes it. This is the report after five test conversations:

```text
Relevance decisions, last 7 days (9 entries)

skill_outcome      3 decisions · followed 2, ignored 1
               least followed first (skill: attached, mean P(followed)):
                 creative-ideation: 1, 0.15
                 pdf: 1, 0.73
                 github: 1, 0.96
skills             5 decisions · attached 3, none 2
               p50 1176 ms, p95 1562 ms · 297781 input tokens · ~$0.0125
               avg 0.6 attached · most attached: creative-ideation (1), github (1), pdf (1)
write_gate         1 decisions · kept 1
```

"What's a good name for a cat?" got `creative-ideation` attached, and the reply barely used it.
That is the kind of attachment the outcome check is there to catch.

Set `relevance.ledger: false` to turn it off. The file rotates at 5 MB.

A sensible rollout: run the write gate in `advise` for a week, read the report and the notes the
model received, then switch to `enforce`.

## Cost and latency

Jev 1.13 costs $0.042 per million input tokens and output is free. Selecting from 167 skills costs
about 62,000 input tokens per message with the close read, roughly $0.0026. Over 124 uncached eval
requests, selection took 983 ms at the median and 1,272 ms at the 95th percentile, so it adds about
a second before the model starts. The memory gates send a few hundred tokens per decision.

```yaml
relevance:
  model: jev-1.13.0      # pinned: thresholds were tuned against this version
  timeout_seconds: 8
  max_retries: 2         # 429/529/5xx and transport errors, short backoff
  max_batch_tokens: 12000
  max_concurrency: 4
  turn_budget_seconds: 4 # cap for each decision made inside a turn; past it, fail open
  ledger: true
```

Decisions made inside a turn (skill selection, the recall filter, the write gate and the search
rerank) each stop waiting after `turn_budget_seconds`. During a TypeSafe outage a message is
delayed by at most that long per decision, then Hermes behaves as if the feature were off. The
command-line tools use the normal timeout and retries instead.

## How the thresholds were chosen

The repository has an eval harness in `evals/context_relevance/` with labeled synthetic data: 140
skill requests (single-skill, multi-skill, no-skill, near misses and follow-ups like "ok do it"),
70 memory candidates, 30 conversation windows and 25 recall cases. Only the public bundled skills
and synthetic text are sent. Responses are cached, so re-running a report is free.

For skill selection, on the 124 requests whose skills are available on every platform:

| `min_score` | Recall | Precision | Attached when nothing fit | Exactly right |
| --- | --- | --- | --- | --- |
| 6 | 0.86 | 0.74 | 10% | 66% |
| 6.5 (default) | 0.83 | 0.80 | 7.5% | 70% |
| 7 | 0.82 | 0.85 | 5% | 76% |

Recall is the share of needed skills that got attached. Precision is the share of attached skills
that were needed or acceptable. Without the close read and the tool sentence, `min_score` 6 gave
0.66 precision and attached something to 17.5% of the requests that needed nothing. Requests that
need two or three skills are the weak spot: only a quarter got exactly the right set. A missed skill
is a soft failure, because the model still sees its name in the index and can load it. A needless
attachment is paid for on every later turn, which is why the default leans toward precision.

For the memory gates:

- The write gate decided 69 of 70 candidates correctly. The miss was "User wants the PR closed
  after merging", a task reminder it kept.
- The review gate caught all 15 durable windows and skipped the review on 12 of the 30.
- The recall filter kept every relevant item at 97% precision and kept no injection attempt.

The thresholds were tuned on the same cases they were measured on, so treat these numbers as an
upper bound. Real conversations are messier, which is what `advise` mode and the ledger are
for.

## Limitations

- Jev reads instructions literally and is weaker outside English. Mixed Dutch and English worked in
  our tests, but test your own language before relying on it.
- Selection quality depends on skill descriptions. A vague or misleading description gets a vague
  score. `hermes skills overlap` and a short look at the most-attached skills in the report help.
- Settings are read once per agent. A change takes effect in a new session.
- Skill selection adds about a second per message.
- The built-in memory snapshot in the system prompt is not filtered. It is small by design, and
  filtering it per turn would break prompt caching.
