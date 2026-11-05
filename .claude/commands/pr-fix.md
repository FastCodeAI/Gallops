---
description: Address a review of a gallops PR — verify each finding is real, fix the ones that are on the PR's head branch, prove each fix with a test, then push and reply on every thread.
argument-hint: "[PR number | blank for the review in this session | path to a review file]"
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, TodoWrite, Agent
---

Address a review of a gallops pull request. This is the other half of
`/pr-review`, borrowed from paddock's `/pr-fix` and adapted to a repo that
trains models instead of scoring them.

Target: **$ARGUMENTS**

- A number (`42`) → the review(s) on PR #42 on `FastCodeAI/Gallops`.
- Blank → the review produced earlier in this session by `/pr-review`.
- A path → read the review from that file.

## The criteria are not in this file

`/pr-review` writes findings against [docs/reviewing.md](../../docs/reviewing.md).
**§12** there has the 🔴/🟠/🟡 scale (paddock's, unchanged) and its rules of
thumb for a trainer. Those rules decide which findings you are obliged to fix.
Read §12 rather than trusting this summary to stay current: 🔴 is fixed before
merge. 🟠 is fixed in this PR or tracked as a follow-up before merge. 🟡 is the
author's call, and is never worth a bigger diff to re-review.

The sections a fix most often has to satisfy:

- §2: contracts imported, not restated
- §3: a checkpoint is whole and true, which means a staged write, one fit per
  run, and a sidecar checked against its weights
- §4: a check is only as independent as its reference
- §9: docs and the PR description agree with the diff
- §10: stacked PRs
- §11: no GPU, and never write into the real `{weights}`

[CLAUDE.md](../../CLAUDE.md), [docs/checkpoints.md](../../docs/checkpoints.md)
and [docs/conventions.md](../../docs/conventions.md) define the contracts.
Everything below is procedure.

## 1. Get the review, all of it

```bash
gh pr view <N> --json reviews --jq '.reviews[] | "[\(.state)] \(.author.login) @\(.commit.oid[0:7])\n\(.body)\n"'
gh api repos/FastCodeAI/Gallops/pulls/<N>/comments --paginate \
  --jq '.[] | "\(.path):\(.line // .original_line) outdated=\(.line == null) id=\(.id) reply_to=\(.in_reply_to_id)\n\(.body)\n"'
gh pr view <N> --json comments --jq '.comments[] | "\(.author.login): \(.body)"'
```

A review can live in three places: summary reviews, inline comments and plain
issue comments. One that sits in only one of them is easy to miss.

- **Read the passes in order.** A later pass can change the verdict, and the
  newest one is the one you owe an answer to.
- **`line: null` means the comment is outdated.** Re-locate the code by what the
  finding describes, not by its number, and check whether it is already fixed.
- **The review names the commit it read.** If the head has moved since, say so,
  and check each finding against the current head.

### Where to do the work

If the PR's branch is the one checked out in the user's tree, fix in place.
Check `git status` first and say what was already dirty. Otherwise use a
worktree, never a `checkout` that disturbs the user's tree or one of the
sibling worktrees the stacked branches live in (`git worktree list`):

```bash
git fetch origin <headRefName>
git worktree add -b prfix-<N> <scratch>/prfix-<N> origin/<headRefName>
git -C <scratch>/prfix-<N> branch --set-upstream-to=origin/<headRefName>
cd <scratch>/prfix-<N>
uv venv -q --python 3.11 .venv && uv pip install -q --python .venv/bin/python -e ".[dev]"
```

Fixes have to land on the PR's head branch to be pushable. Work on that ref,
not on a branch cut from `main`.

**The venv is not optional,** for the reason `/pr-review` gives: the main
checkout's `.venv` installs gallops editable from the *main checkout's* `src/`,
so with it active, your fix's tests run against whichever branch the main
checkout has out, and can pass. Run every reproduction and test with the
worktree's `.venv/bin/python`.

## 2. Verify the finding before you change anything

**This is the step that makes the command worth running.** A review is a claim,
including one you wrote yourself an hour ago. The findings marked *argued*
instead of *reproduced* are the ones that turn out to be wrong.

For each finding, reproduce the failure it describes at the PR head before
touching the code: the degenerate input, the missing file, the mixed run.
Use a scratch script, on synthetic data, outside the worktree.

- **Reproduces** → fix it.
- **Does not reproduce** → do not fix it. Record why, with the evidence, and say
  so in the reply. A fix for a bug that is not there is a diff someone has to
  review for nothing.
- **Cannot tell** → treat it as open, not done, and say what you tried.

Rules for a reproduction:

- **Never write into the real `{weights}`.** Paddock reads it. Every run passes
  a temporary weights root, or a `SiteConfig` built on one.
- **No GPU.** There is one, and it is shared and saturated (docs/STRATEGY.md).
  Check `uptime`, cap BLAS threads (`OMP_NUM_THREADS=4`), `nice` it, and ask
  before anything longer than a few minutes.
- **A finding about paddock's side** (the format string, the probe, a sidecar
  key paddock reads) is checked against paddock *at the ref that will merge*:
  `git -C <paddock> show <ref>:<path>`. Do not check out a branch in the
  user's paddock tree. Find out whether that ref is paddock's `main` or an open
  paddock PR.

Work through findings **one at a time, serially**. Subagents are fine for
*reading*, such as checking a key against paddock's reader. Never let two of
them edit one worktree.

## 3. Fix it the way this repo allows

Each of these has an easy fix that this repo forbids. `make check` goes green
either way, so check them by hand.

- **A portability failure is fixed with a template variable** (`{weights}`,
  `{runs}`, `{data.NAME}`, `{site}`), essentially never with an exemption in
  [tests/test_portability.py](../../tests/test_portability.py). The gate also
  scans untracked files, this one included.
- **No `per-file-ignores`.** `pyproject.toml` starts with none on purpose. A
  lint finding is fixed in the code.
- **A restated contract is fixed by importing it,** not by making the copy
  match. The checkpoint layout comes from `SiteConfig.checkpoint_dir`, the repo
  root from `config.REPO_ROOT`, and trajectory conventions from
  `paddock.policy.protocol`. A string that has to be restated across the two
  repos, such as `rerank.FORMAT`, is only acceptable with a check that runs on
  every load (the probe pattern).
- **The sidecar is a contract with a reader you cannot see.** Before renaming
  or removing a `checkpoint.json` field or `notes` key, find out whether paddock
  reads it (`git -C <paddock> grep -n <key> <ref>`). Paddock's re-ranker report
  reads `notes.holdout_scenes`, for example. Changing what an existing field
  *means* needs a `SIDECAR_VERSION` bump (docs/checkpoints.md).
- **Write a checkpoint with `checkpoint.staged()`** whenever a fix touches how
  one is written. Never reopen a step directory and write into it. A reader
  must see a whole checkpoint or none.
- **Never hand-edit a `checkpoint.json`** to make it agree with its weights or
  with the text. Regenerate the checkpoint, so its provenance stays true.
- **A fix to a trainer does not fix the checkpoint it already shipped.** If the
  finding is that a shipped sidecar is false (a precision, a `fraction`, a
  missing scene list), say so in the reply. Regenerating it writes into the
  real `{weights}`, which paddock reads and which a paddock PR's numbers may
  cite, so that is the user's call. Ask; do not do it as part of a fix.
- **Paddock's half lands on a paddock PR.** If a finding can only be closed on
  paddock's side, say so in the reply and leave paddock alone. Neither repo
  imports the other, and neither one's fix reaches into the other.
- **Do not make `make typecheck` blocking** to close a mypy finding.
- **Never fix a failing test by weakening or skipping it.** If the test is wrong,
  argue that as its own finding.

Fix what the review found and nothing else. A drive-by refactor makes the
re-review expensive and hides the fix inside it. When a finding says a doc
contradicts the code, fix every doc that makes the same claim (STRATEGY.md and
data.md often repeat each other), not only the line it was anchored to.

## 4. Prove each fix

A correctness finding (🔴, and most 🟠) is not addressed until a test fails
without the fix and passes with it. That test runs **with no GPU, no dataset and
no `gallops.toml`**, on synthetic data in `tmp_path`, because that is the
environment `make check` promises.

Then, from the worktree, with nothing from the user's shell leaking in:

```bash
env -u VIRTUAL_ENV -u CONDA_PREFIX -u GALLOPS_CONFIG make check; echo "exit=$?"
```

**Read the exit code directly.** `make check | tail` reports *tail's* status.

**Read CI after you push** (`gh pr checks <N>`). If it shows no checks, the PR
conflicts with its base, or neither side has the workflows yet, and your run is
the only one the fix has had, so say so in the reply.

## 5. Report, commit, and push

Print a table mapping every finding to its outcome:

- **fixed**, with the commit and `file:line`
- **not reproduced**, with what you ran
- **deferred**, with the issue
- **declined**, with the reason

**Every finding gets a row,** including 🟡 ones you skipped. A finding that
quietly vanishes is the one the reviewer raises again next pass.

Commit one finding per commit, or one group that genuinely belongs together.
Name what it fixes in the message, so a reviewer can map the review to the
diff without guessing.

**Five conditions gate the push. If one fails, stop and say so; do not work
around it.**

- **`make check` exited 0.** Never push red.
- **The branch is not `main`.** Fixes go on the PR's head branch.
- **The push fast-forwards.** Never `--force` or `--force-with-lease` on a
  branch someone may have pulled. If the push is rejected, fetch, rebase *your*
  commits onto the new head and push again.
- **You can write to it:**
  `gh pr view <N> --json isCrossRepository,maintainerCanModify`. A fork that
  refuses maintainer edits cannot be pushed to. Leave the commits local, say
  so, and offer the patch.
- **The stack is accounted for.** Gallops PRs stack. A fix on a PR's branch does
  *not* reach the PRs above it, and they will conflict if they touch the same
  files. List them before pushing:

  ```bash
  gh pr list --json number,headRefName,baseRefName --jq '.[] | "\(.number) \(.headRefName) <- \(.baseRefName)"'
  git diff --name-only <fix-base>..HEAD > /tmp/fixed-files   # or your scratch dir
  for h in <heads above>; do git log --oneline --name-only <fix-base>..origin/$h -- $(cat /tmp/fixed-files) | head; done
  ```

  Report which PRs above will conflict, and in which files. **Do not rebase,
  merge into or force-push the PRs above.** That rewrites branches with their
  own reviews and worktrees. It is the user's call, and usually happens after
  this PR merges (with a merge commit, never a squash).

Then close the loop on each thread, so the reviewer does not have to diff the
branch to find out what happened:

```bash
gh api repos/FastCodeAI/Gallops/pulls/<N>/comments/<comment-id>/replies \
  -f body='Fixed in <sha>: <what changed>. <how it was proven>.'
```

Reply on the findings you did **not** fix too. "Did not reproduce, here is what
I ran" is the reply that stops the next pass from raising it again. Findings
that exist only in the summary have no thread, so answer them in one PR
comment that carries the whole table. Keep the strength of each claim marked
(*reproduced*, *verified by reading*, *argued*), as the review did.

If the description quotes something a fix made false, such as a renamed
metric or a changed sidecar key, update that line of the PR body and say so.
Leave the rest of the body alone.

## 6. Clean up

Unlike `/pr-review`, **the branch here carries the fixes.** Check before removing
anything:

```bash
git -C <scratch>/prfix-<N> log --oneline @{u}.. 2>/dev/null   # unpushed commits?
git worktree remove <scratch>/prfix-<N> --force               # only if that was empty
git branch -D prfix-<N>                                       # only if pushed
```

Never `git branch -D` a branch that holds commits that exist nowhere else.
Remove the temporary weights roots and synthetic datasets you wrote. Leave the
user's branch, working tree and sibling worktrees exactly as you found them.
Kill only processes you started, by process group, and never `pkill` by name
on this box.
