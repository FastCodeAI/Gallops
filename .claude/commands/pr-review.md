---
description: Review a gallops PR against docs/reviewing.md — runs make check on the PR head in an isolated worktree with its own venv, checks its claims against the real checkpoint and paddock's side of the contract, then posts one review with severity markers.
argument-hint: "[PR number | branch | blank for the current branch]"
allowed-tools: Bash, Read, Grep, Glob, Write, Edit, TodoWrite, Agent
---

Review a gallops pull request the way `docs/reviewing.md` prescribes.

Target: **$ARGUMENTS**

- A number (`42`) → GitHub PR #42 on `FastCodeAI/Gallops`.
- A branch name → that branch against its PR's base if it has a PR, else `main`.
- Blank → the current branch, by the same rule.

## The criteria are not in this file

**Read [docs/reviewing.md](../../docs/reviewing.md) in full before you start.**
It is the single definition of what a gallops review checks: §1–§11, the scale
in §12, and the quick list. It cites [CLAUDE.md](../../CLAUDE.md),
[docs/checkpoints.md](../../docs/checkpoints.md) and
[docs/conventions.md](../../docs/conventions.md) for the rules it does not
repeat. This command deliberately restates none of them, for the same reason
gallops imports `paddock.policy.protocol` instead of copying it: a copy is
correct today and silently wrong the day the definition moves.

What follows is only the *procedure*.

## 1. Set up an isolated checkout, with its own venv

Never review in the user's working tree, nor in one of the sibling worktrees
the stacked branches live in (`git worktree list`).

```bash
gh pr view <N> --json title,body,baseRefName,headRefName,headRefOid,files,reviews
git fetch origin pull/<N>/head:pr-<N>
git worktree add /tmp/claude-pr-<N> pr-<N>      # or your session's scratch directory
cd /tmp/claude-pr-<N>
uv venv -q --python 3.11 .venv && uv pip install -q --python .venv/bin/python -e ".[dev]"
```

The venv is not optional (§1 says why). Run every test, reproduction and script
with the worktree's `.venv/bin/python`.

## 2. Find what the PR actually adds

```bash
git diff origin/<baseRefName>...pr-<N>
gh pr list --json number,headRefName --jq '.[] | "\(.number) \(.headRefName)"' |
  while read n h; do [ "$n" != <N> ] && git merge-base --is-ancestor "origin/$h" pr-<N> 2>/dev/null && echo "contains #$n"; done
```

Review only what this PR adds (§10), and name the commits you left to other PRs.
Read the description and every doc the PR changes before the code, so you check
§9 as you go.

## 3. Run it

```bash
env -u VIRTUAL_ENV -u CONDA_PREFIX -u GALLOPS_CONFIG make check; echo "exit=$?"
gh pr checks <N>
```

- **Check the exit code directly.** A failing `make check` is the first finding,
  and it is 🔴. Report it with the output; do not work around it. A red check in
  CI is a 🔴 too, `results-pr` included. No checks at all means the PR conflicts
  with its base, or neither side has the workflows yet, so your run is its only
  one: say so.
- **If the PR touches torch code,** a green run proves nothing about it (§1).
  Check that the structural guard covers the classes it touches, or run a smoke
  of a few steps under the training interpreter. Add `[train]` to the venv only
  for that, and say which you did.
- `make typecheck` is advisory, and never a finding by itself.

## 4. Verify what you can, argue the rest, and keep them apart

Track which is which as you go. The review has to state it, and it is the one
thing you cannot reconstruct afterwards.

- **The checkpoint the PR describes.** `gallops config`, run in the main
  checkout where `gallops.toml` lives, resolves `{weights}` and `{runs}`. Read
  the sidecar with `gallops show <dir>`, then check every number the description
  quotes against it, and the sidecar against the weights (§3). Hash the dataset
  against `notes.dataset_sha256`.
- **Reproduce it when that is cheap.** CPU minutes that rebuild bit-identical
  weights from the sidecar alone are the strongest provenance claim a review can
  make.
- **For every check the PR cites,** find its reference, and ask whether that
  reference is independent of the code under test (§4).
- **Paddock's side.** Read it with `git -C <paddock> show <ref>:<path>`, at the
  ref that will merge, without checking anything out in the user's paddock tree.
  Find out whether that ref is paddock's `main` or an open paddock PR
  (`gh pr list --repo FastCodeAI/Paddock --head <branch>`). Load the PR's real
  output with paddock's real reader, from a scratch script.
- **Reproduce every lead,** on synthetic data at the PR head, before it reaches
  the review. A finding from a subagent or from `/code-review` is a lead, not a
  result.
- **Follow §11** for anything you run: no GPU, capped threads, a temporary
  weights root.

For a large diff, consider parallel subagents for independent dimensions
(contracts, sidecar truth, write atomicity, check independence, docs agreement).
Then verify each of their findings yourself.

## 5. Write the review

Follow §12 exactly:
- the verdict and the legend line first;
- then what is specifically right about the PR;
- then the ✅ table;
- then the findings, grouped 🔴 → 🟠 → 🟡, numbered, each anchored to
  `file:line` **in the head commit you reviewed**, each with a failure scenario
  and a fix, and each with the strength of its claim marked.

Add a merge note if the PR sits in a stack (§10).

## 6. Post it

Print the review in the terminal **and post it to the PR**, as **one review
that carries both the summary and the inline comments**:

```bash
gh pr view <N> --json headRefOid,reviews     # unchanged head? a review you already left?
jq -n --rawfile body /tmp/review.md --argjson comments "$(cat /tmp/inline.json)" \
  --arg sha "<headRefOid you reviewed>" \
  '{commit_id:$sha, event:"COMMENT", body:$body, comments:$comments}' > /tmp/review-payload.json
gh api repos/FastCodeAI/Gallops/pulls/<N>/reviews --input /tmp/review-payload.json
```

- **`event` is always `COMMENT`.** A generated review does not gate a merge,
  and GitHub refuses `APPROVE` or `REQUEST_CHANGES` on your own PR anyway.
- **Pass `commit_id`.** It pins the inline comments to the commit you read, even
  if the author pushes while you write.
- **An inline comment only attaches to a line in the PR's diff against its
  base.** Any other line returns 422 — a line from a PR lower in the stack, or
  an unchanged line of a doc the finding contradicts. Quote those as
  `file:line` in the summary instead.
- **A new pass is fine; a duplicate is noise.** Check the PR's existing reviews
  first.

If the target is a local branch with no PR, there is nowhere to post: print the
review, say so, and stop.

## 7. Clean up

```bash
git worktree remove /tmp/claude-pr-<N> --force
git branch -D pr-<N>
```

Remove any temporary weights roots and synthetic datasets you wrote. Leave the
user's branch, working tree and sibling worktrees exactly as you found them.
Kill only processes you started, by process group.
