# autoresearch

This is an experiment to have the LLM do its own research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `prepare.py` — fixed constants, data prep, tokenizer, dataloader, evaluation. Do not modify.
   - `task.py` — the frozen ruler: data root, budget, the validation set, and the
     metric. Do not modify. Its `should_stop` is the one place training may end;
     when the budget counts steps, it should also call `judge.will_overrun`, which
     projects from the pace once it has settled and ends a run that cannot finish
     in time. Without that projection a configuration past the compute ceiling
     burns the whole wall clock before anyone learns it never had a chance, and
     every mechanism near the ceiling pays that fee once.
   - `train.py` — the file you modify. Model architecture, optimizer, training loop.
4. **Verify data exists**: Check that `~/.cache/autoresearch/` contains data shards and a tokenizer. If not, tell the human to run `uv run prepare.py`.
5. **Do not create the tables by hand**: Leave `results.tsv`, `parents.tsv`, `queue.tsv`, and `run.json` uncreated. `uv run judge.py calibrate` writes the baseline rows, the parent list, and the calibration.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You do not launch `train.py` yourself. `judge.py` launches it and records the score.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. Everything is fair game: model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.
- Choose which parent commit to start from, then commit your edit.
- Run `judge.py` commands listed below.
- `git reset` a commit you decided not to keep.

**What you CANNOT do:**
- Modify `prepare.py`, `judge.py`, or `task.py`.
- Edit `results.tsv`, `parents.tsv`, `queue.tsv`, `run.json`, or `ledger.jsonl` by hand. `ledger.jsonl` is append-only: `judge.py trial` writes an `admit` event before training starts and a `result` event after it ends. A run identity (commit + eval_hash + seed + knobs + step budget) that already completed is refused before any compute is spent.
- Install new packages or add dependencies.
- Modify the evaluation harness. `task.score` is the ground truth metric, and `task.load_val` is the only source of a validation set. Take the training loader from `task.load_train` too; never build your own loader over the data directory.
- Change the ruler mid-run. `judge.py` hashes `task.py`, `prepare.py`, and the data directory into one `eval_hash`. If it stops matching `run.json`, every command except `report`, `compare` and `audit` refuses until the human recalibrates, and the night's results split into two incomparable halves.
- Hardcode a random seed. `train.py` must keep the `seed_everything()` call.
- Treat a parent-list update as a validated claim. Do not call a parent-list update a validated claim.

**The goal is simple: get the lowest val_bpb.** The time budget stays 5 minutes. VRAM is a soft constraint. Some increase is acceptable for a meaningful gain, but it should not blow up dramatically.

**Simplicity:** You judge whether added code is worth the gain. `judge.py` does not measure simplicity. A seed-1 `val_bpb` that clears the noise band (`好于父本`) may enter the parent list only if you then run `accept`. A score inside the band (`分不开`) can replace *its own parent* through `accept --simpler --replace <that parent>`, and only when the code is simpler. A score above the band never enters the list.

## The experiment loop

Work on a branch such as `autoresearch/mar5`. The first command on a clean
baseline commit is:

```text
uv run judge.py calibrate
```

It trains five times at seeds 1–5, measures the seed-to-seed standard
deviation, and writes `run.json`. Everything after it — the accept band, the
claim threshold — comes from that number. A trial whose score sits inside the
band is `分不开`: indistinguishable from noise, not an improvement.

LOOP FOREVER:

1. Read `parents.tsv`, `results.tsv`, and `queue.tsv`. When the queue is over
   its limit, `judge.py` refuses new seed-1 trials — go review instead.
2. Check out a parent commit from the list. After enough tries from one
   parent, `judge.py` refuses it and you must pick another.
3. Write down the mechanism in three facts: a short mechanism name, whether you
   expect `val_bpb` to go `lower` or `higher`, and one sentence for what counts
   as failure. The same short name is the same mechanism even if the other two
   sentences change.
4. Edit `train.py` and git commit. The new commit's parent must be the chosen
   parent. List every module-level constant your edit changed; that list is
   `--knobs`. Declaring nothing while the diff changes a constant is refused
   before any training happens.
5. Run:

```text
uv run judge.py trial --seed 1 --mechanism 学习率 --knobs LEARNING_RATE \
    --expect lower --fail 变高或崩溃
```

   The ledger records the admission before the training command starts. If
   the same identity already completed, the command is refused before any
   compute; a failed measurement may be retried.
6. Read the new `results.tsv` row. `judge.py` does not create commits and does
   not reset them. A trial that scores *worse* than its parent is measured once
   and judged. A trial that scores better triggers 追加测量: judge runs the same
   commit at seeds 2 and 3 (`trial_repeats` in `run.json`) and judges the mean of
   all three against a band that is tighter because both arms now have more
   samples. So a winning trial costs about three times the wall clock, and a
   `分不开` that came back from three measurements is a much stronger statement
   than one from a single draw — the detail column tells you which it was.
7. If the decision is `好于父本` and the complexity is worth keeping, leave the
   commit and run `uv run judge.py accept`.
8. If the decision is `分不开` and the code is simpler, leave the commit and run
   `uv run judge.py accept --simpler --replace <parent-hash>`.
9. If the decision is `差于父本`, or the complexity is not worth it, `git reset`
   back to the parent. Do not run `accept`.
10. If the decision is `代码坏了`, fix a typo or missing import and try again. If
    the idea itself cannot run, leave the crash row and move on. Do not describe
    that crash as the mechanism being worse.
11. If the decision is `测量失败`, do not accept the commit. Reset and fix the
    seed printing before counting the idea.
12. When `queue.tsv` is not empty, pick an entry and review it: check out that
    commit and run `uv run judge.py claim`. A winning trial had to beat its
    parent to be accepted, and beating it triggered the extra measurements, so
    an accepted commit already carries seeds 1 through 3 — running
    `trial --seed 2` on it now is refused as a duplicate. Only measure another
    seed if `claim` says it lacks samples. A review also replaces that parent's
    single-draw basis with the mean of every seed it was measured at.

Overnight search uses seed 1 only. The parent list holds at most three commits.

Before calling anything validated, run `uv run judge.py audit`. It is
read-only and reports inconsistencies between the ledger, `results.tsv`,
`parents.tsv` and `queue.tsv` (duplicates, missing eval hashes, unreviewed
parents, multi-knob accepts, orphan queue entries).

`judge.py` rewrites `REPORT.md` after every `calibrate`, `accept`, and `claim`.
Read it instead of keeping your own tally; `uv run judge.py report` recomputes
it on demand.

## Morning claim

Pick one or two commits you might call validated. Both arms usually already
carry the seeds they need — the candidate from the extra measurements its win
triggered, the baseline from `calibrate`. Measure another seed only on an arm
that is short, with a clean tree:

```text
uv run judge.py trial --seed <n> --mechanism <same short name> --expect lower --fail <same failure sentence>
```

Then compare the two arms on their shared seeds:

```text
uv run judge.py compare --base <parent> --treat <candidate> --json
```

`compare` is the paired readout: it requires at least two shared seeds and
the same `eval_hash` on both arms. `significant` with `direction=better` is
the only verdict that reads as an improvement; `inconclusive` means the
materials are insufficient, the effect is inside the noise band, or it is
below `min_improvement`. `claim` applies its own selection penalty on top,
so a passed `compare` is necessary but not sufficient.

Then, on the candidate commit, run:

```text
uv run judge.py claim
```

Both gates size their band from each arm's own noise rather than assuming the
two share one. A mechanism can be noisier than the baseline it came from --
two that are quiet alone have combined into one with more than twice the
spread -- and a band built on the baseline's sigma alone would wave that
through.

`claim` pools every seed each side was measured at, not just the first two.
Its threshold is `(z + sqrt(2 ln N))` standard errors, where N counts the
mechanisms tried: always wider than the `z` standard errors accept used, so
claim is stricter by construction, yet it narrows as either arm is measured
more, so extra seeds buy real sensitivity. `min_improvement` is a separate
gate -- the smallest gain a human cares about, which may be zero. The detail
column records how many measurements each arm contributed.

Only a `claim` row whose decision is `成立` is a validated result. `还判不了` means the materials were insufficient. `进入名单` is not that result.

`judge.py` kills a training run that exceeds 10 minutes and records `代码坏了`.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or gone from a computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read papers referenced in the code, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes you ~5 minutes then you can run approx 12/hour, for a total of about 100 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!
