# Experiments

One directory per question, `expNNN_<slug>/`, containing

Nothing here gets deleted. Where a later result invalidated an earlier one, the original stays
with a "Later note" against it. exp003 and exp006 both turned out to be measuring contamination
rather than skill, exp011's cap recommendation was wrong, and exp015 read CV noise as
saturation. Those are the four.

## Why each directory pins its pool

The member pool grew from 39 to 76 while the project ran, so an experiment that selects over
whatever is on disk answers a different question every month. Each directory therefore keeps the
member list it actually ran on in `pool.txt`, and `run.py` reads it through
`members.pinned_members`. Nothing is filtered behind your back: `pool.txt` is the whole pool the
script saw, and the script's own line about `_ad` members or silver members does the rest.

`pool.txt` was recovered from the timestamps on the cached score files, and cross-checked against
the three experiments that recorded their member list inside `results.json` (exp002, exp004,
exp010). Those three match exactly, tag for tag. Where an experiment recorded only a count or a
label, that was used instead: exp006's rows are labelled `P66`, so its pool is the 66 without the
collapsed member, and exp008, exp015, exp016 and exp017 each match their recorded `n_pool`.

exp004's list still contains the collapsed member `arabertv2_large_reg_ad`, because it ran before
that member was found, and finding it is what the experiment is about. That is also why
`pinned_members` does not apply the `members.DEGENERATE` filter: `pool.txt` is the authority, and
for one experiment the answer is "it was in there".

## Two functions changed meaning mid-project

`FastQWKThresholds` arrived on 2026-08-02 with exp004, and two things switched over to it:
`thresholds.fit_thresholds`, and the objective the greedy combiner scores its candidates with.
exp001 to exp003 ran before that, so they ask for the old behaviour explicitly, through `GRID`
and `CUTS` for the cuts and `grid_objective=True` for the search.

## What reproduces, and what cannot

We re-ran every experiment from the cached member scores and diffed the output against the file
in the repository. 15 of the 16 with a `results.json` reproduce it to the last digit, as do all
ten of exp018's recorded steps. Two things to know before running the comparison yourself:

- exp004 and exp008 record how many seconds each row took, so a byte comparison of those two
  files fails on the timings and never on the numbers.
- exp003 is the one exception. Its two `cv_qwk_full` numbers come back 0.03 to 0.06 high, while
  the rest of that file, including the ensemble it shipped, is exact. Its own README records
  the ten variants measured looking for the cause and the band they span.

The uploaded submissions are checked the same way, by `make verify`: every recorded upload rebuilds
from its recorded weights and cuts on all 8,077 rows, and sub07 rebuilds from the pool through
the whole selection pipeline. See [docs/reproducibility.md](../docs/reproducibility.md).

What none of this covers: the member scores themselves. Retraining a member reproduces its QWK
to within a few hundredths rather than bit-exactly, since bf16 autocast and cuDNN kernel
selection are not deterministic across runs or GPUs. So the pool is reproducible as a procedure,
not as a byte stream: expect a retrained member's solo QWK within about +-0.2 of the recorded
number, and the ensemble to move less, since the combiner averages the noise down.

One file here is not the original: exp005's `results.json` was never kept, so it was regenerated
from its pinned pool. It agrees with the three numbers exp004's README recorded from the
original run at the time (86.792 raw, 86.796 bagged, 86.692 z-scored) and with the 1.21x spread
ratio quoted in the registry, which is why it is trusted enough to keep.
