# The Exam and the Driving Test

### A mini-book on how to judge a driving model, what we built, and what went wrong along the way

Project: `closed-loop-eval-lab`.

This book is written to be read after the project, slowly, with no background assumed. Every number comes
from `reports/pilot.md`, which is generated from the real run files.

---

## Index

**Part 1: The idea**

1. [The big question](#1-the-big-question)
2. [Four words you need](#2-four-words-you-need)
3. [Why a great score can drive badly](#3-why-a-great-score-can-drive-badly)

**Part 2: Building a fair test**

4. [Making the comparison fair](#4-making-the-comparison-fair)
5. [Keeping the exam paper secret](#5-keeping-the-exam-paper-secret)
6. [Metrics that cannot be cheated](#6-metrics-that-cannot-be-cheated)
7. [How sure are we? Uncertainty in plain words](#7-how-sure-are-we-uncertainty-in-plain-words)

**Part 3: What happened**

8. [What we found](#8-what-we-found)
9. [Our mistakes, and how each was caught](#9-our-mistakes-and-how-each-was-caught)
10. [Failure mining: finding the useful test cases](#10-failure-mining-finding-the-useful-test-cases)

**Part 4: Taking it with you**

11. [Six engineering habits worth stealing](#11-six-engineering-habits-worth-stealing)
12. [What this project does not prove](#12-what-this-project-does-not-prove)
13. [Explaining it in your own words](#13-explaining-it-in-your-own-words)
14. [Run it yourself](#14-run-it-yourself)

**Part 5: The harder test**

15. [Why we tested ourselves again](#15-why-we-tested-ourselves-again)
16. [What held up and what did not](#16-what-held-up-and-what-did-not)
17. [Better bugs, and the bug dev could not see](#17-better-bugs-and-the-bug-dev-could-not-see)

**Back of the book**

- [Glossary](#glossary)
- [Concept index](#concept-index)

---

# Part 1: The idea

## 1. The big question

Imagine two student drivers.

Student A scores 99% on the written exam, and panics on a real road. Student B scores 80% on the written
exam, and drives calmly and safely. If you could only read exam scores, you would hire the wrong one.

Self-driving models have the same problem. There is an easy, cheap test that works like a written exam,
and a hard, expensive test that works like a real driving test.

Our whole project asks one question:

> **Does doing well on the cheap test mean doing well on the real one?**

This matters because companies pick models using these tests. If the cheap test misleads you, you ship the
wrong model.

## 2. Four words you need

**Open loop (the written exam).** You show the model a recording of an expert driving, and ask, "Where will
the car go next?" You compare its answer with what really happened. The model's answer changes nothing. The
recording plays on no matter what the model says.

**Closed loop (the driving test).** The model actually drives. What it does changes where the car is, and
that changes what it sees next. A small mistake now becomes the model's own problem a second later.

**ADE, Average Displacement Error.** The average distance, in metres, between where the model said the car
would be and where it really was. Lower is better. It is the exam score.

**FDE, Final Displacement Error.** The same idea, but only at the very last moment of the prediction.

Here is a hand example from our tests. Suppose the car really drives at 10 m/s and the model guesses 11 m/s.
After 0.5 seconds the guess is 0.5 m off, after 1 second it is 1 m off, and so on up to 3 m. The average of
0.5, 1, 1.5, 2, 2.5 and 3 is 1.75, so ADE is 1.75 m, and FDE is 3 m. We worked this out by hand first, then
checked that the code agrees. It does.

## 3. Why a great score can drive badly

The exam only tests situations from an expert's recording, and experts drive neatly. They stay near the middle
of the lane. So the model rarely sees what a car looks like when it is slightly off course.

Now let it drive. It makes a tiny error and ends up slightly off centre. That is a place it barely studied, so
its next decision is worse, which pushes it further off. Errors feed on themselves. This is called
**compounding error**, and the shift into unfamiliar situations is called **distribution shift**.

There is a sneakier trap too. Suppose you let the model see the car's recent past. The easiest way to predict
"where next" is to say "keep doing what you were doing." On a recording of a good driver, that scores very well,
because the good driver was already doing the right thing. When the model drives by itself, it is copying only
itself, and nobody is steering. Researchers call this the **copycat problem**.

Keep this idea in mind as a *suspect*, not a proven fact. In Chapter 8 you will see that our own test did not
prove it.

---

# Part 2: Building a fair test

## 4. Making the comparison fair

A comparison is only meaningful if the things being compared face the same conditions. We made four rules.

**One output format.** Every policy gives the same thing: six waypoints, one every half second, up to 3 seconds
ahead. So the exam and the driving test judge the same output, not two different systems.

**One controller.** A single fixed piece of code, the **tracker**, turns any plan into steering and pedals. It
never knows which policy made the plan. So differences come from the plans.

**The same bad luck.** Each test includes random steering wobble, like gusts of wind. We fix the randomness with
a seed, so every policy gets the identical wobble on the same scenario. This is called **paired** testing. It's
like giving two runners the same wind, so a difference in time means a difference in runners.

**The same first second.** Models using history need one second of past. We found that our first version gave
them an empty past in the driving test, which they had never seen in training. That was unfair, so now the
expert drives the first second for every policy.

### Who took the test

Twelve candidates:

- **Four rule-based drivers:** `rule_center` (careful), `rule_offset` (sits 0.4 m to one side on purpose, which
  costs exam marks but is harmless), `constant_velocity` (never steers), and `stationary` (never moves).
- **Eight learned models:** every combination of three choices, with or without history, trained on clean or
  noisy expert drives, and simple (degree 1) or more flexible (degree 2).

Why all eight combinations? Because if you hand-pick models, you can get any answer you want. Using the full
grid means nobody chose them based on how they scored.

The models are **ridge regression**, a simple statistical method. This is deliberate. The project is about
*testing*, so we wanted models that are fast and easy to inspect.

## 5. Keeping the exam paper secret

Three separate sets of scenarios:

- **Train:** what the models learn from.
- **Dev (development):** what we use for tuning and choosing.
- **Test (held out):** used only once at the end, for the final answer.

If test information gets into training or choosing, this is called **leakage**, and it makes results look better
than they are. It is like seeing the exam paper early.

We split by **scenario family**, not single scenarios. A family is a type of situation, like "tight curves" or
"a car cuts in". If you split randomly, the test would contain near-copies of training scenarios. Holding out the
whole family asks a harder and more honest question: can it handle a situation type it has never seen?

We also made rules into code. The final stage refuses to run if the code has uncommitted changes, and refuses to
overwrite an old result. This means nobody can quietly re-run until they like the answer.

Even with all this care, we had a leak. Chapter 9 tells that story.

## 6. Metrics that cannot be cheated

A metric is a rule, and people, or models, will find ways around a weak rule.

**The parked-car trick.** Our `stationary` policy never moves. Its collisions per episode? Zero. On that number
alone it looks like the safest driver in the world. To stop this, our main score is **safe completion**: you
reach the goal, in time, with no crash, and without leaving the road. The parked car scores 0.

**Who is in the bottom of the fraction?** The bottom number in a rate is the **denominator**, and it hides a lot.
If runs that crashed the software were dropped from the bottom, a buggy model would look better. So every planned
run stays in the denominator, including timeouts and failures.

**Collisions per km can mislead too.** A model that crashes early drives few kilometres, and a parked car drives
none, so "collisions per km" is undefined for it. We report it as "not assessable", never as zero.

**Unassessable is a real answer.** If timestamps are broken or positions are missing, the honest result is "we
cannot judge this", which we call **UNASSESSABLE**. It's never a pass and never a fail. Otherwise, the worst
telemetry could produce the best-looking results.

We also compute crashes from the recorded track with our own geometry, using the **separating axis test**, instead
of simply trusting the simulator. That lets us compare the two, and the comparison turned out to be very useful.

## 7. How sure are we? Uncertainty in plain words

Suppose we run 600 episodes. Do we have 600 independent facts? No.

If a curve is too tight for a policy, it fails that curve on nearly every seed. Five runs of one scenario are
more like one fact said five times. So the honest unit is the **scenario**, not the episode.

To get a range, we use a **bootstrap**: pick scenarios at random, with repeats allowed, and recalculate the score,
thousands of times. The spread of those answers gives a **95% confidence interval (CI)**, a range of plausible
values. Because we resample whole scenarios, this is a **clustered** interval.

How much does this matter? On the fixture, the **design effect** was about 3 to 4, which means seeds are worth far
less than they look. The honest interval came out roughly 1.7 to 1.9 times wider than the naive one for most
policies. A naive analysis would have been falsely confident.

**Zero failures does not mean zero risk.** `rule_center` passed all 72 of its MetaDrive runs. Do we say the failure
rate is 0%? No. With 24 scenarios, the data still allows a true success rate as low as about 85.8%. We use the
**Clopper-Pearson** bound here because a bootstrap would wrongly say "exactly 100%, with no doubt at all".

---

# Part 3: What happened

## 8. What we found

We tested in two worlds:

- **The fixture:** a small hand-written car model. Fast, exact, and a toy. 7,200 episodes.
- **MetaDrive:** a real open-source driving simulator. 864 episodes on two held-out road families.

### Finding 1: choosing by the exam picked a bad model

This is the most important result, and the protocol fixed it before the runs. We chose one model using exam scores
(ADE) and one using driving tests, both on dev data only, then compared them on held-out families:

| World | Chosen by exam | Chosen by driving test | Difference (95% CI) |
|---|---|---|---|
| Fixture | 0.002 | 0.467 | +0.465 (+0.392 to +0.545) |
| MetaDrive | 0.000 | 0.889 | +0.889 (+0.778 to +0.972) |

The numbers are safe completion rates. In both worlds, the model with the best exam score almost never finished
safely. Same pick both times: `ridge_hist_clean_d2`.

### Finding 2: among learned models, the exam order ran backwards

We ranked the eight learned models by exam score and by driving result, and measured how well the orders match,
using **Spearman rank correlation**. If lower error meant better driving, this would be negative.

| World | Correlation (95% CI) |
|---|---|
| Fixture | +0.667 (+0.310 to +0.874) |
| MetaDrive | +0.939 (+0.856 to +0.939) |

Both are positive, so better exam scores went with *worse* driving.

### Finding 3: but the exam itself was tilted, and this is the honest twist

> **Update from Part 5:** the harder test weakened this finding and Finding 2. Read Chapter 16 before quoting either.

A reviewer pointed out a problem. Our exam used calm recordings, while our driving test had wobble. Models trained
on calm drives were being examined on the same kind of data they studied.

So we added a new exam, scored on *wobbly* recordings:

| World | Correlation on wobbly logs (95% CI) | Driving-test pick minus this exam's pick |
|---|---|---|
| Fixture | -0.286 (-0.503 to -0.095) | +0.192 (+0.108 to +0.280) |
| MetaDrive | -0.685 (-0.702 to -0.436) | +0.875 (+0.764 to +0.972) |

Now the correlation is negative, which means the exam agrees with driving. The right lesson is not "ADE is useless."
It is more interesting:

> **An open-loop score depends heavily on which data you score it on. Scoring on data with mistakes in it makes it
> much more honest. Even then, in these two worlds it still picked a worse model than the driving test did.**

This was added after seeing results, so it is exploratory, and we label it that way.

### Finding 4: our favourite explanation was not proven

We expected the copycat story, so we set a check: turn the wobble off. If outside disturbance was the cause, history
models should recover. They did not. `ridge_hist_clean_d1` scored 0.000 with or without wobble, in both worlds.

The check was also weaker than we thought, because the tracker is not the same controller as the expert, so even
without wobble a policy does not replay the recorded drive. So our verdict is **inconclusive**: the failure is real,
and the cause is not established. Writing that down is more useful than telling a neat story.

### One more result worth noticing

In the fixture, `rule_offset` has worse exam marks than `rule_center` (0.78 vs 0.57 m) and yet its safe completion
was not worse (0.952 vs 0.937, with overlapping intervals). A sideways bias costs marks and, here, no safety.

## 9. Our mistakes, and how each was caught

This may be the most valuable chapter, because every one of these almost became a false result.

**The driver who could never finish.** A careful driver kept "timing out" on straight roads. That made no sense.
The run ended one metre before the goal, so nobody could ever reach it. *Lesson: if a good driver fails, suspect your
test first.*

**The ghost car.** Cars ahead reached the end of the road and stopped, while still reporting a speed. Followers trusted
that speed and crashed. We gave the road a run-out section.

**The empty memory.** The very dramatic copycat result appeared, and it looked too clean. We found the unfair empty
history (Chapter 4) and fixed it. The failure survived, which made it more believable. *Lesson: when a result is
exciting, attack it first.*

**The car that "timed out" while leaving the road.** MetaDrive stops as soon as a wheel crosses a line, but our rule
looks at the car's centre, so runs ended before our rule could fire. Cars that left the road were labelled as harmless
timeouts. We caught this because we kept the simulator's flags and compared: 245 disagreements. *Lesson: an independent
cross-check earns its keep.*

**The biggest one: the leak.** After the final runs, an independent reviewer found that our MetaDrive test maps reused
the same random seeds as training maps. MetaDrive builds a map from a seed, so test roads traced the training roads:
59 to 74 percent of a route lay on a single training road. The reviewer and I measured it separately.

We did not quietly fix and overwrite it. We kept the old run, marked it superseded, wrote an amendment explaining why,
committed that *first*, and ran new named runs. The main result held up (+0.861 before, +0.889 after).

A reassuring number is not the point. The point is that we would have published a false claim of "held out".

And it isn't entirely gone. Overlap fell to 38 to 50 percent, against 19 to 25 percent for unrelated maps, because our
test families (SCCS, CSCC) are longer versions of the training families (SCS, CSC). So "held out" in MetaDrive means
"new maps with similar shapes", and we say so.

**The test that could never fail.** Our false-alarm check compared a policy with itself on the same seeds. It could
only ever score zero, so it proved nothing. We switched to a true **A/A test**, on different seeds.

**Tests that could not fail.** We deliberately broke our own code 36 ways to see if the tests noticed. Two breaks
slipped through, and a third never ran because its target line had been reformatted. Passing tests prove little
if the tests cannot fail.

**Small slips I made.** A `sed` command that silently changed nothing. A formatter that reformatted files from the
starter files, which I restored and confirmed against their checksums. Check that an action really did what you
intended.

## 10. Failure mining: finding the useful test cases

> **Update from Part 5:** Chapter 17 repeats this study with fifteen better bugs.

Simulation time is limited, so which scenarios should you run? **Failure mining** means using past failures to choose
the most useful test cases.

We built a miner that looks at dev failures and picks similar yet varied scenarios from a pool, without ever seeing
test outcomes. We then compared it with random choice, using small hand-built bugs called **regressions**.

The result needs careful reading:

| Scenarios tested | Mined: finds the real bug | Random: finds the real bug |
|---|---:|---:|
| 8 | 100% | 65% |
| 16 | 100% | 96% |
| 32 | 100% | 100% |

That looks like a win, but there are three catches:

1. **Only one of our four "bugs" was real.** The others did not make driving worse across the full pool. One even
   helped. So this rests on a single example. I failed to make good test bugs, and we did not change them afterwards.
2. **The setup favoured the miner.** The largest group of dev failures was lead-car crashes, and the one real bug was
   about ignoring the lead car.
3. **It paid a price.** At 8 scenarios, mined packs covered about half the road families, while random covered 92%. A
   pack that misses families is blind to bugs living there.

Honest summary: promising on one narrow example, with a real coverage cost, and not proven. The A/A false-alarm rate was
0.00 in all rows, which is good, though with only three seeds per scenario it is a small check.

---

# Part 4: Taking it with you

## 11. Six engineering habits worth stealing

1. **Write the plan down before the result exists.** This is **pre-registration**. `experiments/protocol.md` was
   committed before the final runs, and git proves the order. It's how you stop fooling yourself.
2. **Keep every failure in the count.** Timeouts, crashes and unassessable runs stay in the denominators.
3. **Make every number traceable.** Each run keeps its code version, config hash, data hash, seeds and machine details
   in a manifest. The report is generated from these files, and regenerating gives a byte-identical file.
4. **Turn promises into code.** "Don't overwrite final results" is a wish. A program that refuses is a guarantee.
5. **Test your tests.** That's **mutation testing**: break the code on purpose and check that the tests fail.
6. **Ask someone who does not share your conclusion.** The reviewer got the code, not my opinion, and found what I
   could not see. When you are wrong, keep the old result visible and say what changed.

## 12. What this project does not prove

Please stay honest about this one whenever you describe the work.

- It says nothing about real vehicles or fleet safety.
- The policies are small toy models, and real planners may behave very differently.
- The fixture is hand-written, and its other cars don't react to you.
- MetaDrive results use one simulator, one lane, light traffic and only two held-out families.
- Confidence intervals cover scenarios *inside* these families. They say nothing about new families.
- Eight learned models is a small group, so correlations could change with a different group.
- On MetaDrive, 6 of 14 collisions (about 43 percent) were seen only by the simulator, so our collision rates undercount.
- Our jerk numbers are not usable as comfort measures.
- The cause of the history models' failure is unknown.

## 13. Explaining it in your own words

**Why can a lower open-loop error give a worse closed-loop result?**
Open loop scores each moment separately, from the expert's states, so a mistake has no consequence. In closed loop,
mistakes change what the model sees next, so they compound. In my project the best-ADE model reached the goal safely
almost never, in two worlds. The cause is not fully proven, and I'd say that.

**What is the denominator for collision rate, and what does it hide?**
Per planned episode, it can reward not moving: a parked car scores zero. Per km, it changes with exposure, and is
undefined at zero km. So I report both, keep every planned run, and use safe completion as the primary measure.

**How would you detect timestamp and frame bugs?**
Refuse to compare unless frames match and timestamps align, and return UNASSESSABLE on gaps or disorder. Test with
hand-calculated cases, including one where numbers match but frame labels differ. Compare trace-derived events with
simulator flags, because disagreement exposes definition bugs. That's how I found the wheel-versus-centre problem.

**How do you prevent failure mining from contaminating the holdout?**
Make it impossible in code: the selection function takes dev rows and scenario definitions, and raises an error if
any row isn't dev. Log lineage for every selected case. Check for hidden leaks too: mine was in map seeds, not the
labels, and only a review plus a geometry measurement caught it.

**What uncertainty remains, and how does it affect a release decision?**
Scenario noise is covered by clustered intervals. Family coverage is not: new situation types are unknown. There's
also the simulator-to-real gap. So a pass means "no evidence of regression on these families within this bound",
not "safe". With 24 scenarios and no failures, the bound still allows about 14% failure, so I'd ask for more scenarios
before signing off.

## 14. Run it yourself

```bash
cd closed-loop-eval-lab
uv sync --extra sim                 # installs everything, including MetaDrive
.venv/bin/python -m pytest tests -q # the checks
.venv/bin/python -m eval_lab.analytic reports/analytic.md   # hand calculations vs code
.venv/bin/python -m eval_lab.report --out reports/pilot.md  # rebuild the report from runs/
```

Fun things to try:

- Open `reports/pilot.md` and find the row for `stationary`. Why are collisions per episode zero?
- Read `tests/test_metrics.py`, starting with `test_a_policy_that_never_moves_cannot_look_superior`.
- Run `scripts/mutation_check.py` on a copy of the repo, and watch tests catch deliberate bugs.
- Try deleting a timeout row from a raw file, and see the report change.

---

# Part 5: The harder test

## 15. Why we tested ourselves again

The pilot left us with three weak spots:

1. Only one of our four practice bugs was real.
2. We had already seen the test roads, so we could not use them again after changing anything.
3. MetaDrive's test maps were too similar to the training maps.

So we built a second, harder test, called M2. Think of a student who passed a mock exam. A fair teacher writes a new
paper with new kinds of questions, and writes the marking rules before anyone sits it.

We did four things before running anything:

- **New roads.** The fixture got eight new road families. Two are easy on purpose, because real driving is mostly easy and
  a good test-picker should not waste its budget there. MetaDrive got roundabouts, junctions and an off-ramp.
- **Roads picked by shape, not by score.** We measured how much each candidate map overlapped with training roads, then
  kept the least similar. We used only geometry and the expert driver, so no candidate model got a preview.
- **Better practice bugs.** We tried 37 bug variants on dev, 17 of which really hurt. Then we froze 15 candidates.
- **Rules first.** We committed the plan to git before the first run. That timestamp is the proof.

---

## 16. What held up and what did not

Good science includes finding out you were partly wrong. Here is the scorecard.

| Pilot claim | Harder test | Verdict |
|---|---|---|
| Picking by the exam (clean ADE) chooses a worse driver | +0.459 on the fixture, +0.156 on MetaDrive | Held up, though smaller on MetaDrive |
| The exam ranks models backwards | +0.143 on the fixture, but -0.708 on MetaDrive, which is the right way round | Did not hold in general |
| A wobbly exam still picks worse | +0.041 on the fixture, unclear. +0.149 on MetaDrive | Mixed |
| The closed-loop pick drives well on held-out maps (0.889) | 0.156, and zero in 6 of 8 families | Did not hold on harder maps |

**The biggest surprise: the exam stopped running backwards.** In the pilot, the models with the best exam scores drove
worst. On the new maps, the exam scored those same models badly too, because a road the model has never seen shows up
in the exam as well. So the exam is not cursed. It can warn you, but only if you sit it on the kind of road you care about.

**Why picking still went wrong.** We chose models using dev roads. Dev could not tell us about roundabouts. The problem
was less "exam versus driving" and more "practice roads versus new roads". That is a sharper lesson than the one we
started with.

**A wobbly exam nearly caught up.** On the fixture, picking by ADE on disturbed logs was about as good as picking by
driving result. So if a closed-loop run is expensive, a better exam may be worth building first.

**The sobering result.** `rule_center`, the simple careful driver, scored 1.000 on MetaDrive. Our best learned model
scored 0.156, and only on two families. Choosing between models does not help much if none of them can do the job.

### Watch out: how many things did we really test?

Our confidence ranges treat each scenario as one piece of evidence. But scenarios in a family are alike, so for a claim
about *new families*, the honest count is 8, not 96.

When we recounted that way, after seeing the results, both MetaDrive ranges reached zero. The result was not wrong. It
just answers a narrower question: these maps, rather than any new kind of map. Next time, we would register that count
from the start.

### Did the copycat idea finally get proven?

No, but we got closer. We reran each model with no outside wobble, using the expert's own controller and starting from
the logged state. The rule driver passed this control, so the test was valid. The clean history models still failed
(0.000 to 0.017 on the fixture and 0.000 on MetaDrive), so they do not need an outside push to fail.

Here is the catch: those roads were new to them, so unfamiliar roads could explain it too. State-only models also drifted
a metre away within seconds. The clean next experiment is to rerun this on training roads.

---

## 17. Better bugs, and the bug dev could not see

This time, we called a bug **real** only if it hurt on dev and also hurt across the whole held-out pool. Six of the
fifteen qualified.

| Scenarios tested | Mined: finds real bugs | Random: finds real bugs | Reading |
|---|---:|---:|---|
| 8 | 59% | 35% | Mined better |
| 16 | 66% | 52% | Mined better |
| 32 | 73% | 64% | Mined better |

Mining passed the test we set. Now the catches, which matter more than the headline:

1. **One bug did most of the work.** At 16 scenarios the mined lead was +0.141, but `reg_stale_lead_2.0s` alone was
   found 94% of the time against 23%. Without it the lead falls to +0.026.
2. **Five of the six bugs share a theme.** They all concern the car ahead, and dev failures were full of that.
3. **It paid again in coverage.** At 16 scenarios, mined packs covered 77% of families against 89%, and raised false flags 3%
   of the time against 0%.

**The best lesson is the bug dev could not see.** We wrote `reg_blind_on_curve`, a car that forgets the vehicle ahead
whenever the road bends. On dev it looked perfect, with no effect at all, because no dev scenario has a car ahead on a curve.
On the new roads it cost 0.221 of safe completion, and random packs found it more often than mined ones (51% against 38%).

Mining looks for failures *like the ones you have already seen*. It is weakest at the failures you have never met. So do
both: mine for depth, and keep some random or unusual cases for surprises.

It went the other way too. Two bugs that hurt on dev (`reg_offset_2.0` and `reg_weave_2.0`) did not hurt on the new roads. So
dev gives you evidence about the roads in dev, and nothing stronger.

---

# Glossary

| Word | Plain meaning |
|---|---|
| A/A test | Compare a thing with itself to see how often the test falsely says "different" |
| ADE | Average distance between predicted and real positions |
| Bootstrap | Resample your data many times to see how much an answer wobbles |
| Closed loop | The model really drives, and its actions affect what it sees next |
| Cluster | A group of related measurements, here one scenario with its seeds |
| Compounding error | Small mistakes leading to bigger mistakes |
| Confidence interval | A range of plausible values for the true answer |
| Denominator | The bottom of a fraction, meaning what you count as "all cases" |
| Design effect | How much correlation shrinks your real sample size |
| Distribution shift | Facing situations unlike the training data |
| Expert | The careful rule-based driver that produced the training recordings |
| FDE | Prediction error at the last time point |
| Family | A type of scenario, such as tight curves |
| Fixture | A small, controlled test world built by hand |
| Holdout | Data locked away until the final test |
| Leakage | Test information sneaking into training or selection |
| Manifest | A file recording exactly how a run was produced |
| Mutation testing | Breaking code on purpose to check that the tests notice |
| Open loop | Scoring predictions against a recording, with no consequences |
| Paired test | Comparing two things under identical conditions |
| Policy | Anything that decides how to drive |
| Pre-registration | Fixing the analysis plan before seeing results |
| Regression | A change that makes things worse |
| Ridge regression | A simple, stable way to fit a formula to data |
| Seed | A number that fixes "random" events so they can be repeated |
| Spearman correlation | How well two rankings agree, from -1 to +1 |
| Tracker | The fixed code turning plans into steering and pedals |
| Unassessable | The data cannot support a verdict |

# Concept index

| Concept | Chapter |
|---|---|
| A/A test | 9, 10 |
| Clopper-Pearson bound | 7 |
| Clustered confidence intervals | 7 |
| Collision detection | 6 |
| Copycat problem | 3, 8 |
| Denominators | 6 |
| Distribution shift | 3, 8 |
| Failure mining | 10 |
| Fair comparison | 4 |
| Leakage | 5, 9 |
| Mutation testing | 9, 11 |
| Paired seeds | 4 |
| Pre-registration | 11 |
| Provenance and manifests | 11 |
| Safe completion | 6 |
| Sensitivity analysis | 8 |
| Simulator versus fixture | 8, 12 |
| Spearman correlation | 8 |
| Unassessable status | 6 |
