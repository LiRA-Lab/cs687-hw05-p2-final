# CS 687 · Homework 5 · Project 2, final checkpoint

This is a standalone student repository for Homework 5. It follows three parts
of the Lecture 4 notes, "The cost of a thousand steps", "A deterministic
sampler" and "Determinism and its costs", and the Lecture 5 notes from "The
path of the noising process" to "Straight paths and few steps". You write the
step of DDIM, which samples the noise predictor of Homework 4 in a few
deterministic steps; train a network by flow matching on straight lines
between noise and data and sample it with Euler steps; show that DDIM is the
Euler sampler run at other times; and measure how the quality of samples of
the two clusters falls as the number of network calls falls from a thousand
to one, for the ancestral sampler, DDIM and the Euler sampler.

This checkpoint completes Project 2. Homework 4 built the noise predictor and
the ancestral sampler, which needs a thousand calls; this one adds the two
faster samplers and the measurement.

The repository uses no code of Project 1. It ships the course's code of
Homework 4 that this homework runs, complete: the noise schedule, the networks,
the two clusters with their exact noise prediction, the noisy image at any
level and the loss of DDPM. It also ships the larger ancestral step, the exact
mean velocity of the two clusters, the velocity versions of the two networks,
the training loop, the three sampling loops, the reader of the digits, and the
course's trained noise predictor of the digits.

Everything on the two clusters runs on the processor in a minute or two.
Section 8 trains a network on the 60,000 training digits and needs a GPU
runtime: the training takes about 17 minutes on a Colab T4 and about 7 minutes
on an L4, and sampling the digits takes a few seconds.

## What you must complete

The Colab notebook contains thirteen assessed answer cells:

- four short guided-calculation cells;
- five programming-task cells; and
- four Markdown report-response cells.

The programming tasks are:

| task | tagged notebook cell | function |
|---|---|---|
| Task 1 | `answer-task-1` | `ddim_step` |
| Task 2 | `answer-task-2` | `flow_matching_loss` |
| Task 3 | `answer-task-3` | `euler_step` |
| Task 4 | `answer-task-4` | `estimates_from_velocity` |
| Task 5 | `answer-task-5` | `line_time` |

All nine code-answer cells contain `NotImplementedError` placeholders. The
surrounding code is supplied complete: the schedule, the networks, the
training loop, which calls your Task 2, the three sampling loops, which call
your Tasks 1 and 3, the exact noise prediction and the exact mean velocity of
the two clusters, and the plots. The four report cells contain written response placeholders. Your
completed notebook is the authoritative submission; you do not need to copy
its implementations into `.py` files.

## Grading

Homework 5 is graded out of 100 points:

| assessed work | points |
|---|---:|
| Four guided calculations | 16 |
| Task 1: one DDIM step | 7 |
| Task 2: the loss of flow matching | 7 |
| Task 3: one Euler step | 3 |
| Task 4: the estimates from a velocity | 3 |
| Task 5: the time of a level on the line | 4 |
| Four written responses | 60 |
| **Total** | **100** |

All nine code exercises are graded all-or-nothing. Each guided calculation
receives either 4 points or 0 points. Each task receives its points only if it
passes every staff-controlled test of that task. Partially correct code does
not receive partial credit. The written responses are graded separately.

The runs themselves are not graded. Their outputs, the statistics of the
samplers, the training curve and the tables of losses by time and by level,
the comparison of DDIM with the Euler sampler, the estimate along the paths,
the measurement against calls and the digits, are the evidence your written
responses are read against, so save the notebook with its outputs. Grading runs your
notebook again from a clean environment on a processor, to check the
calculations and the five tasks, with a short digit run in place of yours.

## Google Colab

The notebook, `homework05_colab.ipynb`, is attached to the Moodle assignment
for Homework 5. Download it from there and upload it to Google Colab. Work in
that uploaded notebook and save it there; do not create a new notebook and
paste the cells into it, because a new notebook loses the cell identifiers and
metadata that grading depends on. Choose a GPU runtime under Runtime > Change
runtime type before you start, because changing the runtime later restarts
the notebook. Run the notebook from top to bottom. Its setup cell clones this
repository and installs the required Python packages. After setup the notebook
needs no network access.

The notebook lets you develop all five implementations interactively. A
focused check follows each code-answer cell so that you receive feedback
before continuing. Near the end, a final cell runs all four calculation checks
and all 25 public tests together. Run these cells inside the notebook: a
separate `pytest` process cannot see definitions that exist only in the
notebook kernel.

The digit run of section 8 saves the trained network in
`runs/digits_velocity.pt`, and running the cell again restores it in seconds
instead of training again, as long as the session's disk is still there.
Delete `runs/digits_velocity.pt` to train again from scratch, for example
after changing Task 2. The sections on the two
clusters are seeded and print the same values every time they run on the same
machine.

You may work on your own computer instead if you can set up the required
Python and Jupyter environment yourself.

## Homework workflow

1. Read the parts "The cost of a thousand steps", "A deterministic sampler"
   and "Determinism and its costs" of the Lecture 4 notes, and the Lecture 5
   notes from "The path of the noising process" to "Straight paths and few
   steps".
2. Work through the notebook in order. Complete the four guided-calculation
   cells and the five programming-task cells, and run each immediate check.
3. Restart the runtime and run the complete notebook from top to bottom.
4. Confirm that the final check reports that all four guided calculations and
   all 25 public tests passed.
5. Complete the four questions in the notebook's report section, using
   evidence from your own runs. If a run trained again before you submit,
   check that every value and phrase your report quotes matches the new
   outputs.

Do not modify the tests to make an implementation pass. The tests describe the
required behavior and grading uses a staff-controlled copy. Write the five
functions yourself: the point of the homework is to see what is inside the
two samplers and the trainer.

## Protect the notebook structure

Automatic grading depends on the released notebook structure. Follow these
rules while working:

- In a code-answer cell, edit only between `YOUR CODE STARTS HERE` and
  `YOUR CODE ENDS HERE`, and keep both marker lines.
- In a written-answer cell, replace only the placeholder below `**Response:**`.
  Do not remove the response marker.
- Do not delete, duplicate, or reorder supplied cells. You may add separate
  scratch cells for your own experiments. Grading removes every cell that is
  not part of the released notebook before it runs your notebook, so a scratch
  cell never counts: each answer must work on its own inside its answer
  region, with every helper and import it needs written there.
- Do not convert a supplied code cell to Markdown or a supplied Markdown cell
  to code.
- Do not change notebook metadata, cell tags, stable cell IDs, the assignment
  identifier, or the assignment-version number.
- Do not edit supplied instructions, demonstrations, setup code, checks, or
  code outside marked answer regions.
- Keep the outputs of sections 2 to 8 in the saved notebook, and leave
  Colab's setting "Omit code cell output when saving this notebook" off.

Execution counts do not matter, because grading runs your notebook again from
a fresh kernel.

If you accidentally change the structure, obtain a fresh copy of the released
notebook and copy only your answers into its designated answer regions. A
submission whose structure has changed is set aside for manual inspection and
loses 20 points, because staff have to repair the file by hand before it can
be graded. An answer that cannot be identified reliably receives zero for that
component.

## Deliverables

- the completed Homework 5 notebook, with its outputs

The completed notebook is the single Homework 5 submission. Save the notebook
in Colab, rename it to `homework05_colab_STUDENTNUMBER.ipynb`, replacing
`STUDENTNUMBER` with your student number, for example
`homework05_colab_22206543.ipynb`, and download it with File > Download >
Download .ipynb. Upload that exact file, and only that file, to the Moodle
assignment for Homework 5. Moodle's account record, rather than the filename
alone, remains the authoritative student identity. The required filename
format is nevertheless mandatory, and a submission whose file name does not
follow it loses 10 points.

A late submission loses 10 points for every hour past the submission time,
and a started hour counts as a whole hour, so a submission ten hours late
receives zero. Moodle's record of the upload time is what counts.

You may submit unanswered work; an unanswered cell receives zero for that
component. An unreadable file or a file that is not an `.ipynb` notebook
cannot be graded and receives a grade of zero.

## Repository layout

```text
README.md            this file
requirements.txt     the Python packages the notebook installs
pytest.ini           the test configuration the notebook's checks use
notebook_support.py  supplied setup, calculation checks, plotting and public-test helpers
cs687/               the course package: the schedule, the networks, the two clusters, the digits,
                     the training loop, the three samplers, and the five Homework 5 tasks left for you
tests/               inspectable public tests run by the notebook, one file per task
data/                the 60,000 training images of the MNIST database of handwritten digits,
                     in their original file
weights/             the course's noise predictor of the digits, trained in Homework 4's digit run
notebooks/           the Colab entry point and authoritative code submission
```
