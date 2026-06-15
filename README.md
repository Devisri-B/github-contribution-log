# Contribution #: 870

- Contribution Number: 1
- Student: Devi Sri Bandaru
- Issue: https://github.com/tinaudio/synth-setter/issues/600
- Status: Phase II Complete (reproduced + plan)

---

## Why I Chose This Issue

This issue sharpens my proficiency in PyTorch Lightening and helps me deep dive into machine learning workflows. By working on this issue, I can learn the importance of clean, meaningful logs; eliminating redundant warnings.

---

## Understanding the Issue

### Problem Description

Every training run of synth-setter prints a PyTorch Lightning warning saying it is "trying to infer the batch_size from an ambiguous collection". The Lightning modules call self.log(...) and self.log_dict(...) without passing a batch_size, so Lightning tries to guess the batch size by looking through the batch. The catch is that the batch is not a plain tensor. For the k-sinusoidal models it is a tuple of (audio, params, synth_fn), and for the Surge Flow-VAE it is a dictionary with "params" and "mel_spec". Since there is more than one tensor in there, Lightning cannot tell which one carries the batch dimension, so it warns on every step.

### Expected Behavior

Training, validation and test logging should run without any batch size warning, and each logged metric should be aggregated using the correct batch size that we pass in ourselves.

### Current Behavior

On every run Lightning prints this (it comes from lightning/pytorch/utilities/data.py line 79):

```
Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 1. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
```

The value it guessed is actually correct here, but it is still log noise, and the metric aggregation ends up relying on a guess instead of a real value.

### Affected Components

- src/synth_setter/models/ksin_ff_module.py (10 self.log calls)
- src/synth_setter/models/ksin_flow_matching_module.py (10 self.log calls)
- src/synth_setter/models/surge_flowvae_module.py (6 self.log and self.log_dict calls)
- The warning is raised inside Lightning's _extract_batch_size helper in lightning/pytorch/utilities/data.py.

The other Surge modules (surge_ff_module.py, surge_flow_matching_module.py, surge_fake_oracle_module.py) have the same pattern, but they are not in the list of files this issue asks me to fix, so I am leaving them as a possible follow up.

---

## Reproduction Process

### Environment Setup

I am on macOS (Apple Silicon) using Python 3.11 in the project's .venv with uv as the package manager. I cloned my fork and ran uv sync from the repo root. The README has a per platform table that says to use uv sync --frozen on macOS, and plain uv sync worked for me too. The prerequisites listed in CONTRIBUTING.md are uv, GNU make, and Python 3.10 or newer.

The part that tripped me up: running make test-fast looks completely clean and the warning never shows up. That is not because the bug is gone. pyproject.toml sets filterwarnings to ignore UserWarning, and the Lightning warning is a PossibleUserWarning, which is a subclass of UserWarning, so the test suite quietly hides it. I got it to show up by turning warnings back on with -W always on the command line. You can also see it from a normal training run, where the pytest filter does not apply. This is also why the upstream CI stays green even though the warning is there.

Working branch: https://github.com/Devisri-B/synth-setter/tree/fix-issue-600

### Steps to Reproduce

1. Clone the fork and go into it:
   git clone https://github.com/Devisri-B/synth-setter.git && cd synth-setter
2. Install the dependencies:
   uv sync
3. Run the project's own fast dev run smoke test with warnings turned back on:
   uv run pytest "tests/test_train.py::test_train_fast_dev_run_tiny_model_tiny_data" -W always -p no:cacheprovider
   This runs a real Trainer with fast_dev_run=True on the default config (model=ffn, datamodule=ksin, trainer=cpu), which is one train step, one val step, and one test step.
4. What should happen: the test passes and there is no batch size warning.
5. What actually happens: the test still passes, but Lightning prints "Trying to infer the batch_size from an ambiguous collection ... use self.log(..., batch_size=batch_size)".

If you would rather see it from a normal training run (no -W flag needed there, since the pytest filter does not apply outside pytest):
uv run synth-setter-train datamodule=ksin model=ffn trainer=cpu trainer.fast_dev_run=true

### Reproduction Evidence

Branch in my fork: https://github.com/Devisri-B/synth-setter/tree/fix-issue-600

Captured log (I ran it twice and got the same result both times):

```
tests/test_train.py::test_train_fast_dev_run_tiny_model_tiny_data
  .../lightning/pytorch/utilities/data.py:79: Trying to infer the `batch_size` from an ambiguous collection. The batch size we found is 1. To avoid any miscalculations, use `self.log(..., batch_size=batch_size)`.
======================== 1 passed, 9 warnings in 11.90s ========================
```

What I found: the warning only shows up when a real Trainer drives the modules' self.log calls. The unit tests in tests/models/test_ksin_ff_module.py mock out self.log, so they never hit it. The batch that reaches each step is a tuple or a dict, not a single tensor, which is exactly what makes the inference ambiguous. The real batch size is the first dimension of the input audio tensor.

---

## Solution Approach

### Analysis

self.log and self.log_dict work out the batch size by scanning the batch for tensors. For a tuple like (x, y, synth_fn) or a dict like {"params", "mel_spec"} there are several tensors plus a non tensor synth_fn, so Lightning cannot pick one with confidence and falls back to a guess with a warning. The value we actually want is the first dimension of the input audio tensor, which is x.shape[0] for the ksin models and mel_spec.shape[0] for the flow VAE.

### Proposed Solution

At the top of each training_step, validation_step and test_step, read the batch size once from the input tensor, then pass batch_size=batch_size into every self.log and self.log_dict call in that step.

### Implementation Plan

Using the UMPIRE framework (adapted):

Understand: Lightning warns on every run because the modules log metrics without telling Lightning the batch size, and the batch is a collection it cannot measure on its own. The fix is to pass the batch size in explicitly. The metric values should not change, since the guessed value was already right.

Match: The warning message itself tells you the fix, which is self.log(..., batch_size=batch_size). The codebase already uses the .shape[0] idiom for batch size. For example ksin_flow_matching_module.py line 162 does batch_size = z.shape[0], so I will follow that and take it from the input audio tensor. self.log_dict takes the same batch_size keyword as self.log.

Plan:
1. ksin_ff_module.py: training_step, validation_step and test_step already unpack inputs (the audio tensor) from model_step, so I add batch_size = inputs.shape[0] and pass it into all 10 self.log calls.
2. ksin_flow_matching_module.py: in training_step take batch_size = batch[0].shape[0], and in validation_step and test_step use inputs.shape[0] (already returned by _sample). Pass batch_size into all the self.log calls (train/loss, train/penalty, val/lsd, val/chamfer, test/param_mse, test/lsd, test/chamfer, test/lad).
3. surge_flowvae_module.py: take batch_size = batch["mel_spec"].shape[0] and pass it into every self.log and self.log_dict in training_step, validation_step and test_step.
4. Add a regression test (see Evaluate) and run the fast suite to check for regressions.

Implement: This is Phase III. Commits will go on https://github.com/Devisri-B/synth-setter/tree/fix-issue-600

Review: I will self review against CONTRIBUTING.md and CLAUDE.md before opening the PR. That means running make format (it runs the pre-commit hooks: ruff, ruff-format, pydoclint, prettier, gitlint), never using --no-verify, using a Conventional Commit message (the repo uses an internal-fix(monitoring): prefix for unreleased code), keeping the change small, and putting Closes #600 in the PR body so it closes the issue on merge.

Evaluate:
- Re run the reproduction command from step 3 and confirm the ambiguous collection warning is gone.
- Add a regression test that runs the fast dev run smoke test and checks that no PossibleUserWarning about batch_size is raised. It has to catch the warning on purpose with recwarn or pytest.warns, since the ini config ignores UserWarning.
- Run make test-fast and confirm the existing tests still pass.

---

## Testing Strategy

### Unit Tests

- [ ] Test 1: ksin_ff training_step calls self.log with a batch_size equal to inputs.shape[0]. This extends the existing mock-log tests in tests/models/test_ksin_ff_module.py.
- [ ] Test 2: ksin_flow_matching validation_step and test_step pass batch_size on every self.log call.
- [ ] Test 3: surge_flowvae training_step passes batch_size on both self.log and self.log_dict.

### Integration Tests

- [ ] test_train_fast_dev_run_tiny_model_tiny_data raises no ambiguous collection warning when captured with recwarn.
- [ ] The full make test-fast suite still passes.

### Manual Testing

Run the reproduction command before and after the fix:
uv run pytest "tests/test_train.py::test_train_fast_dev_run_tiny_model_tiny_data" -W always -p no:cacheprovider
The warning is there before the fix and gone after it.

---

## Implementation Notes

### Week [X] Progress

Phase III, to be filled in as I implement.

### Code Changes

- Files to modify (Phase III): ksin_ff_module.py, ksin_flow_matching_module.py, surge_flowvae_module.py, plus a regression test.
- Key commits: Phase III links.
- Approach decisions: take the batch size from the input audio tensor's first dimension, and only change the logging arguments, not the metric behavior.

---

## Pull Request

PR Link: [GitHub PR URL when submitted]

PR Description: [Draft or final PR description, much of the content above can be adapted]

Maintainer Feedback:
- [Date]: [Summary of feedback received]
- [Date]: [How you addressed it]

Status: [Awaiting review / Iterating / Approved / Merged]

---

## Learnings & Reflections

### Technical Skills Gained

[What you learned technically]

### Challenges Overcome

The thing that stuck with me is that a clean test run does not mean a clean codebase. The warning was hidden by an ignore UserWarning filter, and the only way to find it was to read the pytest config in pyproject.toml instead of trusting the green check.

### What I'd Do Differently Next Time

[Reflection on your process]

---

## Resources Used

- PyTorch Lightning logging and batch_size docs: https://lightning.ai/docs/pytorch/stable/extensions/logging.html
- The Lightning source that raises the warning: lightning/pytorch/utilities/data.py (_extract_batch_size)
- Project docs: CONTRIBUTING.md, CLAUDE.md, README.md (the installation table)
- Issue: https://github.com/tinaudio/synth-setter/issues/600
