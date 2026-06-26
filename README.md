# Contribution #: 870

- Contribution Number: 1
- Student: Devi Sri Bandaru
- Issue: https://github.com/tinaudio/synth-setter/issues/600
- Status: Phase III Complete (fix implemented, tests passing)

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

- [x] test_steps_log_metrics_with_explicit_batch_size (tests/models/test_ksin_ff_module.py): drives the ksin feed-forward train, val and test steps with a mocked logger and asserts every self.log call passes batch_size equal to the input batch size. Built on the existing mock-log helpers already in that file.

The ksin flow-matching and Surge Flow-VAE modules get the same mechanical change. The flow-matching path is also exercised end to end by the integration test below, and the Surge path runs only under the VST-gated suite (requires_vst), which I cannot run locally.

### Integration Tests

- [x] test_train_fast_dev_run_emits_no_batch_size_warning (tests/test_train.py): runs a real fast_dev_run training run (the default model=ffn, datamodule=ksin, trainer=cpu config) and asserts no ambiguous-batch_size warning is raised. It captures warnings directly with warnings.catch_warnings because the suite's filterwarnings = ["ignore::UserWarning"] would otherwise hide the PossibleUserWarning.
- [x] Full fast suite: uv run make test-fast gives 2413 passed, 98 skipped. The single failure (test_pinned_baselines_resolve) is a pre-existing @pytest.mark.network test that fetches a remote baseline ref and cannot run in my offline environment, so it is unrelated to this change.

### Manual Testing

Re-ran the Phase II reproduction command before and after the fix:
uv run pytest "tests/test_train.py::test_train_fast_dev_run_tiny_model_tiny_data" -W always -p no:cacheprovider
Before the fix the ambiguous-collection warning printed; after the fix there are zero such warnings.

---

## Implementation Notes

### Week 1 Progress (Phase III)

What I built:
- Added an explicit batch_size to every self.log and self.log_dict call in the three LightningModules named in the issue. The batch size comes from the input tensor's first dimension: inputs.shape[0] in ksin_ff, batch[0].shape[0] (train) and inputs.shape[0] (val/test) in ksin_flow_matching, and mel_spec.shape[0] in surge_flowvae. In surge_flowvae I unpacked mel_spec from model_step instead of re-indexing the batch, so the tuple-typed step signatures stay type-clean.
- ksin_flow_matching also logs grad norms in on_before_optimizer_step with on_epoch=True, and that hook has no batch to infer from. I cache the batch size during training_step (self._batch_size, initialized to None in __init__) and reuse it there, so the logged values are unchanged.
- Added two tests (see Testing Strategy) and confirmed the warning is gone.

Challenges faced:
- The hardest part was realizing the warning is hidden by pyproject.toml's filterwarnings = ["ignore::UserWarning"], so the test suite never showed it. My integration test captures warnings directly to get around that.
- I added batch_size to the log calls in ksin_ff test_step but at first forgot to define batch_size at the top of that method. ruff check (and a quick re-read of the diff) caught the undefined name before I committed.
- make test-fast first failed with ModuleNotFoundError: h5py because the bare pytest in the Makefile resolved to my base conda Python, not the project .venv. Running uv run make test-fast fixed it.

### Code Changes

- Files modified: src/synth_setter/models/ksin_ff_module.py, src/synth_setter/models/ksin_flow_matching_module.py, src/synth_setter/models/surge_flowvae_module.py, tests/models/test_ksin_ff_module.py, tests/test_train.py
- Branch: https://github.com/Devisri-B/synth-setter/tree/fix-issue-600
- Key commits:
  - [4c01c42](https://github.com/Devisri-B/synth-setter/commit/4c01c42): fix(monitoring): pass batch_size to self.log in LightningModules
  - [2c75c6e](https://github.com/Devisri-B/synth-setter/commit/2c75c6e): test(monitoring): cover explicit batch_size in module logging
- Approach decisions: take the batch size from the input tensor's first dimension and change only the logging arguments, not the metric values. The value Lightning inferred was already correct, so the metrics themselves do not change.

---

## Pull Request

PR Link: https://github.com/tinaudio/synth-setter/pull/1754

PR Description: Adds an explicit batch_size to every self.log / self.log_dict call in the three LightningModules named in #600, so Lightning no longer warns about inferring batch_size from an ambiguous (tuple/dict) batch. Includes a unit test and a warning-capturing integration test.

Maintainer Feedback:
- (awaiting first review)

Status: Awaiting review
---

## Learnings & Reflections

### Technical Skills Gained

- How PyTorch Lightning infers batch size when logging metrics, and why passing batch_size explicitly matters for non-tensor batches (tuples and dicts).
- Reading a project's pytest configuration (filterwarnings) to understand why a real warning was invisible in CI.
- Writing a warning-capturing regression test with warnings.catch_warnings, and extending an existing mock-based unit test file.
- Working with uv and a project .venv, and matching a repo's ruff formatting and conventional-commit conventions.

### Challenges Overcome

The thing that stuck with me is that a clean test run does not mean a clean codebase. The warning was hidden by an ignore UserWarning filter, and the only way to find it was to read the pytest config in pyproject.toml instead of trusting the green check.

### What I'd Do Differently Next Time

I would run ruff check on each file right after editing it instead of after all three. That would have caught the undefined batch_size in test_step right away rather than at the end.

---

## Resources Used

- PyTorch Lightning logging and batch_size docs: https://lightning.ai/docs/pytorch/stable/extensions/logging.html
- The Lightning source that raises the warning: lightning/pytorch/utilities/data.py (_extract_batch_size)
- Project docs: CONTRIBUTING.md, CLAUDE.md, README.md (the installation table)
- Issue: https://github.com/tinaudio/synth-setter/issues/600
