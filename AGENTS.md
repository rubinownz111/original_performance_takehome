# Optimization Journal and Agent Instructions

## Objective
- Reduce cycle count below current baseline while preserving correctness on `python3 tests/submission_tests.py`.

## Hard Rules For Agents Working In This Repo
- Do not modify files under `tests/`.
- Every time you find something new and valuable, update this file immediately.
- Update this file again after each attempted path, whether it works or fails.
- Keep a chronological attempt log with concrete measurement numbers.
- "Something new and valuable" includes:
- A working improvement.
- A dead end that looked promising but regressed or broke correctness.
- A new bottleneck discovered from telemetry or measurement.
- A new architectural idea worth trying next.
- For each logged attempt include:
- What was changed/tested.
- Why it was attempted.
- Result (`cycles`, correctness pass/fail).
- Decision (`keep`, `revert`, `follow-up`).

## Current Baseline
- Command: `python3 tests/submission_tests.py`
- Result: `1065` cycles, correctness passing.
- Kernel stats: `12375 ops`, peak scratch `1528/1536`.
- Note: the currently checked-out worktree on `2026-03-05` re-validates at this `1065` baseline. Older `1078-1082` entries below are useful historical context from earlier architecture exploration, but they do not describe the present file state.

## Findings So Far
- Deep scatter result storage can now be rewritten in place without breaking correctness:
- A new lane-precise partial-vector writer model allows scalar scatter XORs to write directly back into the live `values` vector.
- This removes `244` emitted ops from the `1065` kernel (`12375 -> 12131`) with correctness preserved, but cycle count and slot totals stay unchanged (`1065`, `load=2001`, `valu=6010`, `alu=12122`).
- Implication: a large fraction of remaining emitted ops are non-slot-bearing allocation/dataflow scaffolding; removing them alone is not enough unless the rewrite also changes engine-slot pressure or critical-path overlap.
- Current checked-out `1065` baseline re-measurement on `2026-03-05` confirms the same three hard bottlenecks:
- Final emitted bundle slot totals: `alu=12122`, `valu=6010`, `load=2001`, `flow=785`, `store=32`.
- Scheduler telemetry on this exact state still shows load pressure first:
- `defer_load_full: 30906`
- `defer_flow_full: 8090`
- `defer_valu_full: 6105`
- `defer_valu_full_offload_failed: 4184`
- Implication: reducing scratch lifetime alone is unlikely to help unless it also removes real `load` or `valu` work from the critical rounds.
- Engine pressure is dominated by `valu` and `load`:
- Approx utilization: `valu ~93.9%`, `load ~100%`, `alu ~101.1%`, `flow ~78.5%` (slot totals vs hard bounds).
- Current 1065-case scheduled slot totals / hard lower bounds:
- `valu: 6010` slots -> LB `1002` cycles
- `load: 2001` slots -> LB `1001` cycles
- `alu: 12130` slots -> LB `1011` cycles
- `flow: 785` slots -> LB `785` cycles
- Implication: sub-1000 now requires real reductions in at least three engines:
- `load` (`<=2000`), `alu` (`<=12000`), and `valu` (`<=6000`).
- Important correction on ALU accounting:
- Scheduler telemetry (`slot_alu`) undercounts ALU slots because retroactive ALU offload injects ALU ops into prior bundles, but telemetry records only the current cycle's `slot_counts`.
- Counting final emitted bundles (`kb.instrs`) gives true ALU total for current config:
- `alu: 12130` slots -> LB `1011` cycles
- `valu: 6010` slots -> LB `1002` cycles
- `load: 2001` slots -> LB `1001` cycles
- `flow: 785` slots -> LB `785` cycles
- Updated implication: we are back in a tri-bottleneck regime (`alu` + `load` + `valu`) despite lower flow pressure.
- External reference point:
- Community leaderboard (`kerneloptimization.fun`) currently shows a best of `1001` cycles for this challenge variant, which is consistent with our observation that current architecture families are converging near the low 1000s.
- Top defer reasons from scheduler telemetry:
- `defer_load_full: 31719`
- `defer_valu_full: 10534`
- `defer_flow_full: 4323`
- `defer_valu_full_offload_failed: 2917`
- Deep tree traversal is the main load hotspot:
- Depths `5-10` use scatter path and contribute most scalar `load`+`alu` tree ops.
- New allocator/scheduler finding on the hash-temp reuse path:
- Retroactive ALU split-offload has a concrete vector-block leak for full-vector `write_to` ops.
- Root cause: `_alloc_safe_vector()` can pop/consume a free vector block before it notices `op.vbase` is already mapped (`op.vbase in alloc.vmap`), then returns `False` without restoring the block.
- Reproduction on `REUSE_HASH_COMBINE_TEMPS=True`: observed `68` leaked safe-vector attempts before the scheduler stuck at `421` cycles / `4471/12375` ops done.
- Implication: any write-to-based vector reuse path that allows retroactive ALU offload can deadlock from allocator leakage even if the dataflow itself is otherwise valid.
- Hash-temp reuse correctness is now narrowed further:
- After fixing the `_alloc_safe_vector()` leak, full hash-temp reuse no longer deadlocks, but it still fails correctness at `1070` cycles (`peak scratch 1521/1536`).
- The emitted graph/dataflow is not inherently wrong: a dense one-op-per-cycle execution of the same write-to graph is correct on a virtual-address machine.
- The remaining failure is specific to the tight physical allocator/scheduler regime:
- Re-running the same hash-temp reuse graph with the original scheduler and a very large scratch space (`2_000_000`) is correctness-valid at the same `53` cycles on the small repro case.
- Implication: the unresolved blocker is physical scratch mapping / alias scheduling under the `1536`-word layout, not the mathematical hash rewrite itself.

## Attempts and Outcomes
- Knob sweep on global/tuning constants (concurrency windows, recompute interval, offload policy, depth toggles, engine weights):
- No correctness-valid configuration beat `1082`.
- Several low-cycle outputs were invalid due to scheduler stuck states; these are dead ends.
- Decision: stop brute-force knob tuning and focus on architectural changes.

## Attempt Log
- 2026-03-05: Scatter-vector combine writes directly into `values`.
- What changed/tested: added `SCATTER_VECTOR_XOR_WRITE_TO_VALUES` so the depth-5 vector-scatter path can load into a temporary scatter buffer but write the final vector XOR result back into the live `values` vector instead of a fresh result vector or the scatter buffer. Tested `SCATTER_VECTOR_XOR=True`, `SCATTER_VECTOR_XOR_WRITE_TO_VALUES=True`, `SCATTER_VECTOR_XOR_DEPTHS={5}`.
- Why: check whether the prior `1071` regression from vector scatter combine was caused by result-buffer placement/liveness rather than by the ALU->VALU engine trade itself.
- Result: correctness pass at `1071` cycles, `12151` ops, slot totals `load=2001`, `valu=6027`, `alu=11986`, `flow=785` — identical to the write-to-scatter variant.
- Decision: keep disabled. Result placement is not the problem; the vector-scatter architecture remains non-competitive on current engine bounds.

- 2026-03-05: In-place scatter-vector combine retry after allocator leak fix.
- What changed/tested: added `SCATTER_VECTOR_XOR_INPLACE` so the vector scatter combine can overwrite the loaded scatter buffer, and blocked ALU offload for that specific in-place op id. Tested the architecturally best prior placement: `SCATTER_VECTOR_XOR=True`, `SCATTER_VECTOR_XOR_INPLACE=True`, `SCATTER_VECTOR_XOR_DEPTHS={5}`.
- Why: the previous in-place scatter-vector path had deadlocked before the `_alloc_safe_vector()` leak fix; this re-test checks whether the repaired scheduler makes the lower-ALU path viable.
- Result: correctness pass at `1071` cycles, `12151` ops, slot totals `load=2001`, `valu=6027`, `alu=11986`, `flow=785`.
- Decision: keep the new scaffolding disabled; the path is now stable and correctness-valid, but still regresses because the ALU savings convert into extra VALU pressure instead of lowering total cycles.

- 2026-03-05: Narrow in-place hash rewrite for multiply-add-only stages.
- What changed/tested: added `ENABLE_INPLACE_HASH_MULADD_ONLY` so only hash stages `0` and `4` overwrite the live `values` vector, leaving the temp-heavy combine stages unchanged.
- Why: try to harvest scratch-lifetime benefits from the obviously safe single-op hash overwrites without re-entering the alias bugs of the full in-place hash path.
- Result: correctness pass, but no measurable schedule change: `1065` cycles, `12375` ops, peak scratch still `1528/1536`.
- Decision: keep disabled by default; this rewrite is semantically safe but too weak to matter on its own.

- 2026-03-05: Deep scalar scatter writes directly into `values`.
- What changed/tested:
- generalized partial-vector dependency tracking from coarse block-level writers to lane-precise writers.
- added `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES` so deep scalar scatter XORs can write each lane back into the live `values` vector instead of allocating a separate output vector.
- also checked the natural follow-up with `ENABLE_INPLACE_HASH_MULADD_ONLY=True`.
- Why: remove one whole vector allocation from every deep scatter round, which is the highest-volume remaining SSA-style buffer in the kernel.
- Result:
- `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True` -> `1065` cycles, pass, `12131` ops.
- `ENABLE_INPLACE_SCALAR_SCATTER_TO_VALUES=True` + `ENABLE_INPLACE_HASH_MULADD_ONLY=True` -> `1065` cycles, pass, `12131` ops.
- Peak scratch remained `1528/1536`, and final slot totals were unchanged from baseline.
- Decision: keep the rewrite available but disabled. It is a real graph simplification and now enables future in-place deep-path work, but by itself it does not reduce cycle count.

- 2026-03-05: Local round-gate retune around the current interleaving map.
- What changed/tested: searched a tight neighborhood around `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}` using nearby windows on those same rounds only.
- Why: round gates change overlap without changing semantics or adding new ops, so this was the last configuration-only surface worth checking before returning to structural rewrites.
- Result:
- Best measured configs only tied baseline at `1065` cycles:
- `{7:28,9:24,10:28}` -> `1065` cycles, pass, `12375` ops.
- `{7:26,9:26,10:28}` -> `1065` cycles, pass, `12375` ops.
- `{7:28,9:26,10:28}` -> `1065` cycles, pass, `12373` ops.
- No neighborhood point improved below `1065`; most regressed to `1068+`.
- Decision: keep the existing gate map for now. The tied lower-op map `{7:28,9:26,10:28}` is interesting as scaffolding for future architectural changes, but not worth a baseline change by itself.

- 2026-03-05: Late shallow-round vselect mask single-flip neighborhood search.
- What changed/tested: evaluated all single-group flips around the current late shallow rounds:
- `DEPTH_3_VSELECT_GROUPS_BY_ROUND[14]` relative to the default depth-3 heuristic mask.
- `DEPTH_4_VSELECT_GROUPS_BY_ROUND[15]` relative to the default depth-4 heuristic mask.
- Why: current `1065` telemetry is load-bound, and late rounds `14/15` are the lowest-risk place to trade scatter loads for extra shallow `vselect` work without introducing long-lived setup vectors.
- Result:
- Baseline explicit late-round masks -> `1065` cycles, pass.
- Best depth-3 mutation: remove `g=1` from round-14 mask -> `1067` cycles, pass.
- Best depth-4 mutation: add `g=14` to round-15 mask -> `1071` cycles, pass.
- Several mutations reduced `load` from `2001` to `1993`, but the added flow/valu pressure still regressed overall schedule length.
- Decision: keep the current late-round masks unchanged; shallow single-group flips are not enough to beat `1065`.

- 2026-03-05: Delayed late shallow-tree setup duplication (`DELAY_LATE_D3D4_SETUP`).
- What changed/tested: enabled duplicate depth-3/depth-4 setup for late rounds so the early shallow setup could die after round 4; also confirmed the same result with `DELAY_DEPTH_5_SETUP=True` (which is effectively inert in the current baseline because `DEPTH_5_VSELECT_GROUPS=set()`).
- Why: the new scaffolding looked like a plausible way to lower long-lived scratch pressure and improve schedule freedom without changing tree semantics.
- Result:
- `DELAY_LATE_D3D4_SETUP=False` -> `1065` cycles, pass
- `DELAY_LATE_D3D4_SETUP=True` -> `1103` cycles, pass
- `DELAY_LATE_D3D4_SETUP=True`, `DELAY_DEPTH_5_SETUP=True` -> `1103` cycles, pass
- Ops increased from `12375` to `12420`.
- Decision: keep both delay flags disabled; duplicating shallow setup adds too much extra load/valu work to pay for the shorter lifetime.

- 2026-03-05: Re-validated the checked-out worktree baseline before new optimization work.
- What changed/tested: ran `python3 tests/submission_tests.py` on the current dirty worktree without changing code.
- Why: the journal contained older `1078-1082` history, so the first step was to confirm the actual baseline of the present file state before attempting new optimizations.
- Result: `1065` cycles, correctness pass, `12375 ops`, peak scratch `1528/1536`.
- Decision: keep this as the active baseline for all further measurements in this worktree.

- 2026-03-05: Fixed retroactive offload leak for full-vector `write_to` ops.
- What changed: moved the `_alloc_safe_vector()` early-return checks (`write_size == 0`, `vbase < 0`, and `op.vbase in alloc.vmap`) ahead of free-list / watermark allocation so failed safe-vector attempts no longer consume a block.
- Why: `REUSE_HASH_COMBINE_TEMPS=True` exposed a concrete allocator leak where split-offload tried to allocate a new safe vector for an op that already had a mapped destination.
- Result:
- Baseline behavior unchanged when the feature is off; full `python3 tests/submission_tests.py` re-validation still passes at `1065` cycles.
- Full hash-temp reuse no longer deadlocks, but still fails correctness at `1070` cycles (`peak scratch 1521/1536`).
- `ENABLE_INPLACE_HASH_DATAFLOW=True` also now reaches `1070` cycles but still fails correctness.
- Additional narrowing:
- Dense one-op-per-cycle execution of the emitted write-to graph is correctness-valid.
- The original scheduler with a huge scratch budget (`2_000_000`) is also correctness-valid on the small repro case, so the remaining problem is allocator/layout-specific.
- Decision: keep the scheduler leak fix (real correctness bug in experimental scaffolding); continue investigating the tight-layout alias issue before enabling any hash write-to path.

- 2026-03-05: Stage-specific hash-combine temp reuse sweep.
- What changed: added `REUSE_HASH_COMBINE_TEMP_STAGES` so hash temp reuse can be enabled per stage (`1`, `2/3 fused`, `5`) instead of all-or-nothing.
- Why: isolate whether only one hash combine site is unsafe under the `1536`-word allocator while others remain correctness-valid and potentially cycle-helpful.
- Result (all full-kernel runs, correctness-checked):
- `stage1 ({1})` -> `1070` cycles, fail
- `stage23 ({2})` -> `1072` cycles, fail
- `stage5 ({5})` -> `1070` cycles, fail
- `stage1_23 ({1,2})` -> `1069` cycles, fail
- `stage1_5 ({1,5})` -> `1070` cycles, fail
- `stage23_5 ({2,5})` -> `1070` cycles, fail
- Decision: no stage-local subset is safe under the current allocator; keep `REUSE_HASH_COMBINE_TEMPS=False` and `REUSE_HASH_COMBINE_TEMP_STAGES=set()`.

- 2026-02-24: Implemented round-aware deep-round gating in dependency graph.
- What changed: Added deep-round gate dependencies in `build_kernel`, plus `start_deps` plumbing through `_emit_tree_xor` and `_emit_vselect_tree`.
- Why: limit deep scatter-round pressure independently from global group window.
- Result: initial version used vector refs as round tokens and caused scheduler stuck states (`~384` reported cycles, incorrect output).
- Decision: dead end; do not keep vector-token version.

- 2026-02-24: Reworked deep-round tokens to scalar barrier refs.
- What changed: token emitted as scalar ALU op dependent on deep tree completion, so cross-group gating does not extend vector liveness.
- Why: preserve round gate semantics while preventing scratch liveness blow-up.
- Result (window sweep, correctness-checked):
- `MAX_CONCURRENT_DEEP_ROUNDS=12` -> `1110` cycles (pass)
- `MAX_CONCURRENT_DEEP_ROUNDS=16` -> `1087` cycles (pass)
- `MAX_CONCURRENT_DEEP_ROUNDS=20` -> `1090` cycles (pass)
- `MAX_CONCURRENT_DEEP_ROUNDS=24` -> `1083` cycles (pass)
- `MAX_CONCURRENT_DEEP_ROUNDS=0` -> `1082` cycles (pass)
- Decision: feature kept but disabled by default (`MAX_CONCURRENT_DEEP_ROUNDS=0`) because all tested active windows regress.
- Validation: `python3 tests/submission_tests.py` passes with final state at `1082` cycles.

- 2026-02-24: Attempted in-place vector dataflow for tree/hash outputs.
- What changed: added vector `write_to` support and rewired tree/hash to overwrite `values` buffers in place.
- Why: reduce transient vector allocations and lower scratch pressure.
- Result: scheduler stalled with heavy `defer_scratch_full`, incorrect output, and early stop (`~210` cycles reported, invalid).
- Diagnosis: in-place writes conflicted with current ownership/free semantics and dependency fanout.
- Decision: reverted in-place path; keep original SSA-like vector outputs.

- 2026-02-24: Implemented generalized dead-index elimination (future-use analysis).
- What changed: replaced penultimate-round heuristic with per-group/per-round future index-use analysis.
- Why: remove unnecessary index materialization ops whenever no future round uses `ts.index`.
- Result: correctness pass; ops reduced `12401 -> 12365`; peak scratch reduced `1390 -> 1326`; cycles unchanged at `1082`.
- Decision: keep this change (strict improvement in op count/scratch, no cycle regression).

- 2026-02-24: Re-tested deep-round windowing after dead-index elimination.
- What changed: swept `MAX_CONCURRENT_DEEP_ROUNDS` with new lower-scratch kernel.
- Why: verify whether previous regression was scratch-driven and now recoverable.
- Result: no win. Best active window remained slower (`24 -> 1083`, `16 -> 1087`, `12 -> 1110`), `0 -> 1082`.
- Decision: initially kept disabled; later superseded by selective thresholded gating win (entry below).

- 2026-02-24: Selective deep-round gating by deeper threshold.
- What changed: targeted deep gating to depths `>=6` with `MAX_CONCURRENT_DEEP_ROUNDS=20`.
- Why: reduce gate overhead from earlier full deep gating while still smoothing load-heavy scatter rounds.
- Result: correctness pass and cycle improvement to `1081` (`python3 tests/submission_tests.py` full pass).
- Decision: keep this as new default (`DEEP_ROUND_DEPTH_THRESHOLD=6`, `MAX_CONCURRENT_DEEP_ROUNDS=20`).

- 2026-02-24: Added per-round/depth deep-window override framework.
- What changed: introduced `DEEP_GATE_WINDOW_BY_ROUND` / `DEEP_GATE_WINDOW_BY_DEPTH` and helper `_deep_gate_window`.
- Why: enable true per-round gating experiments instead of only threshold-based gating.
- Result: framework validated; default behavior unchanged at `1081` and full correctness pass.
- Decision: keep framework (enables future structural search without code churn).

- 2026-02-24: Explicit per-round deep-gate map search (global fallback disabled).
- What changed: tested explicit round maps on rounds `6..10` with window `20` and nearby structured variants.
- Why: verify if selective round gating could beat thresholded global winner.
- Result: best remained exactly the current full map equivalent (`{6,7,8,9,10}:20`) at `1081`.
- Decision: no config change.

- 2026-02-24: Scheduler architecture attempt: defer offloadable VALU ops for later ALU offload.
- What changed: added temporary deferral path (`DEFER_OFFLOADABLE_VALU`) when split-offload fails.
- Why: attempt to reduce VALU saturation by giving ALU offload extra cycles to succeed.
- Result: severe regression (`1212` cycles), correctness still passed.
- Decision: keep code path off by default (`DEFER_OFFLOADABLE_VALU=False`); treat approach as dead end in current form.

- 2026-02-24: Guided per-round deep-window coordinate search.
- What changed: explored per-round windows around `{6..10:20}` with explicit round-map overrides.
- Why: check if interactions among deep rounds can beat `1081`.
- Result: no improvement; best remained `{6:20,7:20,8:20,9:20,10:20}` at `1081`.
- Decision: keep current map-equivalent default.

- 2026-02-24: Load-relief attempt via preferred flow const materialization.
- What changed: added `PREFER_FLOW_CONST` path to emit non-zero consts via `flow/add_imm` when possible.
- Why: reduce load-engine pressure (`defer_load_full` is dominant).
- Result: regression to `1092` cycles (correctness pass).
- Decision: keep feature flag off by default (`PREFER_FLOW_CONST=False`).

- 2026-02-24: Deep-gate token overhead reduction attempt (vector tokens).
- What changed: added `USE_VECTOR_DEEP_TOKENS` option to reuse deep tree result vectors as gating tokens instead of scalar ALU token ops.
- Why: remove token-op overhead while preserving selective deep gating.
- Result: deadlock/incorrect (`~385` cycles reported, invalid output) when enabled.
- Decision: keep disabled (`USE_VECTOR_DEEP_TOKENS=False`); scalar tokens remain required for correctness/stability.

- 2026-02-24: In-place vector dataflow retry with allocator ownership transfer.
- What changed: added `ENABLE_INPLACE_VECTOR_DATAFLOW` path plus allocator ownership-transfer tracking (`vbase_owner` / `owner_vbase`) for full-vector write_to chains.
- Why: try to reduce scratch pressure and improve schedule freedom without semantic alias bugs.
- Result: still deadlocked/incorrect (`~123` cycles reported, invalid output) when enabled.
- Decision: keep disabled (`ENABLE_INPLACE_VECTOR_DATAFLOW=False`); approach still unresolved.

- 2026-02-24: Structural neighborhood check for depth-3/depth-4 vselect masks.
- What changed: evaluated single-group flips around current depth-3 and depth-4 mask sets under the `1081` configuration.
- Why: verify local optimality of current scatter/vselect partition after new gating changes.
- Result: no single flip improved beyond `1081`.
- Decision: keep current mask sets unchanged.

- 2026-02-24: Attempted round-major emission architecture.
- What changed: emitted rounds across all groups (experimental path) while preserving dependency correctness.
- Why: reshape schedule critical path and improve engine overlap.
- Result: correctness pass but major regression (`1261` cycles).
- Decision: reverted round-major path; keep group-major emission.

- 2026-02-24: Targeted depth-3 vselect alignment experiment (group-specific).
- What changed: forced depth-3 vselect on groups that were scatter in baseline (`g=11`, then `{11,23}`) to trade deep scatter loads for flow/valu work.
- Why: reduce load pressure at depth-3/late rounds where load is the dominant defer reason.
- Result: regressions (`g=11 -> 1093`, `{11,23} -> 1118`), correctness passed.
- Decision: revert/keep original depth-3 selection mask.

- 2026-02-24: Strided `group_offset` dependency chain experiment.
- What changed: replaced fully serial `group_offset[g]=group_offset[g-1]+V` with a strided chain to reduce setup critical depth.
- Why: make high-group load addresses available earlier without changing semantics.
- Result: no cycle gain (`1082`), +1 op overhead in implementation variant.
- Decision: reverted to original chain for simplicity.

- 2026-02-24: Re-audited slot lower bounds using final emitted bundles (not cycle telemetry).
- What changed: compared scheduler telemetry totals against direct counts from `kb.instrs` slot tuples.
- Why: verify true lower bounds before selecting next architecture direction.
- Result: discovered telemetry undercounts ALU due retroactive split-offload insertion into past bundles; corrected ALU total from `11178` to `12151` in current 1081 config.
- Decision: keep current code, but treat prior ALU lower-bound analysis as superseded. Future planning now targets ALU+VALU+LOAD reductions together.

- 2026-02-24: Compile-time header-pointer specialization attempt (`forest_values_p` / `inp_values_p`).
- What changed: replaced setup header loads (`mem[4]`, `mem[6]`) with constants derived from `build_mem_image` layout.
- Why: reduce load slots by removing two scalar memory loads in setup.
- Result: correctness passed but large regression to `1094` cycles; peak scratch increased from `1380` to `1474`.
- Decision: reverted. Treat as dead end due liveness/scheduling side effects outweighing load-slot savings.

- 2026-02-24: Added depth-5 selective vselect architecture path (configurable group mask + diff pairs).
- What changed: introduced `DEPTH_5_VSELECT_GROUPS`, `DEPTH_5_VSELECT_DIFF_PAIRS`, depth-5 setup vectors/diffs, and depth-5 tree branch in `_emit_tree_xor`.
- Why: attempt a structural reduction of deep scatter-load pressure.
- Result:
- Framework is baseline-neutral when disabled (`1081`, pass).
- Single-group depth-5 vselect (`diff_pairs=0`) correctness/pacing is highly group-sensitive: many placements trigger scheduler-stuck invalid outputs (e.g. reported `~385-484` cycles), and all correctness-valid placements regress (`~1093-1143`).
- Decision: keep framework off by default (`DEPTH_5_VSELECT_GROUPS = set()`); treat current depth-5 vselect direction as a dead end for cycle reduction.

- 2026-02-24: Offload-policy architecture audit (global and op-class selective).
- What changed: evaluated proactive/deferred offload modes and selective ALU-offload eligibility classes (`hash-only`, `no_hash`, `no_branch_index`, etc.).
- Why: validate whether sub-1000 requires a different offload architecture rather than more tree changes.
- Result: current default remained best at `1081`; alternatives regressed (`1086-1279`).
- Decision: keep current offload policy (`PROACTIVE_VALU_OFFLOAD=True`, `DEFER_OFFLOADABLE_VALU=False`) and default op eligibility.

- 2026-02-24: Dead branch-bit emission analysis and implementation.
- What changed: added future bit-use analysis (`_tree_consumed_bits`, `compute_future_bit_needed`) to skip `%2` branch-bit ops when neither future tree selection nor index update needs them.
- Why: remove dead vector `%` work and reduce VALU/ALU slots.
- Result: no branch-bit ops were actually dead in current baseline shape; emitted ops and cycles unchanged (`12425 ops`, `1081`).
- Decision: keep the analysis scaffolding (correct and safe), but no immediate performance gain.

- 2026-02-24: Retroactive ALU split completion-cycle propagation.
- What changed: `_try_split_alu_offload` now returns effective completion cycle and successor earliest times use that completion cycle.
- Why: avoid conservative latency when split-offloaded work finishes in earlier bundles.
- Result: no measurable change (`1081`) because no offloaded op completed strictly in past-only bundles under current schedule (`past_only = 0` observed).
- Decision: keep as correctness-preserving scheduler improvement, but currently performance-neutral.

- 2026-02-24: Scatter combine redesign (`SCATTER_VECTOR_XOR`).
- What changed: experimental scatter path that loads 8 node values into a vector then applies one vector XOR instead of 8 scalar ALU XORs.
- Why: reduce hard-wired ALU pressure and make scatter combine schedulable on VALU/ALU-offload.
- Result:
- In-place overwrite variant (`write_to=scatter`) caused invalid/stuck schedules.
- Safe non-inplace variant was correct but regressed to `1093` cycles.
- Decision: keep feature flag off by default (`SCATTER_VECTOR_XOR=False`); treat current form as dead end.

- 2026-02-24: Full depth-5 vselect conversion sweep (all groups, varied diff pairs and bit order).
- What changed: enabled `DEPTH_5_VSELECT_GROUPS = all groups`, swept `DEPTH_5_VSELECT_DIFF_PAIRS` and `REVERSE_TREE_BIT_ORDER_DEPTH_5`.
- Why: check whether a true deep architectural swap could break into sub-1000-like regime.
- Result: all tested configurations were invalid/stuck (very low bogus cycle counts such as `~53-199`, correctness failed).
- Decision: treat full depth-5 vselect conversion as an unstable dead end with current dependency/allocation model.

- 2026-02-24: Partial `SCATTER_VECTOR_XOR` by depth/group.
- What changed: added `SCATTER_VECTOR_XOR_DEPTHS` and `SCATTER_VECTOR_XOR_GROUPS` filters and tested targeted activation.
- Why: see if selective ALU->VALU trade helps without global regression.
- Result:
- Best selective cases were neutral or regressive (`depth=5` all groups `1083`; many single-group depth5 masks `1081`; no case beat `1081`).
- Wider depth sets regressed (`1089-1096` range).
- Decision: keep filters as experiment scaffolding but leave feature off by default.

- 2026-02-24: Deep-gate token reuse from existing scatter lane op.
- What changed: replaced explicit scalar deep-round token op with dependency on existing `*_scat_xor_7` op for depth>=6.
- Why: remove 60 token ops while preserving deep-round gating semantics.
- Result: correctness passed but regressed to `1082` cycles despite lower op count (`12365` ops), indicating token no longer represented full deep-tree completion strongly enough for load smoothing.
- Decision: reverted to explicit scalar token op path; restored `1081`.

- 2026-02-24: Group interleaving architecture attempt (`GROUP_EMIT_ORDER_MODE`).
- What changed: decoupled logical emit order from physical memory group index, with modes `identity`, `light_first`, `heavy_first`.
- Why: test fundamentally different group interleaving to reshape bottleneck overlap (especially depth3/4 scatter pressure).
- Result:
- `identity`: `1081` (pass, baseline).
- `light_first`: `1094` (pass).
- `heavy_first`: `1091` (pass).
- Decision: keep feature only as disabled scaffolding (`identity` default), treat tested reorderings as dead ends.

- 2026-02-24: Web-research guided architecture intake.
- What changed: reviewed public discussion hints (HN thread on this challenge) for sub-1000 strategy patterns.
- Why: identify fundamentally different approaches rather than knob tuning.
- Result: extracted concrete candidate families: stage-5/tree fusion, round-class specialized bundles, speculative preloading from earlier hash stages, and collision-aware/broadcast-oriented depth strategy.
- Decision: follow-up; use these as architecture hypotheses and test in code.

- 2026-02-24: Depth-3/4 leaf-lowering architecture (VALU<->FLOW rebalance).
- What changed: introduced `DEPTH_3_VSELECT_DIFF_PAIRS` and `DEPTH_4_VSELECT_DIFF_PAIRS` to control how many vselect leaves use `multiply_add` vs pure `flow/vselect`.
- Why: trade VALU pressure into FLOW headroom to reduce hard VALU lower bound while preserving correctness.
- Result:
- Broad sweep (`d3 pairs 0..4`, `d4 pairs 0..8`) found multiple correctness-valid points.
- New best: `DEPTH_3_VSELECT_DIFF_PAIRS=3`, `DEPTH_4_VSELECT_DIFF_PAIRS=2` -> `1080` cycles (pass), `12422 ops`.
- Slot totals at new best: `load=2017`, `alu=12108`, `valu=5981`, `flow=794`.
- Some low-pair settings caused scheduler-stuck invalid outputs (bogus ~`438-503` cycle reports).
- Decision: keep new default (`DEPTH_4_VSELECT_DIFF_PAIRS=2`), establishing `1080` as the new baseline.

- 2026-02-24: Selective stage-5 fusion windowing (round-aware skip set).
- What changed: added `STAGE5_FUSION_SKIP_ROUNDS` and rewired build logic so `biased_prev` and `include_stage5_const` are driven by per-round skip decisions rather than all non-final rounds.
- Why: preserve only the potentially beneficial subset of stage-5 fusion without paying full global-fusion cost.
- Result:
- Late-round skip test (`[10,11,12,13,14]`) was correctness-valid only at reduced `MAX_CONCURRENT_GROUPS` and regressed badly (`1256-1303`).
- At default group concurrency it often became scheduler-stuck/invalid (bogus ~`456-580` cycle outputs).
- Decision: keep framework for future targeted experiments, but keep `ENABLE_STAGE5_TREE_FUSION=False` in baseline.

- 2026-02-24: Constant materialization engine experiment.
- What changed: added `CONST_NONZERO_ENGINE` with `flow` mode (`add_imm` from zero) to shift scalar const generation from load to flow.
- Why: cut load slots toward the `load <= 2000` lower-bound requirement.
- Result: load dropped `2017 -> 2006`, flow increased `794 -> 805`, but cycle regressed to `1092`.
- Decision: keep feature flag as scaffolding (`CONST_NONZERO_ENGINE='load'` baseline).

- 2026-02-24: Depth-3 all-group vselect architecture test.
- What changed: added explicit depth mask overrides (`DEPTH_3_VSELECT_GROUPS_OVERRIDE`, `DEPTH_4_VSELECT_GROUPS_OVERRIDE`) and tested full depth-3 vselect activation (all groups) with varied windows/concurrency.
- Why: push deeper load reduction in line with slot-bound model.
- Result:
- Many configurations became scheduler-stuck at default concurrency.
- Correctness-valid configurations required reduced concurrency and regressed heavily (`~1288-1338` cycles best).
- Decision: dead end in current scheduler/allocation regime.

- 2026-02-24: Offload policy for scatter vector-xor combine.
- What changed: added `DISABLE_OFFLOAD_SCATTER_VEC_XOR` to keep `*_scat_xor_vec` on VALU when desired.
- Why: prevent proactive ALU split from canceling intended ALU relief of selective scatter-vector combine.
- Result:
- Confirmed expected static-op delta exists (`-144 ALU`, `+18 VALU` in op graph for one selective case).
- In scheduled bundles, dynamic offload rebalancing still erased most gains; best measured cycle remained regressive (`1088+`).
- Decision: keep knob as analysis/control scaffolding; not baseline-worthy yet.

- 2026-02-24: Round-specific depth-4 vselect overrides.
- What changed: added `DEPTH_3_VSELECT_GROUPS_BY_ROUND` / `DEPTH_4_VSELECT_GROUPS_BY_ROUND` and plumbed round-aware checks through future-use analysis + tree emission.
- Why: enable per-round structural shifts (e.g., round-4 vs round-15 asymmetry) without globally changing depth masks.
- Result:
- Several single-group round-specific flips were correctness-valid.
- Best remained a tie at `1080` (e.g., round-4 add `g=17`), while most variants regressed (`1082-1138`).
- Decision: keep round-specific override framework; no baseline change.

- 2026-02-24: Selective offload-family sweeps (tokenized blocklists).
- What changed: profiled offloaded op families and tested selective ALU-offload blocking across hash/tree op categories (`_hash5_combine`, `_hash1_b`, `tree_xor`, `branch_bit`, etc.).
- Why: reduce ALU hard lower bound without reintroducing load pressure.
- Result:
- Multiple policies reduced ALU materially (for example, blocking `_hash5_combine` gave `alu=11788`) but all correctness-valid runs stayed `>=1080`.
- Best remained a tie at `1080` (e.g., blocking `_hash1_b`).
- Decision: keep as dead-end for now; no default policy change.

- 2026-02-24: Deep-round window stochastic retune on 1080 baseline.
- What changed: randomized search over explicit round windows for rounds `5..10` with threshold fallback disabled.
- Why: verify if revised architecture changed the old local optimum of deep-round gating.
- Result: many maps tied at `1080` but none improved below it.
- Decision: keep current simple default behavior; no map override promoted.

- 2026-02-24: Exhaustive single-group depth-5 vselect re-evaluation.
- What changed: swept all depth-5 single-group placements across `DEPTH_5_VSELECT_DIFF_PAIRS` and bit-order modes under updated baseline.
- Why: re-check prior dead-end after new 1080 architecture changes.
- Result: best correctness-valid case regressed to `1086`; no placement beat baseline.
- Decision: keep depth-5 vselect disabled by default.

- 2026-02-24: Round-specific depth-3/depth-4 diff-pair architecture.
- What changed:
- Added `DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND` / `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND`.
- Extended setup/emission so depth-3/4 trees can use asymmetric per-round leaf-lowering while preserving correctness.
- Why: depth `3` and `4` appear in two distinct rounds each (`3/14` and `4/15`) with different schedule pressure; global diff-pair constants were too rigid.
- Result:
- Exhaustive sweep over `(d3_r3,d3_r14,d4_r4,d4_r15)` found a new best:
- `(4,3,3,1)` -> `1079` cycles (full submission tests pass), `12431 ops`.
- New slot totals: `load=2017`, `alu=12142`, `valu=6008`, `flow=770`.
- Decision: keep as new default (`DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND={3:4,14:3}`, `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND={4:3,15:1}`), establishing `1079` baseline.

- 2026-02-24: Full round-aware interleaving windowing framework (round end-gates).
- What changed:
- Added `ROUND_GATE_WINDOW_BY_ROUND` / `ROUND_GATE_WINDOW_BY_DEPTH`.
- Added `_round_gate_window()` and integrated round-start deps with round-end scalar tokens (`*_round_gate_token`) so selected rounds can cap in-flight groups independently of deep-tree-only gating.
- `_emit_index_update()` now returns emitted refs so round-tail token deps can include branch/index updates when present.
- Why: deep-only round gating was too narrow; this adds a true round-aware interleaving architecture to test round/depth-specific windows.
- Result: framework is baseline-neutral when disabled; `python3 tests/submission_tests.py` still passes at `1079` cycles (`12431 ops`, peak scratch `1499/1536`).
- Decision: keep as experimentation scaffolding and run targeted round-window searches.

- 2026-02-24: Depth-targeted round-window sweeps (full-round gates).
- What changed: swept `ROUND_GATE_WINDOW_BY_DEPTH` across depth-3/4/5 and multi-depth sets with varying windows.
- Why: test whether broad depth-class interleaving caps can improve bottleneck overlap.
- Result:
- Many aggressive windows were correctness-invalid/stuck (bogus sub-1000 reports).
- Correctness-valid cases were mostly regressive (`1080+`) and best tied baseline (`1079`) only when windows were effectively loose/no-op (`32`).
- Decision: treat broad depth-window gating as dead end for cycle reduction.

- 2026-02-24: Sparse round-map interleaving search with deep-gate interaction.
- What changed: sampled round-specific maps over rounds `5..10` under both `MAX_CONCURRENT_DEEP_ROUNDS=20` and `0`.
- Why: discover asymmetric interleaving windows impossible with depth-only gating.
- Result:
- With deep gating on (`20`): best remained `1079` (empty map).
- With deep gating off (`0`): found new best `1078` at `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}`.
- New-best stats: `12387 ops`, peak scratch `1456/1536`, slot totals `load=2017`, `alu=12170`, `valu=5999`, `flow=770`.
- Decision: promote sparse round-map gating + deep-gate disable as new baseline architecture.

- 2026-02-24: Local neighborhood sweeps around new `1078` round-map.
- What changed:
- Sampled and partial exhaustive searches around rounds `7/9/10` windows, plus targeted additions on rounds `5/6/8`.
- Compared alternate tied maps (e.g. `{7:24,9:20,10:26}`) and deep-window reintroductions.
- Why: verify whether `1078` is a local optimum and attempt `1077`.
- Result:
- Multiple `1078` ties, no correctness-valid `1077` found.
- Re-enabling deep gating (`>0`) with these maps regressed (`1080+`) or became invalid at aggressive settings.
- Decision: keep `MAX_CONCURRENT_DEEP_ROUNDS=0` and `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}` as default.

- 2026-02-24: Baseline promotion and full validation.
- What changed: set defaults to sparse round-map gating baseline (`MAX_CONCURRENT_DEEP_ROUNDS=0`, `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}`).
- Why: operationalize the new best configuration.
- Result: `python3 tests/submission_tests.py` full pass at `1078` cycles (`12387 ops`).
- Decision: keep as current baseline.

- 2026-02-24: `SCATTER_VECTOR_XOR` retest on 1078 baseline.
- What changed: re-ran selective scatter-vector combine variants with new round-map baseline, including offload suppression.
- Why: verify whether previous ALU-relief path becomes beneficial after round-aware interleaving win.
- Result:
- Best case remained regressive (`1080` at depth-5 only; baseline `1078` unchanged).
- Wider depth activation further regressed (`1081-1084`) despite lower ALU slot totals in some cases.
- Decision: keep `SCATTER_VECTOR_XOR=False`; still a dead end for cycle reduction.

- 2026-02-24: Stage-5 fusion retest on 1078 baseline.
- What changed: re-tested `ENABLE_STAGE5_TREE_FUSION=True` with several sparse late-round skip sets.
- Why: check if new interleaving architecture makes stage-5 fusion viable.
- Result:
- Correctness-valid variants regressed (`1087-1089`), and several skip sets were correctness-invalid.
- Decision: keep `ENABLE_STAGE5_TREE_FUSION=False`; treat as dead end under current architecture.

- 2026-02-24: Joint depth-3/4 diff-pair retune under new round-map baseline (partial exhaustive).
- What changed: began exhaustive sweep of `(d3_r3,d3_r14,d4_r4,d4_r15)` while fixing round-map baseline `{7:28,9:24,10:28}`.
- Why: test whether previous leaf-fusion optimum shifted after interleaving architecture change.
- Result:
- First 1000/2025 combos completed; best reached only `1080` (no improvement vs `1078` baseline).
- Search was stopped after this signal to prioritize larger-architecture paths.
- Decision: currently a likely dead end; revisit only if coupled with a new architecture that changes round pressure distribution.

- 2026-02-24: Single-group depth-5 vselect retest on 1078 baseline.
- What changed: swept all `DEPTH_5_VSELECT_GROUPS={g}` (`g=0..31`) with round-map baseline.
- Why: target the remaining `load` lower-bound gap using minimal depth-5 vselect substitution.
- Result:
- Most groups were correctness-invalid/stuck.
- Correctness-valid groups regressed (`best = 1087` at `g=12`) even though load dropped slightly (`2017 -> 2013` in best valid case).
- Decision: single-group depth-5 vselect remains a dead end.

- 2026-02-24: Multi-group depth-5 vselect (stable-group subset) retest.
- What changed: tested pair/triple combinations among the only single-group-valid set `{1,3,6,7,8,12,15,18,22,25,27}`.
- Why: see if small multi-group conversion can cross the `load <= 2000` threshold while preserving cycle wins.
- Result:
- Best correctness-valid case was heavily regressive (`1114` cycles at groups `(1,3)`), despite larger load drop (`load=2006`).
- Decision: reject this direction for current scheduler/dataflow model.

- 2026-02-24: Full depth-5 vselect with explicit round-5 windowing (new round-gate architecture).
- What changed: enabled `DEPTH_5_VSELECT_GROUPS=all` and swept `ROUND_GATE_WINDOW_BY_ROUND` (including round-5 caps), `DEPTH_5_VSELECT_DIFF_PAIRS`, and depth-5 bit-order.
- Why: test whether the new round-aware end-gate architecture can stabilize previously invalid full depth-5 conversion.
- Result:
- All tested configurations remained correctness-invalid/stuck (198/198 invalid in broad sweep), including aggressive round-5 throttling.
- Very low bogus cycle reports persisted.
- Decision: full depth-5 vselect remains an unstable dead end under current allocator/dependency model.

- 2026-02-24: Global scatter-vector combine + fusion retune.
- What changed: enabled `SCATTER_VECTOR_XOR` globally and retuned depth-3/4 diff-pair settings + scatter offload suppression.
- Why: aggressively cut ALU pressure from scalar scatter XOR path while rebalancing flow/VALU with leaf fusion.
- Result:
- ALU dropped substantially in best variant (`alu=11787`) but VALU rose (`valu=6074`) and best cycle stayed regressive (`1084`).
- Decision: keep `SCATTER_VECTOR_XOR=False` baseline; architecture is not winning in current form.

- 2026-02-24: Immediate group load-address generation architecture.
- What changed:
- Added `USE_IMMEDIATE_GROUP_LOAD_ADDR` to replace serial `group_offset` ALU chain with per-group `flow/add_imm` load-address generation.
- Fixed latent IR bug uncovered by this path: flow scalar writers now infer write-size correctly (`add_imm`/`select` -> scalar write).
- Why: remove setup/startup ALU chain depth and trade into non-bottleneck flow slots.
- Result:
- Correctness-valid but large regression when enabled (`1109` cycles, `12355 ops`).
- Decision: keep feature disabled by default (`USE_IMMEDIATE_GROUP_LOAD_ADDR=False`); keep flow write-size inference fix as correctness/IR hygiene.

- 2026-02-24: Group/path-selective stage-5 fusion architecture.
- What changed:
- Added `STAGE5_FUSION_SELECTIVE_BY_PATH`.
- Reworked fusion control to allow per-group stage-5 const skipping based on next-round scatter/vselect path (instead of only round-global skip sets).
- Why: retain stage-5 fusion wins where next round can absorb const cheaply, avoid scatter compensation overhead elsewhere.
- Result:
- Baseline-neutral with fusion disabled.
- With fusion enabled, selective path produced correctness-invalid outputs (example: `1084` cycles, fail); explicit late-round skip sets remained correctness-valid but regressive (`1089`).
- Decision: keep as disabled experimentation framework only (`ENABLE_STAGE5_TREE_FUSION=False` baseline).

- 2026-02-24: Tail store-address recomputation architecture.
- What changed: added `RECOMPUTE_STORE_ADDR_AT_TAIL` to recompute `vstore` address near group tail instead of keeping initial load address live through all rounds.
- Why: reduce long-lived scalar liveness pressure and allocator contention near scratch limit.
- Result: correctness-valid but regressive (`1081` cycles, `12419 ops`) when enabled.
- Decision: keep disabled by default.

- 2026-02-24: Late-round stage-5 fusion + round-window co-search.
- What changed: sampled round-window maps (`rounds 7..10`) under the only correctness-valid fusion schedule (`STAGE5_FUSION_SKIP_ROUNDS={10,11,12,13,14}`).
- Why: verify whether round-aware interleaving can rescue stage-5 fusion once fusion correctness is constrained.
- Result: best remained regressive (`1084` cycles at empty round-window map).
- Decision: treat current stage-5 fusion path as non-competitive with 1078 baseline.

- 2026-02-25: Interrupted pairwise additive mask search completed.
- What changed: completed exhaustive `(d3@r3 + one excluded group) x (d4@r4 + one excluded group)` (126 combos) around the `1065` baseline masks.
- Why: finish the interrupted structural neighborhood search and verify no missed win from two-step additive interactions.
- Result:
- All 126 combos correctness-valid.
- Best was regressive at `1069` cycles (`g3=11`, `g4=31`).
- Decision: keep existing masks unchanged.

- 2026-02-25: In-place dataflow + tree-bit-order architecture retest.
- What changed: swept `ENABLE_INPLACE_*` vector/tree/hash flags with `REVERSE_TREE_BIT_ORDER_DEPTH_{3,4,5}`.
- Why: check whether dependency-shape/dataflow rewrite can reduce critical path without mask tuning.
- Result:
- Only configurations with all in-place flags disabled remained valid.
- Reversing depth-3/depth-4 bit order regressed heavily (`1129+` / `1163+`); baseline remained best at `1065`.
- Decision: keep in-place vector/tree/hash disabled and bit-order defaults unchanged.

- 2026-02-25: Stage-5 selective fusion correctness root cause and generalized fix attempt.
- What changed:
- Diagnosed original selective-fusion invalidity: partial branch-bit bias history corrupted later scatter/vselect selection.
- Added index-bias tracking and scatter index unbiasing (`idx ^ reversed_mask`) plus a vselect de-bias fallback path under fusion.
- Added bit-mask safety checks to selective skip logic for next-round vselect compatibility.
- Why: recover the previously promising selective-fusion slot profile while preserving correctness.
- Result:
- Correctness was recovered for selective fusion (`ok=True`) but cycle regressed to `1124` with higher op/slot pressure (`12530 ops`).
- Structured/global skip-set sweeps under the corrected fusion framework remained regressive; best seen was `1070` (e.g. skip `{14}`).
- Decision: keep fusion framework as experimental scaffolding only; baseline remains `ENABLE_STAGE5_TREE_FUSION=False`.

- 2026-02-25: Stage-5 fusion schedule sweep (structured windows/prefix/suffix).
- What changed: evaluated curated global `STAGE5_FUSION_SKIP_ROUNDS` sets (prefixes, suffixes, contiguous windows, hand-picked mixes).
- Why: verify whether corrected fusion can beat baseline with a different round schedule.
- Result:
- Best correctness-valid configuration remained regressive (`1070` at skip `{14}`).
- No schedule beat `1065`.
- Decision: reject stage-5 fusion as a current optimization path.

- 2026-02-25: Scheduler offload-policy architecture sweep.
- What changed: swept `CRITICALITY_AWARE_OFFLOAD`, priority cutoff, and deferred-offload knobs around current baseline.
- Why: attempt a structural ALU/VALU rebalance without changing kernel math.
- Result:
- No improvement; best tied baseline (`1065`).
- Most variants regressed (`1069-1268`), with deferral especially harmful.
- Decision: keep offload policy defaults unchanged.

- 2026-02-25: Scheduler priority-weight heuristic sweep.
- What changed: tested broad and focused ranges of `CRITICAL_PATH_SCALE`, `EMIT_ORDER_SCALE`, late-flow cost/threshold, and base flow/load engine costs.
- Why: check if current 1065 is heuristic-limited rather than algorithm-limited.
- Result:
- Best tied baseline (`1065`) at/near current default regime.
- Many alternative weightings regressed materially (`1072-1116+`).
- Decision: keep current scheduler heuristic defaults.

- 2026-02-25: Depth-5 vselect sparse-path retest on 1065 baseline.
- What changed:
- Re-tested full depth-5 activation (`all groups`, varied diff-pairs/reverse order) and sparse single-group activations.
- Why: revisit load-bottleneck reduction after mask/round-window architecture improvements.
- Result:
- Full depth-5 remained invalid/stuck.
- Single-group valid points were all regressive; best `1072` (`group=27`, `diff_pairs=8`).
- Decision: depth-5 vselect remains a dead end under current allocator/scheduler model.

- 2026-02-25: Round-window neighborhood/random map retest at 1065 baseline.
- What changed: random and neighborhood sampling over `ROUND_GATE_WINDOW_BY_ROUND` for rounds `6..11`.
- Why: validate whether 1065 masks changed the round-gating optimum.
- Result:
- No map beat current default; repeated best stayed exactly `{7:28,9:24,10:28}` at `1065`.
- Decision: keep current round-window map unchanged.

- 2026-02-25: Constant-materialization engine path retest.
- What changed: evaluated `PREFER_FLOW_CONST` and `CONST_NONZERO_ENGINE=flow`, including co-tuning depth-3/4 diff-pairs.
- Why: reduce load-engine pressure (currently `2001`) to push below `2000` without deep-path changes.
- Result:
- `PREFER_FLOW_CONST=True` reduced load to `2000` but regressed (`1070` best local `1069` with diff retune).
- `CONST_NONZERO_ENGINE=flow` reduced load to `1990` but still regressed (`1070` best after diff retune).
- Decision: keep baseline constant strategy (`CONST_NONZERO_ENGINE=\"load\"`, `PREFER_FLOW_CONST=False`).

- 2026-02-25: Diff-pair architecture re-validation on new baseline.
- What changed:
- Exhaustive local sweeps for `(d3@r3, d4@r4)` and `(d3@r14, d4@r15)` under current 1065 setup.
- Why: ensure no hidden drift after broader architectural changes.
- Result:
- Baseline settings remained optimal:
- `DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND[3]=4`
- `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND[4]=3`
- `DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND[14]=3`
- `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND[15]=1`
- Decision: keep diff-pair defaults unchanged.

## Active Architectural Plan
- Round-aware interleaving windowing is implemented and enabled selectively:
- `DEEP_ROUND_DEPTH_THRESHOLD=6`
- `MAX_CONCURRENT_DEEP_ROUNDS=0`
- `ROUND_GATE_WINDOW_BY_ROUND={7:28,9:24,10:28}`
- Current validated baseline from submission tests is `1065` cycles (`12375 ops`).
- Follow-up: major slot-count reduction path is needed now (especially VALU/load), not additional gating micro-tuning.
- Priority focus: reduce deep scatter load footprint or reduce hash-stage VALU count with a mathematically valid fusion.

## Next Candidates After Round-Aware Interleaving
- Next High-Value Path (immediate): `Depth-5+ Scatter Pair-Fusion (load+ALU collapse)`.
- Why this is highest value now: current floor is still dominated by deep scatter rounds where we pay `8 scalar loads + 8 scalar XORs` per group/round; scheduler-only and mask-only tuning has plateaued.
- Core idea: replace scalar scatter XOR emission with a pair-fused scatter micro-kernel that computes two lanes together from shared address structure, so deep rounds consume fewer scalar load/alu slots without requiring full depth-5 vselect.
- Implementation direction: introduce a new deep-scatter path that emits lane-pair operations (`(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`) with shared address prep and reduced per-lane scalar XOR fanout, gated by depth/round so only heavy rounds (`5..10`) use it.
- Guardrails: keep exact math/branch semantics unchanged, keep stage-5 fusion disabled, and keep round-window map fixed while validating this path in isolation.
- Success criteria for keeping the path: correctness pass with slot movement in the right direction (`load < 2001` and `alu < 12130`) and cycle below `1065`.
- Dependency-aware ownership model for safe in-place vector writes (current allocator/dependency model was a blocker).
- Hash-stage algebraic reductions beyond existing stage2-3 fusion (primary remaining path for meaningful VALU cuts).
- Deep-depth hybrid selection that does not inflate flow critical path (current depth-5 vselect variants regressed).
- Rework VALU offload from runtime split heuristic into finer IR-level lane ops for better scheduler control.

- 2026-03-05: Current baseline validation on synced `main`.
- What changed: no code changes; ran `python3 tests/submission_tests.py` on the current checked-out tree after confirming local `main` matched `origin/main`.
- Why: measure the real current cycle count before further optimization work.
- Result: correctness pass; `1065` cycles, `12375 ops`, peak scratch `1528/1536`.
- Decision: baseline unchanged; keep current configuration.

- 2026-03-05: Current slot-budget audit and hot-path decomposition.
- What changed: no code changes; profiled the emitted bundles and op families from the current baseline kernel.
- Why: quantify exactly which structural regions still dominate cycle lower bounds before attempting more tuning.
- Result:
- Current final bundle slot totals are `load=2001`, `valu=6010`, `alu=12122`, `flow=785`, so the kernel is still simultaneously load-, VALU-, and ALU-bound.
- Tree scatter remains the largest reducible pool: `1952` scalar scatter loads and `1952` scalar scatter XOR ALU ops total.
- Deep scatter depths `5..10` account for `1536` of those loads and `1536` of those ALU XORs (32 groups across 6 rounds), while residual depth-3/4 scatter accounts for the remaining `416 + 416`.
- Hash remains the other major structural pressure source with `5632` hash-family ops across rounds, so materially lower cycles likely still require a real hash VALU reduction in addition to scatter relief.
- Decision: treat scheduler/gating retunes as exhausted for now; next winning paths should target deep-scatter structural reduction first, with hash-stage fusion as the parallel second candidate.

- 2026-03-05: Delayed depth-5 vselect setup architecture.
- What changed:
- Added `DELAY_DEPTH_5_SETUP` and moved depth-5 setup emission into a helper so depth-5 vectors/diffs can be materialized only when the first depth-5 round reaches them.
- Baseline remains neutral when the flag is off (`python3 tests/submission_tests.py` still passes at `1065`).
- Why: prior full depth-5 vselect path looked allocator/scratch-limited because all one-round-only setup refs were ready at cycle 0 and stayed live too early.
- Result:
- Full all-group depth-5 vselect remained correctness-invalid/stuck even with delayed setup.
- Single-group variants became broadly correctness-valid under delayed setup, but all were still regressive.
- Exhaustive single-group sweep best valid point was `1071` cycles at `DEPTH_5_VSELECT_GROUPS={30}`, `DEPTH_5_VSELECT_DIFF_PAIRS=12`, `REVERSE_TREE_BIT_ORDER_DEPTH_5=False`, `DELAY_DEPTH_5_SETUP=True`.
- Additional cross-tuning around the strongest group (`g=27`) still bottomed out at `1071-1072`.
- Decision: keep delayed setup only as disabled scaffolding (`DELAY_DEPTH_5_SETUP=False`); depth-5 vselect remains non-competitive on this branch.

- 2026-03-05: Late shallow setup split for depth-3/depth-4.
- What changed:
- Added `DELAY_LATE_D3D4_SETUP` plus a late-copy setup helper so rounds `14/15` can use duplicated depth-3/depth-4 setup refs emitted near first late use, allowing the early copy to die after round `4`.
- Baseline remains neutral when the flag is off (`1065` cycles, correctness passing).
- Why: current peak scratch is near the limit, and d3/d4 setup refs otherwise stay live across deep rounds `5..10` even though late shallow rounds are far away.
- Result:
- Enabling the split regressed materially instead of helping: `1103` cycles on the plain baseline, `1094` with round-window map removed, and `1090` in the best tested variant with `PREFER_FLOW_CONST=True`.
- Slot pressure also moved the wrong way (`load=2004`, `valu=6031`, `alu=12212` in the direct-on case).
- Decision: keep the feature disabled (`DELAY_LATE_D3D4_SETUP=False`); the added duplicate setup overhead outweighs any liveness benefit.

- 2026-03-05: Bounded mixed config search over surviving scaffolds.
- What changed: ran a 120-candidate bounded search combining nearby round-window maps, constant-materialization modes, selective scatter-vector XOR depths, updated depth-3/4 diff-pair maps, delayed depth-5 single-group variants, and the new late shallow setup split.
- Why: verify that no obvious cross-term among the remaining scaffolds beats baseline before committing to another larger architectural rewrite.
- Result:
- Best valid configuration remained the empty/default config at `1065` cycles (`12375 ops`, slots `load=2001`, `flow=785`, `alu=12122`, `valu=6010`, `store=32`).
- No candidate beat baseline; no new correctness-valid improvement surfaced from this neighborhood.
- Decision: treat current scaffold combinations as exhausted on this branch. Next required path is a genuinely new slot-reducing architecture, most likely a deep-scatter load reduction that does not explode flow/VALU, or a new algebraic hash reduction.

- 2026-03-05: Delayed depth-5 multi-group amortization search.
- What changed: searched delayed depth-5 vselect subset combinations (up to 8 candidate groups, sizes `2..6`) to see whether shared setup amortization can make the new depth-5 rewrite competitive.
- Why: single-group delayed depth-5 results were clearly setup-amortization-limited; multi-group subsets are the first place the rewrite could plausibly beat baseline.
- Result:
- Shared setup did improve the rewrite versus the earlier single-group numbers, but not enough.
- Best found subset was `DEPTH_5_VSELECT_GROUPS={22,30}`, `DEPTH_5_VSELECT_DIFF_PAIRS=16`, `DELAY_DEPTH_5_SETUP=True` with the baseline round-window map, reaching `1076` cycles (correctness passing).
- Best slot profile at that point: `load=1989`, `flow=817`, `alu=12174`, `valu=6068`, `store=32`.
- Decision: keep delayed depth-5 setup as disabled scaffolding only; the amortized rewrite still loses because load savings are overpaid by flow/VALU growth.

- 2026-03-05: Delayed depth-5 + low-VALU d3/d4 rebalance search.
- What changed: searched the new delayed depth-5 rewrite jointly with aggressive round-specific d3/d4 diff-pair reductions and loose round-window maps, focusing on configurations that can move `load` below `2000` and pull `valu` back down.
- Why: the multi-group delayed depth-5 rewrite proved load-saving but VALU/flow-heavy; low-fusion d3/d4 settings are the one existing mechanism that can materially reduce VALU without touching deep scatter semantics.
- Result:
- This family came closest to a new win but still did not beat baseline.
- Best seen configuration was:
- `DEPTH_5_VSELECT_GROUPS={27}`
- `DEPTH_5_VSELECT_DIFF_PAIRS=12`
- `DELAY_DEPTH_5_SETUP=True`
- `ROUND_GATE_WINDOW_BY_ROUND={}`
- `DEPTH_3_VSELECT_DIFF_PAIRS_BY_ROUND={3:4,14:3}`
- `DEPTH_4_VSELECT_DIFF_PAIRS_BY_ROUND={4:1,15:1}`
- Result at that point: `1067` cycles (correctness passing), `12425 ops`, slots `load=1997`, `flow=834`, `alu=12129`, `valu=6023`, `store=32`.
- A focused local search around this point did not improve further; best nearby follow-up seen was still regressive (`1075+`).
- Decision: keep the rewrite disabled by default. This path is now a near miss rather than a winner, but still not baseline-worthy.

- 2026-03-05: Hash temp-reuse rewrite (`REUSE_HASH_COMBINE_TEMPS`).
- What changed: added a new local hash rewrite that reuses freshly-created hash temp vectors (`ha` / `hb`) for the final combine output instead of allocating a third output vector, while leaving full input-buffer in-place overwrites disabled.
- Why: reduce transient vector allocation pressure in the hash hot path without invoking the much riskier full in-place hash dataflow model.
- Result:
- Baseline remains neutral when disabled.
- When enabled, the scheduler still deadlocks/halts incorrectly (`~419` cycles reported, incorrect output), so the current alias/ownership model still cannot support this reuse path safely.
- Decision: keep `REUSE_HASH_COMBINE_TEMPS=False`; treat it as another allocator-model blocker, not a promotable optimization.
