# Handoff: MoE expert pruning of Qwen3.5-35B-A3B

Read learn/README.html first: it has the full pipeline, exact commands, and results table.
(README.md is only a short GitHub landing page; the HTML page is the source and is edited directly.)
This file is the working context for continuing in a new session.

## Goal
The user wants to **learn the process** of shrinking an LLM by pruning / distilling /
removing experts, and keep every command so they can repeat it. Original wish was
"~9B, Q4/IQ4". Started from Qwen3.8-27B (dense) but switched to the MoE
`Qwen/Qwen3.5-35B-A3B` because expert removal is the most tractable on one GPU.
There is no official Qwen3.8 9B; `Qwen/Qwen3.5-9B` exists as a quality reference.

## Machine
RTX 5070 Ti 16 GB (sm_120, Blackwell), 32 GB RAM, WSL2 Ubuntu 24.04, shell fish
(scripts are bash). Windows driver: exceeding VRAM spills to system RAM silently
(sysmem fallback) instead of OOM. F: drive has ~946 GB free; the WSL disk was nearly full.

## Status (2026-10-22, run 2 = keep 128, the current artifacts)
- Run 1 artifacts were deleted (results kept in results/run1-k48/). Teacher model
  models/Qwen3.5-35B-A3B DELETED again on 2026-10-23 (with gguf/q35-distill-k128 and the 2B duplicate) at the
  user's request; scripts/01_download.sh brings it back in ~7 min. WSL disk: 96 GB free after that clean-up.
- Run 2: REAP keep 128 -> models/q35-reap-k128 (18.99B, 36 GiB, unhealed), ppl 5.92
  (orig 4.92). gguf/q35-reap-k128/: Q4_K_M 11G, IQ4_XS 9.5G, imatrix.gguf (BF16 GGUF deleted;
  07_gguf.sh re-converts it from models/q35-reap-k128 and skips the imatrix if present). Quants are
  within noise of BF16 on wikitext (8.52/8.51 vs 8.55). Generation test (scripts/ask.py):
  all correct, thinking works, ~190 tok/s fully on GPU. Router healing again slightly
  worse (5.96) and was undone.

## Run 3 (2026-10-22): code specialist, 48 experts
02_calib_data.py --preset code -> data/calib_code.pt + eval_code.pt; stats/code.pt; models/q35-code-k48
(8.92 B, 17 GiB); gguf/q35-code-k48 Q4_K_M 4.9G / IQ4_XS 4.5G / BF16 16G (imatrix on code text via
CALIB=data/calib_code.pt). Code ppl 2.80 -> 3.23 (+15%) vs general calibration 4.73. ask.py --set code
--check (3 runs x 5 graded prompts): code-k48 12/15, general k128 10/15; both 0/3 on SQL. General text
much worse (eval.pt all 16.4; "capital of Australia is A" answer). Teacher deleted again;
models/q35-reap-k128 deleted (rebuild with 04_prune.py from stats/calib.pt). Results in
results/eval_code.jsonl, results/code-test-*.txt, results/general-test-code-k48.txt.
## Step 2 (2026-10-22): benchmark, DONE
scripts/08_bench.py (GSM8K 250 / HumanEval 164 / MMLU 300, greedy, thinking off; MMLU scored by next-token
letter probability after "The correct option is **"), run_bench.sh (llama-server --fit on -np 4),
bench_report.py (--js writes learn/e-benchmark-data.js), download_baseline.sh (unsloth Q4_K_M of the
unpruned model, gguf/original-unsloth, 20.5 GiB). Graders validated on reference answers (250/164/300).
Results % (95% CI): original 96.4/95.7/86.7; general k128 95.6/79.9/73.7; code k48 74.8/87.2/39.0.
Raw answers in results/bench/. First MMLU parser scored below chance (format, not knowledge) - fixed.
## Step 3 (2026-10-22): router distillation, DONE
02_calib_data.py --preset train -> data/train.pt (1M tok); 09_teacher_logits.py -> data/teacher_top32.pt
(top-32 covers 96.7%); 10_distill.py: student = teacher pruned on the fly (k128 REAP), only routers +
shared-expert gates train, loss = CE to teacher top-32, streamed forward/backward (one layer on GPU,
inputs saved in RAM), AdamW 1e-4, 62 steps x 32k tok, 144 min. --gradcheck: ratios 0.99 / 0.90 (eps must be
~1e-2 along the gradient; 1e-3 vanishes in bf16, random 2e-3 per weight switches experts).
Val KL 0.1047 -> 0.0859 (-18%, but val set was code-only - fixed to spread). ppl 5.916 -> 5.879;
bench distill-k128 94.8/81.7/73.3 vs 95.6/79.9/73.7: within noise. 11_patch_gguf.py wrote
gguf/q35-distill-k128 (routers are F32 in GGUF). Results: results/distill/, results/bench/distill-k128-*.
Ideas 4 queued/run via logs/idea4.sh: 12_expert_tokens.py (appendix G data), 13_layer_sensitivity.py
(per-layer K; non-uniform K cannot be saved in HF/GGUF, eval only). DONE: expert explorer = appendix G
(learn/g-experts.html, data 4.4 MB; lift ranking, examples of top tokens). Sensitivity (prune one layer to 32,
32 train seqs, noise ~0.3%): late layers most sensitive (39: +4.1%, 33-38: +1.4-2.1%), layer 1 +1.6%.
Allocation by sqrt(sensitivity) did NOT help: avg48 11.717 vs 11.715 uniform, avg128 6.133 vs 5.916 (worse):
noise treated as signal, extrapolated 32->16, layer 39 capped at 256. Results: results/layer_sensitivity.json,
results/alloc_k*.json, results/eval.jsonl tags alloc-k48/alloc-k128.
## Step 4 (2026-10-22): quantization curve, DONE
14_quant_sweep.sh (BF16 -> each type: quantize with gguf/q35-reap-k128/imatrix.gguf, measure, delete), 15_measure_gguf.sh
(wikitext ppl 64x2048, KLD vs BF16 16x2048 via gguf/quant-sweep/q128-base.kld, speed_test.py, run_bench.sh),
quant_report.py --js -> learn/h-quant-data.js (appendix H). Rows: results/quant/sweep.jsonl, results/speed.jsonl.
19 B: KLD Q8_0 .008 / Q4_K_M .028 / Q3_K_M .074 / IQ3_XXS .104 / IQ2_XXS .334; ppl flat (8.51-8.60) down to Q3_K_M;
benchmarks within noise down to Q3_K_M, MMLU -5 at IQ3_XXS, IQ2 collapses. Speed ~200 tok/s for any file <= 12.4 GiB,
Q6_K 14.2 GiB 108, Q8_0 54, BF16 20 (RESERVE 2.5 GiB in h-quant.html). Original 35 B (unsloth UD): UD-IQ2_M 10.6 GiB
avg 91.4, UD-IQ3_XXS 12.2 GiB avg 92.2 vs pruned 19 B Q4_K_M 10.6 GiB avg 82.7 -> quantizing beats pruning at equal size.
Same Q4_K_M re-run: HumanEval 78.7 vs 79.9 in step 2 (-np 4 batching is not deterministic). Sweep files (BF16 35 GiB + base.kld 7.6 GiB) now in
/mnt/f/qwen-scratch/quant-sweep (see step 5); pruned HF checkpoint at /mnt/f/qwen-scratch/q35-reap-k128;
gguf/original-unsloth emptied. README.html sections #step-4-results, #quant-findings, #quant-choose.

## Step 5 (2026-10-22): ablation + merging, DONE (run_step5.sh, ~70 min; appendix I learn/i-ablation.html)
Teacher re-downloaded (models/Qwen3.5-35B-A3B, 67 GiB); gguf/quant-sweep MOVED to /mnt/f/qwen-scratch/quant-sweep.
moe_forward is now DETERMINISTIC (fixed-order sum instead of index_add_ atomics: repeat runs differed by up to 0.15% per
domain before); references shifted: original 4.915->4.913, k128 5.916->5.919, k48 11.72->11.68 (tags step5-*).
02_calib_data.py --preset domains -> data/domains.pt (8 x 16 seqs: camel-ai bio/chem/phys, MATH algebra/geometry/
number_theory, py code, web). 16_ablate.py: single experts (results/ablation.json, ablation_tokens.pt) - every one
<1% on any domain; damage follows share of the domain's tokens, not lift (algebra expert 5.39 carries 0.8% of algebra);
35.227 (random control) carries 47% of geometry -> +0.98%, all 16 seqs worse; "biology" 20.241 is chemistry on this data.
--teams 4 16 (results/ablation_teams.json, chosen on half A, measured on half B): x4 own domain +25..51%, random <1.2%;
maths domains share experts; web is not a topic. 17_expert_similarity.py -> stats/similarity.pt (output cos ~0.02).
05_eval.py/04_prune.py --merge naive|aligned (+ 05 only: aligned-remap), merge_layer_state/align_neurons in moe_stream
(Hungarian on gate/up/down cosines with sign flips; scipy added). K=128: prune 5.92, naive 7.89, aligned 7.51, router-sim
7.31, full-router remap 7.67; K=48: 11.68 / 38.7 / 33.1 / 27.3 / 37.9. Hubs: groups up to 37 (K=128), survivors' own
weight median 53%. No merged checkpoint/benchmark built (too far behind). ablation_report.py --js -> i-ablation-data.js.

## Theory chapters 13-16 + step 6 practical (2026-10-23), DONE
README concepts 13-16 (attention/DeltaNet/KV cache, number formats, speed, sampling) + pages 13..16-*.html, data
learn/theory-data.js (scripts/theory_report.py from kv_memory.sh, batch_speed.py, number_formats.py, sampling probs via
llama-server n_probs). KV cache exact: 20 KiB/token (10 attn layers x 2 x 2 heads x 256 x 2B); DeltaNet state 63 MiB fixed.
Decode reads 1.80 GiB/token (19B Q4_K_M) -> ceiling 463 tok/s, measured 212. Batch 16: 488 tok/s total.
Step 6 (README #step-6-practical, appendix j-practical.html, data j-practical-data.js from practical_report.py):
make_gguf.sh (any HF repo -> BF16 GGUF + mmproj -> imatrix on wikitext -> quantize -> perplexity) tested on Qwen3.5-2B
(BF16 10.64, Q8_0 10.65, Q4_K_M 11.04, IQ4_XS 11.21, IQ3_XXS 12.48); MTP block needs --tensor-type "blk\.N\.=q8_0" for
IQ1-3 (no imatrix data). offload_sweep.sh: 35B Q4_K_M (on /mnt/f) --n-cpu-moe 12 -> 78 tok/s, 30 -> 43, --cpu-moe -> 33,
--fit on -> 66. llama-server here: --load-mode none (no --no-mmap), --reasoning on|off. model_card.py + upload_gguf.sh
(private by default, asks; NOT run: needs the user's hf login). 2B files moved to /mnt/f/qwen-scratch/step6/.
WSL disk ~18 GB free; move/resize instructions in README #move-wsl (user has not done it).

## Step 7 (2026-10-23): speculative decoding, DONE (README #step-7-speculative, chapter 15 steps 7-8)
scripts/spec_sweep.sh -> spec_test.py (4 prompts, greedy, 256 tok) -> results/spec.jsonl; llama-bench -p 1..16 ->
results/verify_cost.jsonl; results/spec_divergence.json. Only our 2B GGUF has MTP (unsloth 35B and our 19B don't).
2B: MTP n=3 math/edit 338->490 (+45%, 91-95% kept), prose never faster (42% kept). 19B + 2B draft-simple: 2-5x SLOWER
(draft 5 ms/token in the loop); ngram-simple 0-17% slower. Check 4 tokens: dense +11%, MoE +39%. MTP prose/code outputs
differ from plain at near-ties (" principle" 48.3/47.5 -> 47.0/48.7): batched-pass rounding. The 2B GGUF lives on
/mnt/f/qwen-scratch/step6 (copy to gguf/ before re-running spec_sweep.sh).

## Learning aids (2026-10-23)
learn/start.html: learning path (6 tracks, ticks in localStorage) + cheat sheet of key numbers; linked from index.html
and README.html. learn/quiz.js: 3-4 "Check yourself" questions per chapter/appendix (all 26), rendered above the pager by
buildQuiz() in common.js; every chapter page loads quiz.js before common.js. New page -> add a QUIZ entry and the script tag.
Public repo github.com/filip-lebiecki/qwen-moe-pruning, GitHub Pages serves learn/ (push to main redeploys); buildPager adds a
"This page online" line from SITE_URL/REPO_URL in common.js to all 26 chapter pages.

## Run 1 history (keep 48; its model/GGUF files were deleted, results in results/run1-k48/)
Done and verified:
- 00 setup: CUDA 13.0 toolkit (apt, wsl-ubuntu repo), uv venv (torch 2.14 cu130,
  transformers 5.20, flash-linear-attention), llama.cpp built with CUDA sm_120.
- 02 calibration data: data/calib.pt (512x1024: web, chat, think-math, think-code),
  data/eval.pt (128x1024: web, chat, think, wikitext).
- 03 stats: stats/calib.pt (freq, gate, REAP, EAN, prob sums per layer x expert, per-source freq).
- 05 sweep: results/eval.jsonl. Baseline ppl 4.92; REAP keep128 5.91, keep64 9.72,
  keep48 11.7; freq48 14.6; random48 20.3.
- 04 prune: models/q35-reap-k48 (8.92B params, 17 GB, unhealed - the best k48 variant).
- 06 heal (router-only, layer-wise MSE): made ppl WORSE (target=layer 15.8, target=moe 12.3).
- 07 GGUF: gguf/q35-reap-k48/{BF16, Q4_K_M 4.9G, IQ4_XS 4.5G, imatrix}. Runs ~180 tok/s.
  Math answers OK; code broken; factual Q degenerates into repetition loops.
  BF16 equally broken -> damage is from pruning 81% of experts, not quantization.

## Next steps
All of the originally offered steps are done (run 2, distillation = step 3, specialist = run 3, benchmark = step 2,
quantization curve = step 4, ablation/merging = step 5). Ask the user what to learn next.

## learn/README.html
The project notes (formerly README.md, converted once to HTML). Edit it directly; it uses
learn/common.css plus its own <style> block, and links to the chapter pages next to it.

## Learning pages
learn/index.html + 01..16-*.html: one animated page per README concept section (12 = whole
pipeline + glossary; 13-16 = attention/KV cache, number formats, speed, sampling, added 2026-10-23 with
data from scripts/kv_memory.sh, batch_speed.py, number_formats.py -> theory_report.py -> learn/theory-data.js), plus appendices a-options.html (ways to shrink), b-answers.html (real
outputs 48 vs 128 experts, copied verbatim from the test runs), c-code.html (code tour + the
four bugs with logged numbers), d..j = run 3, steps 2-6 and the expert explorer. CHAPTERS order in common.js
matters: 12-pipeline.html's glossary links chapters BY POSITION (shift its numbers if pages are inserted).
SVG text color: pass fill via S(...) (common.js turns it into style.fill; a plain fill attribute loses to common.css). Shared learn/common.css (color tokens with fixed meanings: blue hidden
state, amber active, teal kept, red pruned, violet router) and learn/common.js (Story step
player, SVG helpers, seeded rng). Plain classic scripts so they work from file://. Real
numbers come from results/ and from llama-server queries of the K=128 Q4_K_M; synthetic demo
data is labelled as such on each page.

## Code layout / conventions
- scripts/moe_stream.py: the shared engine. Checkpoint (lazy safetensors), Streamer
  (hidden states of all tokens on CPU pinned, one layer on GPU, next layer prefetched),
  moe_forward (re-implementation of the HF MoE block + stats), patch_moe,
  select_experts / expert_scores (criteria), prune_layer_state.
- 04_prune.py writes ONE SHARD PER LAYER; 06_heal.py relies on that to rewrite a layer.
- 06_heal.py modifies the student IN PLACE; re-run 04_prune.py to get an unhealed copy.
- Long jobs: launch with `setsid nohup ... > logs/x.log 2>&1 < /dev/null &` so they
  survive the session ending (a stats pass died once when the session ended).
- Run Python with `.venv/bin/python -I` from the repo root (paths are relative).
- Converter: `convert_hf_to_gguf.py --no-mtp` is required (MTP head dropped, config
  mtp_num_hidden_layers=0).

## Gotchas already hit (do not repeat)
- Layers built with load_state_dict(assign=True) have requires_grad=True -> always
  no_grad for inference (build_layer calls requires_grad_(False)).
- `mlp.forward = lambda x, m=mlp: ...` is a reference cycle -> 1.7 GB per layer
  leaked on GPU until GC; use patch_moe (weakref).
- moe_stream caps VRAM at 85% (VRAM_FRACTION env) + expandable_segments so overflow
  fails loudly instead of silently spilling to system RAM.
- `pkill -f <pattern>` kills the calling shell when the pattern is in its own command
  line; use `pkill -x llama-server` or a PID.
- python stdout piped through grep is block-buffered: use print(..., flush=True) and
  `grep --line-buffered`.
- Qwen3.5 RMSNorm is zero-centered (weight + 1); llama.cpp converter handles it.

## After moving the project
.venv and llama.cpp/build contain absolute paths -> recreate with scripts/00_setup.sh
(skips the CUDA toolkit if already installed). Re-download the teacher with
scripts/01_download.sh (took ~2 min) instead of copying 67 GiB. Don't run the
pipeline from /mnt/f (drvfs is slow for 67 GiB streaming passes); prefer moving the
WSL distro to F: (PowerShell: `wsl --shutdown` then
`wsl --manage Ubuntu-20.04 --move F:\WSL\Ubuntu` - the distro is *registered* as
Ubuntu-20.04 although the OS inside was upgraded to 24.04) or
a native ext4 location.
