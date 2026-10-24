/* "Check yourself" questions for every chapter and appendix. Rendered by buildPager() in common.js, answers hidden in
   <details>. Each answer uses this project's own numbers; the README section of the chapter has the details. */
"use strict";
const QUIZ = {
  "01-tokens.html": [
    ["How many numbers describe one token inside this model, and where do they first come from?", "2048 (the hidden size). The token ID selects one row of the embedding table (248,320 rows x 2048), and that row is the token's starting hidden state."],
    ["How does the last hidden state become a prediction?", "The lm_head multiplies it by a 2048 x 248,320 matrix, giving one score (logit) per vocabulary entry; a softmax turns the scores into probabilities that add up to 1."],
    ["Why is \" Australia\" (with a leading space) a different token from \"Australia\"?", "The tokenizer learned the space as part of common word pieces, so a word at the start of a text and the same word after a space get different IDs. Text is split into such pieces, not into whole words or letters."],
  ],
  "02-layers.html": [
    ["What is the residual stream?", "The running hidden state of each token. Every layer reads it and adds a correction to it; nothing is overwritten, so later layers see everything earlier layers wrote."],
    ["How could a 67 GiB model be measured on a 16 GB GPU with 32 GB of RAM?", "By streaming: the hidden states of all tokens stay in RAM, and one 1.7 GB layer at a time is loaded to the GPU, run on every token, and freed (moe_stream.py)."],
    ["Why does pruning damage pile up across layers?", "A pruned layer writes a slightly wrong correction into the residual stream; every later layer reads that wrong value and builds on it, so small errors in 40 layers compound."],
  ],
  "03-experts.html": [
    ["Of the 35 B parameters, how many does one token use, and why so few?", "About 3 B. Each layer has 256 experts but the router picks only 8 per token (plus the shared expert), so most expert weights sit idle for any given token."],
    ["What share of the parameters are routed experts?", "About 90% (32.2 B of 35 B): 40 layers x 256 experts x 3 matrices of 2048 x 512. That is why deleting experts is the way to shrink this model."],
    ["What is the shared expert, and was it pruned?", "An extra expert that every token uses, scaled by a learned gate. It was never pruned: it is small and used by everything."],
  ],
  "04-router.html": [
    ["Name the router's four steps.", "Score all 256 experts (a 256 x 2048 matrix times the hidden state), softmax the scores, keep the top 8, renormalize those 8 weights to add up to 1, then mix the 8 experts' outputs with those weights."],
    ["What does the load-balancing loss do, and what did it mean for pruning?", "It pushed training to spread tokens evenly over experts. So there were no dead experts to delete for free: the 128 most-used experts of a layer carried only 74% of the traffic."],
    ["What happens to the routing when an expert is deleted?", "Its router row is deleted too; the softmax and top-8 run over the experts that are left, so its tokens go to their next choices."],
  ],
  "05-pruning.html": [
    ["What physically changes in the files when you prune to K = 128?", "Each layer's fused expert tensors are sliced from 256 to 128 experts, the router matrix loses the matching rows, and config.json says num_experts: 128. Nothing else changes."],
    ["Why did K = 48 break the model when K = 128 did not?", "At K = 128 the kept experts still received 65% of the router's original choices; at K = 48 only 28%. Most tokens lost most of their preferred experts (perplexity 11.7 against 5.92)."],
    ["How many parameters does K = 128 leave, and is the model faster?", "About 19 B (35 GiB in BF16 → 11 GiB at Q4_K_M). Speed barely changes: a token still uses 8 experts, so about 3 B active parameters."],
  ],
  "06-reap.html": [
    ["What does REAP measure for an expert?", "The average, over the tokens routed to it, of router weight x the size of the expert's output (g·‖E(x)‖): how much it actually changes the result when it is used."],
    ["How can a rarely used expert rank near the top by REAP?", "If it dominates the output whenever it fires. The 'biology' expert 20·241 was 229th of 256 by frequency but 2nd by REAP."],
    ["What were the K = 128 perplexities for REAP, frequency and random selection?", "5.92, 6.47 and 7.93 (original 4.92). The criterion matters a lot."],
  ],
  "07-calibration.html": [
    ["Why did the run 3 code model answer no SQL question correctly?", "Its calibration text was Python only. Experts needed for SQL looked useless on that text and were deleted; the calibration decides which skills survive."],
    ["Runs 1 and 3 both kept 48 experts. What was different, and what changed?", "Run 1 calibrated on a general mix and broke; run 3 calibrated on code and became a usable code specialist (code perplexity +15%), at the cost of general text."],
    ["Why is chat data rendered with the model's chat template before calibrating?", "So the router sees the same special tokens (<|im_start|>, <think> …) that it sees in real use; experts that handle them must look important."],
  ],
  "08-perplexity.html": [
    ["What does a perplexity of 5 mean intuitively?", "On average the model is as uncertain as if it had to choose among 5 equally likely next tokens. It is e to the power of the average −ln p of the correct token."],
    ["Why measure it on held-out text?", "Text used for calibration (or training) would flatter the model; held-out text shows how it does on text it has not been tuned on."],
    ["What did perplexity miss in this project?", "The 19 B prune was only +20% perplexity and answered the test prompts, yet the benchmarks showed −16 points on HumanEval and −13 on MMLU."],
  ],
  "09-distillation.html": [
    ["Why did layer-wise healing make perplexity worse (5.92 → 5.96)?", "Matching each layer's hidden states (MSE) is not the same goal as predicting the next token; the layers were also asked to undo drift from earlier layers they could not undo."],
    ["What does the KL divergence to the teacher measure?", "How different the student's next-token distribution is from the teacher's, over the whole vocabulary, not just for the correct token."],
    ["Router distillation cut the KL by 18%. Why did the benchmarks not improve?", "The routers can only choose among the surviving experts; the knowledge in the 128 deleted experts per layer is gone, and 1 M tokens is far too little to teach it again."],
  ],
  "10-quantization.html": [
    ["What does a 4-bit block of 32 weights store?", "One scale (and sometimes an offset) for the block, and a 4-bit integer 0–15 per weight saying which of 16 levels it is rounded to."],
    ["What is the difference between Q4_K_M and IQ4_XS?", "Q4_K_M is a k-quant with uniform levels, block scales stored in a few bits, and some sensitive tensors at 6 bits (about 4.8 bits per weight); IQ4_XS uses 16 non-uniform levels, denser near zero (4.25 bits) and relies on the imatrix."],
    ["Why does quantizing make generation faster, not just smaller?", "Generating is limited by reading weights from memory; fewer bytes per weight means each token is read faster (chapter 15)."],
  ],
  "11-imatrix.html": [
    ["What does the importance matrix record?", "For every weight column, the mean of x² of its inputs over calibration text: which columns see large activations, so the quantizer spends its precision there."],
    ["Why should the imatrix text match the intended use?", "Columns that matter only for, say, code look unimportant on general text and get rounded harder. Run 3 computed its imatrix on code."],
    ["Why did IQ3_XXS fail on the 2 B model in step 6?", "Its MTP block never runs in a normal pass, so the imatrix had no data for it, and very low-bit types refuse to quantize a tensor without it. The fix stores that block at Q8_0."],
  ],
  "12-pipeline.html": [
    ["Put these in order: 04_prune, 02_calib_data, 07_gguf, 03_expert_stats, 05_eval.", "02 (calibration and eval text) → 03 (expert statistics) → 05 (virtual sweep to choose K) → 04 (write the pruned checkpoint) → 05 again (verify it) → 07 (GGUF, imatrix, quantize)."],
    ["Why was the original model deleted before converting to GGUF?", "Disk space: the pruned checkpoint (36 GiB) plus its BF16 GGUF (35 GiB) did not fit next to the 67 GiB original. It re-downloads in a few minutes."],
    ["Which steps need the original model on disk?", "Statistics, pruning, evaluation by streaming, distillation and ablation (03, 04, 05, 09, 10, 16, 17). GGUF work does not."],
  ],
  "13-attention.html": [
    ["How big is this model's KV cache for 100,000 tokens (f16)?", "About 2 GiB: 10 attention layers x 2 (K and V) x 2 KV heads x 256 x 2 bytes = 20 KiB per token."],
    ["Why does the Gated DeltaNet state not grow with the context?", "Each head keeps one fixed 128 x 128 matrix, updated in place for every token (delta rule with a forget gate): 63 MiB for all 30 layers at any length."],
    ["What did a q8_0 KV cache buy on the 16 GB card?", "It halves the cache (1.06 bytes per number), which made the full 262,144-token context fit: 13.8 GiB projected instead of 15.7."],
    ["What does grouped-query attention save?", "16 query heads share 2 key/value heads, so only 2 heads' keys and values are cached: 8x less KV memory than 16 KV heads."],
  ],
  "14-numbers.html": [
    ["Why is BF16 called 'FP32 with 16 bits cut off'?", "It keeps FP32's sign and 8 exponent bits (the same range) and only the top 7 of its 23 mantissa bits (about 2 significant digits)."],
    ["Where does FP16 run out?", "Above 65,504 (overflow to infinity) and below 6·10⁻⁵ (subnormals, then 0 under 6·10⁻⁸). 0.43% of one expert's weights were below its normal range."],
    ["Why did two identical runs give slightly different perplexities (bug 4)?", "GPU atomic additions summed the 8 expert outputs in a varying order, and in BF16 the rounding depends on the order. A fixed-order sum made runs bit-identical."],
  ],
  "15-speed.html": [
    ["Estimate the speed limit: 1.80 GiB read per token at 896 GB/s.", "896 / 1.93 GB ≈ 463 tokens per second. Measured: 212, 46% of the ceiling."],
    ["Why does a MoE generate fast for its size?", "A token reads only its 8 experts per layer: the experts are 86% of the 19 B file but 32% of what one token reads."],
    ["Why did 16 parallel requests give 2.7x, not 16x, the throughput?", "Each request picks its own experts, so the batch must read the union of their choices (about 82 of 128 experts per layer at 16 requests)."],
    ["When does speculative decoding help?", "When drafting is much cheaper than the model and the text is predictable: the 2 B's own MTP head made math and edits 45% faster, prose not at all; a 2 B draft model made the 19 B MoE slower."],
  ],
  "16-sampling.html": [
    ["What did temperature 1.5 do to \" Canberra\", and why?", "It fell from 69.6% to 19.2%. No wrong token became likely; the 248,000 unlikely tokens each gained a little and together took 77%."],
    ["Top-k = 3 versus top-p = 0.5 on 'The capital of Australia is': what is kept?", "Top-k keeps Canberra, a and located regardless; top-p keeps only Canberra, because its 69.6% already passes 50%."],
    ["How is thinking turned off?", "Not inside the network: the chat template pre-fills an empty <think></think>, so the model sees its reasoning is over and starts the answer."],
  ],
  "a-options.html": [
    ["What is the difference between quantizing and pruning?", "Quantization keeps every parameter and stores it in fewer bits; pruning removes parameters. A quantized 35 B is still a 35 B model."],
    ["Why delete experts rather than layers or matrix widths on one GPU?", "Cutting layers or widths of a dense model damages every token and needs billions of tokens of retraining; deleting experts only affects the tokens that used them, which re-route, and can be scored one layer at a time."],
    ["At the same 10.6 GiB, which won: the pruned 19 B at 4.9 bits or the original 35 B at 2.6 bits?", "The 35 B at 2.6 bits: average 91.4 against 82.7 over GSM8K, HumanEval and MMLU (step 4)."],
  ],
  "b-answers.html": [
    ["How did the 48-expert model (perplexity 11.7) fail in practice?", "Math still mostly worked, code was broken, and factual questions degenerated into repetition loops; the BF16 file did the same, so it was the pruning, not quantization."],
    ["How did the 128-expert model (perplexity 5.92) do on the same prompts?", "All three test prompts were answered correctly, with and without thinking, at about 190 tokens per second. The benchmarks later showed what three prompts could not."],
  ],
  "c-code.html": [
    ["Why did the first statistics pass report 58 GB allocated on a 16 GB GPU?", "Layers loaded with assign=True had requires_grad=True and there was no torch.no_grad(), so autograd kept every micro-batch's activations; the excess spilled silently into system RAM."],
    ["Why must the patched MoE forward hold only a weakref?", "A function stored on a module that refers back to the module is a reference cycle: each 1.7 GB layer stayed on the GPU until Python's cycle collector happened to run."],
    ["Why cap PyTorch at 85% of VRAM?", "On WSL/Windows, exceeding VRAM does not fail: the driver spills to system RAM and layers get up to 30x slower. A cap turns that into a clear error."],
  ],
  "d-specialist.html": [
    ["What did calibrating the 48-expert prune on code buy?", "A usable code model: code perplexity only +15%, HumanEval 87.2% (the 19 B general prune: 79.9%), 12 of 15 graded coding answers."],
    ["What did it cost?", "Everything outside code: MMLU 39% (close to guessing), GSM8K 74.8%, general text perplexity 16.4, no working SQL."],
  ],
  "e-benchmark.html": [
    ["Why report a 95% confidence interval with every score?", "With 164–300 questions a score is uncertain by several points (±6 on HumanEval); two models whose intervals overlap may well be equally good."],
    ["Why did the first MMLU parser score below chance?", "It graded the answer format, not the knowledge: the model's letter was often not where the parser looked. Scoring the next-token probability of each letter fixed it."],
    ["What did the benchmarks reveal that perplexity did not?", "The 19 B prune kept math (GSM8K 95.6 against 96.4) but lost 16 points on HumanEval and 13 on MMLU."],
  ],
  "f-training.html": [
    ["What is a gradient?", "For every trainable weight, which direction and how strongly to change it to lower the loss. The optimizer moves each weight a small step (learning rate) that way."],
    ["How can you backpropagate through a model that does not fit in memory?", "Run forward layer by layer, saving each layer's input in RAM; then go backwards, recomputing one layer at a time on the GPU and passing the gradient of the hidden state down."],
    ["Why did the gradient check need a nudge of about 1e-2?", "Smaller nudges vanished in BF16 rounding, and random per-weight nudges switched which experts were chosen; a 1e-2 step along the gradient gave ratios of 0.99 and 0.90."],
  ],
  "g-experts.html": [
    ["Why rank an expert's tokens by lift instead of by count?", "By count every expert's top tokens are 'the', ',' and ' of'. Lift divides by how common the token is overall, showing what is special about the expert."],
    ["Does a high-lift word list prove an expert is needed for that topic?", "No, it is correlation. Appendix I tested it: deleting one named expert changed its topic by under 1%; deleting a team of 4 topic experts per layer cost the topic 25–51%."],
  ],
  "h-quant.html": [
    ["Why is KL divergence a more sensitive measure than perplexity?", "Perplexity only looks at the probability of the correct token; KL compares the whole distribution. From Q8_0 to Q3_K_M the KL grew 9x while perplexity did not move."],
    ["Down to which type did the 19 B lose nothing measurable on the benchmarks?", "Q3_K_M (3.9 bits). MMLU dropped about 5 points at IQ3_XXS; at 2 bits everything collapsed."],
    ["Why was Q6_K (14.2 GiB) so much slower than Q5_K_M (12.4 GiB): 108 against 186 tokens per second?", "It did not fit in VRAM with its context; the weights that spilled into system RAM came over PCIe, about 28x slower per byte."],
  ],
  "i-ablation.html": [
    ["Why delete whole teams of experts and not just one?", "One expert has 255 neighbours to re-route to, so its effect is tiny (under 1%); deleting the 4 most topic-specific experts per layer cost the topic 25–51% while random teams cost under 1.2%."],
    ["What predicted the damage of deleting one expert: its lift or its share?", "Its share of the topic's tokens. The 'algebra expert' carried 0.8% of algebra and did nothing when deleted; a geometry expert carrying 47% cost geometry about 1%."],
    ["Why did merging dropped experts lose to deleting them?", "The experts are nearly unrelated (output cosine about 0.02), so an average is close to neither; a few survivors absorbed up to 37 experts each and were diluted. Aligning neurons fixed their order, not their difference."],
  ],
  "j-practical.html": [
    ["What does --n-cpu-moe 12 do, and why does it work well for a MoE?", "It keeps the experts of the first 12 layers in system RAM, computed by the CPU. A token reads only 8 of 256 experts per layer, so the CPU's share is small: the 20.5 GiB 35 B ran at 78 tokens per second."],
    ["The 35 B at Q4_K_M offloaded, or at UD-IQ3_XXS fully on the GPU: which would you run on 16 GB?", "The UD-IQ3_XXS: 173 tokens per second against 78, with nearly the same benchmark average (92.2 against 92.9)."],
    ["What should a model card's YAML header contain?", "The licence (the base model's: Apache 2.0 here), base_model (links the repo into the original's model tree), tags, and library_name: gguf."],
  ],
};
