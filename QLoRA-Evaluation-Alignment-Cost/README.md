# Topic 01 - QLoRA, Evaluation, the Cost of Alignment, and DPO

One activity, four notebooks, run in order. You fine-tune a small language model, prove
whether the fine-tune was worth it, find out what it cost, and then use preference
optimization to remove an unwanted learned habit.

The task is deliberately small and visible: teach Microsoft's Phi-3 Mini 4K Instruct to
translate English into Yoda-speak. The task is a vehicle. The method is the point.

---

## Run Order

### 1. Fine-Tuning with QLoRA

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elaheJ/IT344-IllinoisStateUniversity-Techniques-and-Tools-in-Applied-Machine-Learning/blob/main/QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-DanielGodoyYoda-Chapter0.ipynb)
[Notebook](notebooks/344-FA26-DanielGodoyYoda-Chapter0.ipynb)

Load Phi-3 Mini in 4-bit NF4, attach LoRA adapters with `peft`, format the
`yoda_sentences` dataset, and train with `SFTTrainer` from `trl`. Ends with a trained
adapter you can push to the Hugging Face Hub.

**Adapted from Chapter 0 of Daniel Voigt Godoy's book. See [Credits](#credits).**
Cells marked **Attention 344** are the additions made for this course.

**You produce:** a LoRA adapter folder. Download it. Notebooks 2, 3, and 4 need it.

---

### 2. Evaluation - Base vs. Prompted vs. Fine-Tuned

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elaheJ/IT344-IllinoisStateUniversity-Techniques-and-Tools-in-Applied-Machine-Learning/blob/main/QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-QLoRA-Yoda-Evaluation.ipynb)
[Notebook](notebooks/344-FA26-QLoRA-Yoda-Evaluation.ipynb)

Three conditions on the same 20 held-out sentences:

1. Base Phi-3, no style instruction.
2. Base Phi-3 plus an explicit Yoda system prompt.
3. The fine-tuned adapter, with no system prompt.

Scored with BERTScore and chrF under deterministic generation, with a per-example win
count so you can see where the aggregate metrics hide disagreement.

**The question:** did fine-tuning beat a well-written prompt? Ends in a 200 to 300 word
written analysis.

**Needs:** your adapter from notebook 1, and `data/yoda_external_test.csv`.

---

### 3. The Alignment Tax

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elaheJ/IT344-IllinoisStateUniversity-Techniques-and-Tools-in-Applied-Machine-Learning/blob/main/QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-AlignmentTax-QLoRA-Yoda.ipynb)
[Notebook](notebooks/344-FA26-AlignmentTax-QLoRA-Yoda.ipynb)

The question the style metrics cannot ask: did the adapter make the model worse at
everything else? Measures two axes as the adapter scaling factor `alpha` sweeps from 0
to 1.

| Axis | Measured on | Metric |
| :-- | :-- | :-- |
| Style | `yoda_external_test.csv`, 20 held-out sentences | BERTScore F1, chrF |
| Capability | `phi3_capability_probe.csv` | category-level pass rate |

Also tests whether the adapter went prompt-deaf. This is the asymmetry that separates
fine-tuning from retrieval: RAG and CAG add context and leave the weights alone, so they
cannot damage the model. QLoRA edits the weights, so it can.

**The question:** where is the knee, the strongest style you can buy before capability
starts to fall? Ends in a 300 to 400 word written analysis.

**Needs:** your adapter from notebook 1, plus both CSV files in `data/`.

---

### 4. Preference Tuning with DPO

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elaheJ/IT344-IllinoisStateUniversity-Techniques-and-Tools-in-Applied-Machine-Learning/blob/main/QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-DPO-Yoda-Preferences.ipynb)
[Notebook](notebooks/344-FA26-DPO-Yoda-Preferences.ipynb)

The supervised adapter learned Yoda-like inversion and an unintended verbal tic from the
same demonstrations. This notebook turns the dataset's clean and interjection-bearing
targets into explicit `chosen` and `rejected` responses, then trains one copy of the adapter
against a frozen reference copy with Direct Preference Optimization (DPO).

The notebook treats training as an experiment rather than a victory condition. It:

1. Self-tests the interjection detector and capability scorer.
2. Reports unusable preference pairs and checks for held-out-data leakage.
3. Measures response-length and wording confounds before training.
4. Verifies that the policy and reference adapters are identical before DPO.
5. Evaluates held-out preference margins, interjection rate, Yoda-style similarity,
   capability-probe accuracy, and output length after training.

**The question:** can DPO remove the unwanted habit without erasing the desired style or
further reducing performance on the calibrated capability probe? A technically consistent
null result is still an interpretable result.

**Needs:** your adapter from notebook 1 and both CSV files in `data/`. The notebook pins a
tested `transformers`/`peft`/`trl` stack and takes roughly 10 to 20 minutes for its training
cell on a Colab T4, plus evaluation time.

---

## Data

| File | Rows | Purpose |
| :-- | :-- | :-- |
| [`data/yoda_external_test.csv`](data/yoda_external_test.csv) | 20 | Style test set, written outside the published 720-row training data so it is genuinely held out. |
| [`data/phi3_capability_probe.csv`](data/phi3_capability_probe.csv) | see file | General-capability probe covering factual recall, format compliance, and multi-part instructions. |

---

## Requirements

Colab with a **T4 GPU** runtime is enough for all four notebooks. Notebook 1 pins the
package versions used in the original book. Notebook 1 also asks for a Hugging Face token
if you choose to push your adapter to the Hub, which is optional.

> **Heads up on file paths.** The notebooks read and write under a Google Drive path such
> as `/content/drive/MyDrive/344/...`. Change those paths to your own Drive, or replace
> them with Colab's file upload widget. Each path sits in a clearly marked cell near the
> top of its notebook.

---

## Method References

- [Direct Preference Optimization: Your Language Model Is Secretly a Reward Model](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a85b405ed65c6477a4fe8302b5e06ce7-Abstract-Conference.html)
- [TRL 0.23.1 DPO Trainer documentation](https://huggingface.co/docs/trl/v0.23.1/en/dpo_trainer), matching the notebook's pinned, tested API

---

## Credits

Notebook 1 is adapted from **Chapter 0** of
[*A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face*](https://github.com/dvgodoy/FineTuningLLMs)
by **Daniel Voigt Godoy**, along with his
[`yoda_sentences`](https://huggingface.co/datasets/dvgodoy/yoda_sentences) dataset. The
original work is his. Check his [repository](https://github.com/dvgodoy/FineTuningLLMs)
for licensing before reusing it outside this course.

Notebooks 2, 3, and 4 were written for IT 344. Notebook 4 implements DPO with
Hugging Face TRL and uses the Apache-2.0 `yoda_sentences` dataset described above.
