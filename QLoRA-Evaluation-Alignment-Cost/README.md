# Topic 01 - QLoRA, Evaluation, and the Cost of Alignment

One activity, three notebooks, run in order. You fine-tune a small language model, you
prove whether the fine-tune was worth it, and then you find out what it cost.

The task is deliberately small and visible: teach Microsoft's Phi-3 Mini 4K Instruct to
translate English into Yoda-speak. The task is a vehicle. The method is the point.

---

## Run Order

### 1. Fine-Tuning with QLoRA

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elaheJ/IT344-IllinoisStateUniversity-TechniquesAndToolsInAppliedMachineLearning/blob/main/QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-DanielGodoyYoda-Chapter0.ipynb)
[Notebook](notebooks/344-FA26-DanielGodoyYoda-Chapter0.ipynb)

Load Phi-3 Mini in 4-bit NF4, attach LoRA adapters with `peft`, format the
`yoda_sentences` dataset, and train with `SFTTrainer` from `trl`. Ends with a trained
adapter you can push to the Hugging Face Hub.

**Adapted from Chapter 0 of Daniel Voigt Godoy's book. See [Credits](#credits).**
Cells marked **Attention 344** are the additions made for this course.

**You produce:** a LoRA adapter folder. Download it. Notebooks 2 and 3 both need it.

---

### 2. Evaluation - Base vs. Prompted vs. Fine-Tuned

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elaheJ/IT344-IllinoisStateUniversity-TechniquesAndToolsInAppliedMachineLearning/blob/main/QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-QLoRA-Yoda-Evaluation.ipynb)
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

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elaheJ/IT344-IllinoisStateUniversity-TechniquesAndToolsInAppliedMachineLearning/blob/main/QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-AlignmentTax-QLoRA-Yoda.ipynb)
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

## Data

| File | Rows | Purpose |
| :-- | :-- | :-- |
| [`data/yoda_external_test.csv`](data/yoda_external_test.csv) | 20 | Style test set, written outside the published 720-row training data so it is genuinely held out. |
| [`data/phi3_capability_probe.csv`](data/phi3_capability_probe.csv) | see file | General-capability probe covering factual recall, format compliance, and multi-part instructions. |

---

## Requirements

Colab with a **T4 GPU** runtime is enough for all three notebooks. Notebook 1 pins the
package versions used in the original book. Notebook 1 also asks for a Hugging Face token
if you choose to push your adapter to the Hub, which is optional.

> **Heads up on file paths.** The notebooks read and write under a Google Drive path such
> as `/content/drive/MyDrive/344/...`. Change those paths to your own Drive, or replace
> them with Colab's file upload widget. Each path sits in a clearly marked cell near the
> top of its notebook.

---

## Credits

Notebook 1 is adapted from **Chapter 0** of
[*A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face*](https://github.com/dvgodoy/FineTuningLLMs)
by **Daniel Voigt Godoy**, along with his
[`yoda_sentences`](https://huggingface.co/datasets/dvgodoy/yoda_sentences) dataset. The
original work is his. Check his [repository](https://github.com/dvgodoy/FineTuningLLMs)
for licensing before reusing it outside this course.

Notebooks 2 and 3 were written for IT 344.
