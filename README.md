# IT 344 - Techniques and Tools in Applied Machine Learning

**Illinois State University · School of Information Technology · Fall 2026**

Course codebase for IT 344. Each class activity lives in its own topic folder with the
notebooks, the data files it needs, and a short topic README. Notebooks are written for
Google Colab on a T4 GPU runtime.

---

## Class Activities

| # | Topic | Description | Notebooks |
| :-- | :-- | :-- | :-- |
| 01 | [QLoRA, Evaluation, and the Cost of Alignment](QLoRA-Evaluation-Alignment-Cost/) | Fine-tune Phi-3 Mini on a style-transfer task with QLoRA, measure whether the fine-tune actually beat prompting, then measure what the fine-tune cost the model everywhere else. Runs end to end on a free Colab T4. | [1. Fine-Tuning](QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-DanielGodoyYoda-Chapter0.ipynb) · [2. Evaluation](QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-QLoRA-Yoda-Evaluation.ipynb) · [3. Alignment Tax](QLoRA-Evaluation-Alignment-Cost/notebooks/344-FA26-AlignmentTax-QLoRA-Yoda.ipynb) |

<!--
To add the next activity, copy the row above, increment the number, and point the links
at the new topic folder. Keep one topic per folder.
-->

---

## How to Run a Notebook

1. Open the notebook link in the table above. Every notebook has an **Open in Colab** badge in its topic README.
2. In Colab, set **Runtime → Change runtime type → T4 GPU**.
3. Upload the CSV files from the topic's `data/` folder when the notebook asks for them.
4. Run the cells in order. Cells marked **Attention 344** are the ones edited for this course.

Notebooks are committed **with their outputs intact** so you can see the expected results
before running anything. Your own numbers will differ, since generation and training are
not fully deterministic across runs.

---

## Repository Layout

```
QLoRA-Evaluation-Alignment-Cost/
├── README.md          topic overview, learning goals, run order
├── notebooks/         the three Colab notebooks, in run order
└── data/              held-out test set and capability probe
```

---

## Credits

The fine-tuning notebook is adapted from **Chapter 0** of
[*A Hands-On Guide to Fine-Tuning LLMs with PyTorch and Hugging Face*](https://github.com/dvgodoy/FineTuningLLMs)
by **Daniel Voigt Godoy**. The [`yoda_sentences`](https://huggingface.co/datasets/dvgodoy/yoda_sentences)
dataset is his as well. Full credit for the original material goes to him. Please consult
the [original repository](https://github.com/dvgodoy/FineTuningLLMs) for its license before
reusing that notebook outside this course.

The evaluation and alignment-tax notebooks were written for IT 344 and build on top of the
adapter produced by his notebook.

Base model: [`microsoft/Phi-3-mini-4k-instruct`](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct).

---

## Contact

Course materials maintained by the IT 344 teaching team. Open an issue for a broken link
or a notebook that fails to run.
