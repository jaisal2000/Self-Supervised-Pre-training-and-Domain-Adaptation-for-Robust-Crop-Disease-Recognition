# Bridging the Lab-to-Field Gap: Self-Supervised Pre-training and Domain Adaptation for Robust Crop Disease Recognition

---

## Abstract

We investigate whether self-supervised pre-training on unlabeled field imagery improves crop disease recognition under lab-to-field domain shift, comparing it against ImageNet and CLIP pre-training. Early results show small-scale in-domain SimCLR underperforms generic large-scale features, motivating continued pre-training on top of foundation models instead of training from scratch on limited field data.

---

## 1. Introduction

Expert annotation of crop disease imagery is costly and slow, while most labeled datasets (e.g., PlantVillage) are collected under clean lab conditions that differ sharply from real field imagery (e.g., PlantDoc). This project asks: **can self-supervised pre-training on unlabeled field imagery close this lab-to-field gap better than standard ImageNet or CLIP pre-training?**

## 2. Related Work

- **SimCLR** (Chen et al., 2020) — contrastive framework used for self-supervised pre-training on unlabeled field images.
- **DANN** (Ganin & Lempitsky, 2015) — domain-adversarial training, planned for later-stage domain alignment.
- **CLIP** (Radford et al., 2021) — foundation-model baseline and initialization for continued pre-training.
- *(Placeholder: 1–2 papers from target labs on precision ag / field sensing.)*

## 3. Datasets

| Dataset | Domain | Classes (raw) | Images | Role |
|---|---|---|---|---|
| PlantVillage | Lab (clean, controlled) | 38 | ~54,305 | Source domain — supervised training |
| PlantDoc | Field (real-world) | 28 | ~2,578 | Target domain — unlabeled pre-training + labeled eval |

Overlapping classes were aligned between taxonomies; classes with fewer than 10 PlantDoc samples were dropped, leaving **27 aligned classes**.

**Domain gap visualization:** `figures/domain_gap_samples.png` — lab vs. field samples per class.

## 4. Methods

**4.1 Baseline:** ImageNet-pretrained ResNet18, fine-tuned on PlantVillage (AdamW, cosine LR, early stopping), evaluated on both PlantVillage val and PlantDoc test.

**4.2 Ag-SimCLR:** ResNet18 trained from scratch via SimCLR (NT-Xent loss) on ~2,578 unlabeled PlantDoc images.

**4.3 CLIP comparison:** CLIP ViT-B/32 evaluated frozen, both zero-shot (text-prompt similarity) and via linear probe.

**4.4 Ag-CLIP-SimCLR:** Continued contrastive pre-training on top of CLIP's ViT vision encoder, using the same unlabeled corpus. Only the last 2 transformer blocks + a new projection head (~16.9% of params) are unfrozen, to avoid catastrophic forgetting; low LR (1e-5).

*(Planned: DANN and pseudo-label self-training on the best backbone.)*

## 5. Results (in progress)

### Table 1: Frozen-feature comparison on PlantDoc test set

| Method | Setting | Accuracy |
|---|---|---|
| ImageNet ResNet18 | Linear probe (frozen) | 0.5085 |
| CLIP ViT-B/32 | Linear probe (frozen) | 0.4746 |
| CLIP ViT-B/32 | Zero-shot (no training) | 0.3051 |
| Ag-SimCLR ResNet18 (13 epochs, from scratch) | Linear probe (frozen) | 0.2458 |
| ImageNet ResNet18 | **Fully fine-tuned** on PlantVillage | 0.2246 |
| Ag-CLIP-SimCLR (continued pre-training) | Linear probe (frozen) | *pending* |

**Domain gap:** PlantVillage val 0.9973 vs. PlantDoc test 0.2246 (baseline fine-tune) — a 0.7727 drop under domain shift.

*(Planned: Fig. 1 accuracy-vs-labeled-data curve; Fig. 3 augmentation ablation; Fig. 4 failure cases.)*

**Figure 2 (t-SNE):** `figures/tsne_simclr_vs_imagenet.png` — frozen feature clustering, Ag-SimCLR vs. ImageNet.

**Confusion matrix:** the fine-tuned baseline shows a **sink-class collapse** — predictions collapse into a few "attractor" classes rather than degrading uniformly, suggesting reliance on lab-specific background/lighting cues over lesion morphology. See `figures/baseline_confusion_matrix_plantdoc.png`.

## 6. Discussion (in progress)

At small scale, **pre-training data volume matters more than domain match**: Ag-SimCLR underperforms both ImageNet features and CLIP zero-shot, despite CLIP never seeing agricultural imagery or any labels. This motivates the continued-pretraining experiment (4.4) as a way to combine foundation-model scale with domain-specific adaptation.

*(Placeholder: expand with DANN/limited-label results, hardest-to-generalize diseases, and field-deployment implications.)*

## 7. Conclusion & Future Work

*(Placeholder — to be completed once the pipeline finishes.)*

Planned extensions: multi-task learning (disease + pest + weed), UAV/edge deployment, larger unlabeled corpora.

---

## Repository Structure
```
/notebooks     - Colab notebook(s)
/figures       - Exported plots (PNG/PDF)
/metrics       - Exported JSON/CSV result files
/checkpoints   - (gitignored - weights too large for version control)
```

## Reproducibility
- Random seed: 42 (Python, NumPy, PyTorch)
- Hardware: Google Colab Free Tier, NVIDIA T4 GPU
- Key dependencies: `torch`, `torchvision`, `lightly`, `transformers`, `scikit-learn`

---

*This project is being developed as a research portfolio piece. Sections marked "pending" or "in progress" will be completed as experiments finish.*