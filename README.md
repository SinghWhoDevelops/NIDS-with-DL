# Network Intrusion Detection with Deep Learning (CIC-IDS2017)

Multi-class network intrusion detection on CIC-IDS2017 flow data, comparing a Random Forest baseline with a PyTorch MLP, followed by an adversarial-robustness study (FGSM / PGD + adversarial training) and a feature-importance analysis.

**Author:** `SinghWhoDevelops` · **Status:** work in progress

## Highlights

- Built a full pipeline: EDA → cleaning → leakage-safe preprocessing → baseline → deep model → robustness → explainability.
- Both models detect attacks vs. benign traffic with ~99.5% recall (RF 0.9952, MLP 0.9941), but **macro-F1 is much lower (RF 0.8585, MLP 0.8029)** because a few rare web-attack classes are hard. Accuracy (99.85% / 99.65%) is misleading here since 85% of the data is benign.
- The MLP **did not beat Random Forest** on this tabular data. The gap comes mostly from two classes (Web Attack - XSS and the merged Rare Attack class), see "Findings".
- Merging the two near-identical web-attack classes (XSS + Brute Force) lifts macro-F1 to 0.9177 (RF) / 0.8862 (MLP), while the other 11 classes stay essentially unchanged (0.9131 / 0.8816 vs. 0.9151 / 0.8781).
- Under a white-box, feature-space attack the original MLP is fragile (49% of attack flows evade at eps = 0.2 std under PGD-10, versus 0.6% on clean data). Adversarial training largely fixes this up to the training strength but protection fades beyond it. PGD is the more trustworthy estimate than FGSM here (see "Adversarial robustness").

## Dataset

**CIC-IDS2017**, created by the Canadian Institute for Cybersecurity (University of New Brunswick): https://www.unb.ca/cic/datasets/ids-2017.html

The dataset is **not included in this repository** (it is several GB and is distributed by its authors). This project was run on a pre-processed **parquet "no-metadata" version** of CIC-IDS2017: 8 files, 2,313,810 flows, 77 features + label, 15 original classes.

Download the parquet files from `<PASTE_DOWNLOAD_LINK>` and place them in `data/raw/`:

```
data/raw/
├── Benign-Monday-no-metadata.parquet
├── Bruteforce-Tuesday-no-metadata.parquet
├── DoS-Wednesday-no-metadata.parquet
├── WebAttacks-Thursday-no-metadata.parquet
├── Infiltration-Thursday-no-metadata.parquet
├── Portscan-Friday-no-metadata.parquet
├── DDoS-Friday-no-metadata.parquet
└── Botnet-Friday-no-metadata.parquet
```

To confirm you have the same version, `python eda.py` should print `Files loaded: 8` and `Shape: (2313810, 78)`.

**Only this parquet version was tested.** The original CIC-IDS2017 CSV files differ (more rows, `inf`/`NaN` values, upper-case `BENIGN` label), and these scripts do not handle those differences.

**Citation:** I. Sharafaldin, A. Habibi Lashkari and A. A. Ghorbani, "Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization", ICISSP 2018.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `eda.py` | Merge files, class distribution, inf/NaN, duplicates, constant and correlated columns |
| 2 | `clean.py` | Remove 82,004 duplicate rows and 2,836 rows with invalid negative values (3.67% of the data); drop 8 constant and 12 highly correlated (\|r\| > 0.99) columns; clean label names |
| 3 | `prepare.py` | Merge 3 tiny classes, encode labels, stratified 70/15/15 split, `log1p` + `StandardScaler` fit on **train only** |
| 4 | `baseline_rf.py` | Random Forest (100 trees) |
| 5 | `train_mlp.py` | PyTorch MLP, model selection on validation macro-F1 |
| 6 | `merge_experiment.py` | Extra: merge XSS + Brute Force into one class (12 classes) |
| 7 | `fgsm.py` | FGSM / PGD robustness test and adversarial training |
| 8 | `explain.py` | Permutation feature importance for both models |

Design decisions worth knowing:

- **Duplicates removed before splitting**, so identical rows cannot appear in both train and test.
- **Scaler fit on train only** (no leakage). `Init Fwd/Bwd Win Bytes` contain `-1` as a "no value" placeholder; these were kept and turned into two explicit indicator features instead of being dropped.
- **Rare classes merged:** Heartbleed (7 rows after deduplication), Infiltration (35) and Web Attack - SQL Injection (21) became one `Rare Attack` class (63 rows), because a 70/15/15 split of 7 rows leaves ~1 test sample.
- Final data: 2,228,970 rows, 59 features, 13 classes. Split sizes: 1,560,279 train / 334,345 val / 334,346 test.
- The test set is used once per model; best epoch/checkpoint is chosen on validation.

## Results (13 classes, test set, main benchmark)

| Model | Macro-F1 | Accuracy | Attack-vs-benign precision | Attack-vs-benign recall |
|---|---|---|---|---|
| Random Forest | **0.8585** | 0.9985 | 0.9970 | 0.9952 |
| MLP (59→256→128→64→13, BatchNorm, Dropout, sqrt-inverse class weights) | 0.8029 | 0.9965 | 0.9861 | 0.9941 |

Per-class F1 on the test set:

| Class | Test rows | Random Forest | MLP |
|---|---|---|---|
| Benign | 283,901 | 0.9993 | 0.9982 |
| Bot | 215 | 0.6984 | 0.6960 |
| DDoS | 19,200 | 0.9995 | 0.9982 |
| DoS GoldenEye | 1,542 | 0.9961 | 0.9634 |
| DoS Hulk | 25,903 | 0.9973 | 0.9923 |
| DoS Slowhttptest | 784 | 0.9689 | 0.8937 |
| DoS slowloris | 808 | 0.9944 | 0.9717 |
| FTP-Patator | 889 | 0.9966 | 0.9966 |
| PortScan | 294 | 0.9247 | 0.9322 |
| Rare Attack (merged) | 9 | 0.5000 | 0.2759 |
| SSH-Patator | 483 | 0.9906 | 0.9404 |
| Web Attack - Brute Force | 220 | 0.7638 | 0.7784 |
| Web Attack - XSS | 98 | 0.3314 | 0.0000 |

## Findings

- **Accuracy hides the problem.** 85.46% of flows are benign, so macro-F1 and per-class results are the numbers to look at.
- **XSS vs. Brute Force are nearly indistinguishable** for flow-level features. In the MLP's confusion matrix, about 90% of XSS test flows are predicted as Brute Force and 10% as Benign, which is why XSS F1 is 0.0 for the MLP. The Random Forest also only reaches 0.33.
- **Two classes explain most of the MLP-RF gap:** XSS (0.3314 / 13 ≈ 0.026 of macro-F1) and Rare Attack ((0.5000 − 0.2759) / 13 ≈ 0.017) account for ~77% of the 0.0556 difference.
- Packet-length features (e.g. `Bwd Packet Length Std/Mean`, `Packet Length Std`) were the most important Random Forest features, both by impurity and by permutation importance. The MLP relies on partly different features (see "Feature importance").
- The MLP's validation macro-F1 stopped improving after epoch 9 while training loss kept falling, so the best-validation checkpoint was used.

## Extra experiment: merging XSS and Brute Force (12 classes)

XSS and Brute Force were merged into a single `Web Attack` class, and both models were retrained with identical settings. The 13-class result above remains the main benchmark; this is an extra experiment.

| Model | Classes | Macro-F1 | Macro-F1 on the 11 unchanged classes |
|---|---|---|---|
| Random Forest | 13 | 0.8585 | 0.9151 |
| MLP | 13 | 0.8029 | 0.8781 |
| Random Forest | 12 | 0.9177 | 0.9131 |
| MLP | 12 | 0.8862 | 0.8816 |

- 12-class and 13-class macro-F1 are **not directly comparable** (the 12-class problem is easier). The last column, computed on the same 11 classes in both settings, is the fair comparison: it barely moves, so the merge did not hurt the other classes. Differences of ~0.002-0.004 are within noise (one extra test row in `Rare Attack` shifts that column by ~0.009).
- The merged `Web Attack` class reaches F1 0.9678 (RF, recall 0.9465) and 0.9361 (MLP, recall 0.9214) on 318 test flows.
- The RF-MLP macro-F1 gap shrinks from 0.0556 (13 classes) to 0.0315 (12 classes). `Rare Attack` stays weak (F1 0.4615 RF / 0.3226 MLP, 9 test rows).

## Adversarial robustness (FGSM / PGD)

**Threat model:** white-box attacker, perturbs only attack flows, each feature may move by at most `eps` standard deviations (data is standardized), perturbed values are clipped to the training range, and the two binary indicator features are not perturbed.

The 13-class MLP was attacked with FGSM (1 step) and PGD-10 (10 steps) on the test set, then retrained with FGSM-based adversarial training (`eps` sampled up to 0.2, clean + adversarial examples per batch, best epoch chosen on validation). "Detection rate" is the share of attack flows not classified as Benign.

![Robustness curves](reports/figures/fgsm_robustness.png)

Test results (`eps` is in standard deviations; the test set has 50,445 attack flows):

**Detection rate** (share of attack flows not classified as Benign)

| eps | Original, FGSM | Original, PGD-10 | AdvTrained, FGSM | AdvTrained, PGD-10 |
|---|---|---|---|---|
| 0.0 | 0.9941 | 0.9941 | 0.9944 | 0.9944 |
| 0.05 | 0.9893 | 0.9890 | 0.9941 | 0.9941 |
| 0.1 | 0.9551 | 0.9292 | 0.9934 | 0.9929 |
| 0.2 | 0.6869 | 0.5079 | 0.9897 | 0.9818 |
| 0.3 | 0.2648 | 0.1965 | 0.9720 | 0.8977 |
| 0.5 | 0.1252 | 0.1604 | 0.5919 | 0.2550 |
| 1.0 | 0.0887 | 0.1592 | 0.0795 | 0.1197 |

**Macro-F1** (13 classes; only attack flows are perturbed)

| eps | Original, FGSM | Original, PGD-10 | AdvTrained, FGSM | AdvTrained, PGD-10 |
|---|---|---|---|---|
| 0.0 | 0.8029 | 0.8029 | 0.8013 | 0.8013 |
| 0.05 | 0.7892 | 0.7878 | 0.7975 | 0.7959 |
| 0.1 | 0.7060 | 0.6410 | 0.7919 | 0.7901 |
| 0.2 | 0.3395 | 0.2258 | 0.7878 | 0.7316 |
| 0.3 | 0.1360 | 0.0903 | 0.8187 | 0.5544 |
| 0.5 | 0.0750 | 0.0715 | 0.4664 | 0.1354 |
| 1.0 | 0.1022 | 0.0715 | 0.1446 | 0.0712 |

- **The original MLP is fragile.** At eps = 0.2 under PGD-10, detection falls from 0.9941 to 0.5079: about 24,800 of the 50,445 attack flows evade, versus about 300 missed on clean data (roughly 80x more). Under FGSM the same eps gives 0.6869.
- **PGD-10 is stronger than FGSM** up to eps = 0.3 (lower detection and macro-F1). At eps >= 0.5 both attacks saturate (values are clipped to the training range), so the ordering there is not meaningful.
- **Adversarial training works within its training strength, at almost no clean cost.** Clean macro-F1 moves from 0.8029 to 0.8013 (-0.0016). At eps = 0.2 under PGD-10, detection is 0.9818 (about 920 evasions instead of about 24,800, roughly 27x fewer), and at eps = 0.3 it is still 0.8977 even though training only went up to eps = 0.2. Protection collapses beyond that (0.2550 at eps = 0.5; at eps = 1.0 it is no better than the original).
- **FGSM alone overstates robustness.** For the adversarially trained model, FGSM macro-F1 at eps = 0.3 (0.8187) is *higher* than on clean data (0.8013), while PGD-10 gives 0.5544 at the same eps. This is a known warning sign (label leaking / gradient masking), so the PGD-10 numbers are the more honest estimate.
- **PGD-10 is not the strongest possible attack.** Stronger or adaptive attacks (more steps, random restarts) could lower the defended model's numbers further; this was not tested.

**Caveat:** this is a feature-space test. Real traffic features are interdependent (packets, bytes, timing), so a real attacker could not set them freely. Treat the numbers as a worst-case robustness estimate, not a demonstrated exploit.

## Feature importance

Importance = drop in macro-F1 when one feature's values are shuffled (mean of 3 repeats), on a validation sample capped at 5,000 rows per class.

| Rank | MLP | Drop | Random Forest | Drop |
|---|---|---|---|---|
| 1 | Fwd Seg Size Min | 0.1462 | Bwd Packet Length Std | 0.0569 |
| 2 | Fwd Act Data Packets | 0.0821 | Bwd Packet Length Max | 0.0211 |
| 3 | Fwd IAT Min | 0.0640 | Bwd Packet Length Mean | 0.0195 |
| 4 | Init Bwd Win Bytes | 0.0623 | Fwd IAT Std | 0.0149 |
| 5 | ACK Flag Count | 0.0611 | Flow IAT Min | 0.0121 |
| 6 | Flow IAT Min | 0.0596 | Fwd Header Length | 0.0118 |
| 7 | PSH Flag Count | 0.0566 | Bwd Packets/s | 0.0114 |
| 8 | Bwd Packet Length Mean | 0.0562 | Flow IAT Std | 0.0104 |
| 9 | Packet Length Mean | 0.0513 | Fwd IAT Min | 0.0093 |
| 10 | Total Fwd Packets | 0.0490 | Packet Length Std | 0.0091 |
| 11 | Fwd Packet Length Max | 0.0443 | Fwd Packet Length Max | 0.0091 |
| 12 | Packet Length Max | 0.0394 | Bwd Packets Length Total | 0.0073 |
| 13 | Packet Length Variance | 0.0331 | Total Fwd Packets | 0.0070 |
| 14 | Bwd IAT Min | 0.0322 | Init Fwd Win Bytes | 0.0066 |
| 15 | Packet Length Std | 0.0286 | Flow Packets/s | 0.0064 |

![Feature importance](reports/figures/feature_importance.png)

- Only **6 of the top 15** features are shared (`Bwd Packet Length Mean`, `Flow IAT Min`, `Fwd IAT Min`, `Fwd Packet Length Max`, `Packet Length Std`, `Total Fwd Packets`); the Spearman rank correlation over all 59 features is **0.356**. The two models rely on partly different evidence.
- The Random Forest leans on packet-length and inter-arrival-time statistics. The MLP leans more on TCP-level features (`Fwd Seg Size Min`, `Init Bwd Win Bytes`, `ACK`/`PSH` flag counts) and depends on its top feature ~2.6x more strongly (0.1462 vs. 0.0569).
- **Hypothesis (not tested):** TCP window and segment-size features can reflect the attacker machine's OS/tool configuration rather than attack behaviour, so the MLP may partly be learning a capture-environment fingerprint. A feature-ablation run (retraining without these features) would test this.

Permutation importance is computed identically for both models on a class-capped validation sample. Highly correlated features share importance, so a low score does not necessarily mean a feature is useless.

## Limitations

- Single dataset, single capture environment, single random seed; results may not transfer to other networks.
- Several classes have very few test rows (Rare Attack: 9, XSS: 98), so their scores are noisy.
- No hyperparameter search was done; the RF baseline uses no class weights while the MLP uses sqrt-inverse class weights.
- Robustness results are feature-space only (see caveat above).
- The MLP's reliance on TCP-level features may reflect capture-environment artifacts rather than attack behaviour; this was not tested.

## Reproduce

```bash
pip install -r requirements.txt
# download the 8 parquet files into data/raw/ (see "Dataset" above)
python eda.py
python clean.py
python prepare.py
python baseline_rf.py
python train_mlp.py
python merge_experiment.py
python fgsm.py
python explain.py
```

Outputs go to `data/interim/`, `data/processed/`, `models/` and `reports/` (data and models are git-ignored where large).
