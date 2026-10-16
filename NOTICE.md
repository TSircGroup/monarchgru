# Notice

The code and the checkpoint in this repository are released under the MIT License (`LICENSE`), except
for the files below.

- **HazyResearch/fly** (https://github.com/HazyResearch/fly), Apache License 2.0.
  `blockdiag_butterfly_multiply.py`, `structured_linear.py` and `monarch_linear.py` are derived from
  that repository's Monarch layer code and were modified for this project. They remain under the
  Apache License 2.0; its full text is in `LICENSE-apache` (a copy of the upstream `LICENSE`).
  The method is from Dao et al., "Monarch: Expressive Structured Matrices for Efficient and Accurate
  Training", ICML 2022.

- **Qualcomm-AI-research/bcresnet** (https://github.com/Qualcomm-AI-research/bcresnet), BSD-3-Clause-Clear
  license.
  The data preparation and augmentation in `utils.py` follow that repository's pipeline, from
  Kim et al., "Broadcasted Residual Learning for Efficient Keyword Spotting", Interspeech 2021
  (arXiv:2106.04140). Parts of `utils.py` are adapted from that repository's code and stay under its
  BSD-3-Clause-Clear license; the file carries its copyright line.
  The license text is in `LICENSE-bcresnet` (Copyright (c) 2023 Qualcomm Technologies, Inc.). This project
  is not endorsed by Qualcomm.

- **Paper figures.** `assets/system_overview.jpg` and `assets/monarchgru_cell.jpg` are Figs. 1 and 3 of
  "MonarchGRU: A Low-Footprint Recurrent Neural Network for Audio Keyword Spotting System", IEEE AICAS 2026.
  © 2026 IEEE. They are reused by the authors with credit and are not covered by the MIT License.
