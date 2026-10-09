# License & Intellectual Property Provenance

**Document Type: Legal & Compliance Record**  
**Status: Resolved**  
**Confirmed Date: 2026-10-09**

This document records the open-source licensing, copyright ownership, and dataset boundary agreements for `dit-ad-classifier`.

---

## 1. Codebase License Grant

All maintained source code in this repository is distributed under the [MIT License](../LICENSE).

```
Codebase Ownership Architecture:

[ Repository Code Assets ]
       |
       |-- dit/ (Maintained package) ----------------------> [ MIT License ]
       |                                                     Copyright (c) 2026 Circumsized
       |-- configs/ (YAML configurations) -----------------> [ MIT License ]
       |
       |-- tests/ (Automated test suite) ------------------> [ MIT License ]
       |
       |-- legacy/ (2020 course submission scripts) -------> [ MIT License ]
       |
       +-- legacy/afq2020_reference/ (2020 upstream) ------> [ MIT License ]
           (Confirmed as author's original work on 2026-10-09; covered under root MIT License)
```

### Copyright History

- The draft placeholder `<COPYRIGHT-HOLDER-TO-BE-CONFIRMED>` was resolved on 2026-10-08.
- Root `LICENSE` declares: `Copyright (c) 2026 Circumsized`.
- `pyproject.toml` declares: `license = { file = "LICENSE" }`.

---

## 2. Clinical Dataset Boundaries

```
Repository vs Dataset Isolation:

 +-------------------------------------------------------------+
 | Codebase (dit-ad-classifier)                                |
 |  - Source code, preprocessing, models, test harness         |
 |  - Permissive open-source usage under MIT License            |
 +-------------------------------------------------------------+
                               ^
                     Strict Air-Gap Boundary
                               v
 +-------------------------------------------------------------+
 | Clinical Neuroimaging Dataset (AI4AD / YongLiuLab)          |
 |  - MCAD_AFQ_competition.mat, DTI diffusion measurements     |
 |  - Controlled clinical research data; requires organizer DUA|
 |  - MIT license grants zero rights to proprietary data       |
 +-------------------------------------------------------------+
```

1. **No Proprietary Data Distribution**: This repository does not host, distribute, or vendor real AI4AD patient `.mat` files.
2. **Synthetic Evaluation**: Tests and demonstrations use synthetic data generated via `dit.data.synthetic`. Synthetic data contains zero Protected Health Information (PHI).
3. **Data Access**: Users working with real clinical data are responsible for securing proper access and institutional review board (IRB) approvals from the dataset organizers (CASIA / YongLiuLab).
