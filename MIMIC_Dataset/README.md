# MIMIC_Dataset (not included)

This folder is intentionally empty in the repository. Place the official PhysioNet
releases here, unchanged (the original `.csv.gz` files and each release's
`SHA256SUMS.txt`), with exactly these folder names:

```
MIMIC_Dataset/
├── MIMIC-IV-Demo-2.2/        open access   https://physionet.org/content/mimic-iv-demo/2.2/
├── MIMIC-IV-ED-Demo-2.2/     open access   https://physionet.org/content/mimic-iv-ed-demo/2.2/
├── MIMIC-IV-Full-2.2/        credentialed  https://physionet.org/content/mimiciv/2.2/
└── MIMIC-IV-ED-Full-2.2/     credentialed  https://physionet.org/content/mimic-iv-ed/2.2/
```

Files actually read by the pipeline:

| Release | Files |
|---|---|
| MIMIC-IV (Demo and Full) | `hosp/patients.csv.gz`, `hosp/admissions.csv.gz`, `hosp/labevents.csv.gz`, `icu/icustays.csv.gz` (Full only, for the ICU-patient null) |
| MIMIC-IV-ED (Demo and Full) | `ed/edstays.csv.gz`, `ed/triage.csv.gz`, `ed/vitalsign.csv.gz`, `ed/medrecon.csv.gz`, `ed/pyxis.csv.gz` (`ed/diagnosis.csv.gz` is deliberately not used: ED diagnoses are coded after discharge) |

The ED preparation also reads `hosp/patients.csv.gz` and `hosp/admissions.csv.gz` from the
matching MIMIC-IV release (Demo with ED-Demo, Full with ED-Full).

Nothing in this folder may be committed. `.gitignore` excludes everything here except this file.
