# Third-party notices

This repository contains or downloads the following third-party materials.

## Included source

`vendor/loralib/` is copied from Microsoft LoRA commit
`c4593f060e6a368d7bb5af5273b8e42810cdef90`. It is distributed under the MIT
License reproduced in `vendor/LICENSE.md`. The upstream source URLs and SHA-256
digests are recorded in `asset_manifest.json`.

## Downloaded during a full rerun

- `FacebookAI/roberta-base`, revision
  `e2da8e2f811d1448a5b465c236feacd80ffbac7b`.
- `nyu-mll/glue` MRPC, revision
  `bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c`.
- Google Research Datasets PAWS-Wiki `labeled_final` test data, revision
  `161ece9501cf0a11f3e48bd356eaa82de46d6a09`.

The first two are not redistributed here because the local artifact did not
contain sufficient upstream license documentation to justify republication.
The PAWS dataset notice says the dataset may be freely used for any purpose and
requests acknowledgement of Google LLC; that notice is reproduced in
`PAWS-LICENSE.txt`. The raw PAWS data is still excluded to keep the repository
small and to avoid embedding third-party sentence text in evidence files.

Dependency names and exact versions are listed in `requirements-lock.txt`.
Those packages are installed from their package indexes and are not vendored in
this repository.

