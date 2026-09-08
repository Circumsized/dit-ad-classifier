# LICENSE is unset and needs an owner decision

`pyproject.toml` does not declare a license, and there is no `LICENSE` file.

This is deliberate. The license and copyright holder are a legal/ownership
decision that an automated change must not make on the author's behalf. An
earlier revision declared `license = { text = "MIT" }` in metadata without a
matching `LICENSE` file, which would misstate the distribution terms.

## What the repository owner needs to do

1. Choose a license (for example MIT, Apache-2.0, BSD-3-Clause, or a
   proprietary/all-rights-reserved statement).
2. Add a `LICENSE` file with the full text and the correct copyright holder and
   year.
3. Restore the metadata declaration in `pyproject.toml`, e.g.:

   ```toml
   license = { file = "LICENSE" }
   ```

Until then the code is unlicensed by default, which for others means no granted
rights to use, copy, or distribute it. Course-assignment provenance and the
AI4AD data-use terms should be considered when choosing.
