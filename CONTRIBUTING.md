# Contributing

Thank you for helping. Three kinds of contribution are especially useful.

**Corrections.** If an equation, sign or parameter is wrong, open an issue with the reference (book section or paper) and, if you can, a failing test.

**Better tuning.** Every controller's scorecard is reproducible. If your gains score better on the benchmark missions, open a pull request that changes only the gains and paste the before and after scorecards.

**New vehicles.** Add one file in `auvschool/vehicles/`. State where every parameter comes from and its licence. Do not copy parameters from sources that forbid reuse.

## Ground rules

- Run `pytest` before opening a pull request; all checks must pass.
- Keep episode scripts short and readable. They are teaching material first.
- Cite sources. Do not paste figures or text from books or papers.
- Be kind in issues and reviews. Many readers are students.
