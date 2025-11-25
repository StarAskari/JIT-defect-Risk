# Labeling Protocol (File-level Linkage)
- Fix keywords: fix, bug, hotfix, resolve, patch, regression, NRE, NullReference, exception, crash
- Exclusions: docs, formatting, rename, merges
- Mapping: for each fix, choose the most recent prior commit that changed the main fixed file → label = 1
- Negatives: random commits from same period not linked to fixes → label = 0
- Audit: ~50–100 samples to estimate label precision