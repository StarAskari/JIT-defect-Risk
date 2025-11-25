# Dataset Card
Scope: ~5,000 commits (Java/.NET)
Schema: repo, commit_sha, message, label(0/1), fix_sha, file, date, notes
Labeling: file-level one-hop from fix commits; small manual audit
Splits: time-aware within project (train/dev/test by chronology)
Licenses: public OSS only