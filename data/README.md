# Data boundary

`raw/` and `processed/` are local ignored directories. Track schema, provenance, dataset references
and checksums rather than sensitive data in Git. Keep labels separate until evaluation policy allows
joining them. Define sensitive-field redaction, retention and access before ingesting actual data.

Synthetic event fixtures live in `contracts/examples/v1`. No real transactions or fraud labels are
included. Use external versioned storage/DVC or a data registry after the data platform is selected.
