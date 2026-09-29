# Terraform integration boundary

The cloud/provider, region, account, networking and state backend are unspecified. `versions.tf`
defines an initial Terraform version constraint; there are no deployable resources yet.

After provider selection, create reviewed modules for network, cluster, managed stores, identities
and observability. Separate environment state and credentials; configure an encrypted remote
backend with locking; review plans before any apply. Never commit state, secrets or private keys.
Do not invent provider resources or budget claims to make the skeleton appear deployed.
