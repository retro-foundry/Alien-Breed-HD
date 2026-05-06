# Verifying Beta Release Downloads

The Windows beta archive attached to each GitHub Release is built by GitHub Actions from the tagged source revision. The release is marked as a prerelease while the port is still in beta.

## Verify Checksums

Download the Windows zip and `SHA256SUMS.txt` from the same release.

PowerShell:

```powershell
Get-FileHash .\alien-breed-3d-i-v0.9.0-beta.1-windows-x64.zip -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
```

The hash printed by `Get-FileHash` should match the line for the zip in `SHA256SUMS.txt`.

Git Bash, WSL, Linux, or macOS:

```bash
sha256sum -c SHA256SUMS.txt
```

## Verify GitHub Provenance

GitHub artifact attestations are created for the packaged zip and checksum file when the release workflow runs. With the GitHub CLI installed:

```bash
gh attestation verify alien-breed-3d-i-v0.9.0-beta.1-windows-x64.zip --repo retro-foundry/Alien-Breed-HD
gh attestation verify SHA256SUMS.txt --repo retro-foundry/Alien-Breed-HD
```

The attestation should identify `.github/workflows/release.yml` as the workflow that built or published the asset from the tagged source.
