# Verifying Release Downloads

The Windows, Linux and DOS archives attached to a GitHub Release are built by GitHub Actions from the tagged source revision. The DOS package is named `ABHD-DOS.zip`; its build and hardware requirements are described in [the DOS build notes](https://github.com/retro-foundry/Alien-Breed-HD/blob/main/docs/dos-build.md). Stable tags such as `v1.0.0` are published as releases; tags with a prerelease suffix such as `v1.0.0-beta.1` or `v1.0.0-rc.1` are marked as prereleases. Older releases may have fewer platform packages.

## Verify Checksums

Download your platform's archive and `SHA256SUMS.txt` from the same release.

PowerShell:

```powershell
Get-FileHash .\alien-breed-3d-i-v1.0.0-windows-x64.zip -Algorithm SHA256
Get-Content .\SHA256SUMS.txt
```

The hash printed by `Get-FileHash` should match the line for the zip in `SHA256SUMS.txt`.

Git Bash, WSL, Linux, or macOS:

```bash
sha256sum --ignore-missing -c SHA256SUMS.txt
```

## Verify GitHub Provenance

GitHub artifact attestations are created for each packaged archive and the checksum file when the release workflow runs. With the GitHub CLI installed, verify the archive you downloaded:

```bash
gh attestation verify alien-breed-3d-i-v1.0.0-windows-x64.zip --repo retro-foundry/Alien-Breed-HD
gh attestation verify alien-breed-3d-i-v1.0.0-linux-x64.tar.gz --repo retro-foundry/Alien-Breed-HD

gh attestation verify ABHD-DOS.zip --repo retro-foundry/Alien-Breed-HD
gh attestation verify SHA256SUMS.txt --repo retro-foundry/Alien-Breed-HD
```

The attestation should identify `.github/workflows/release.yml` as the workflow that built or published the asset from the tagged source.
