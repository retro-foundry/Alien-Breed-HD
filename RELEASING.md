# Releasing

Alien Breed 3D I releases are currently Windows-only beta builds produced by GitHub Actions.

## Create a Beta Release

1. Update the numeric CMake project version in `CMakeLists.txt` when the underlying beta series changes.
2. Commit the release-ready source.
3. Create and push a beta tag:

```bash
git tag v0.9.0-beta.1
git push origin v0.9.0-beta.1
```

Tags matching `v*.*.*-beta*` run `.github/workflows/release.yml`. The workflow builds a clean Windows x64 Release binary, packages the staged runtime files, creates `SHA256SUMS.txt`, creates GitHub artifact attestations where supported, and publishes a GitHub prerelease.

Release assets appear on the GitHub Release for the tag:

- `alien-breed-3d-i-<tag>-windows-x64.zip`
- `SHA256SUMS.txt`

The bundled `VERIFY_RELEASE.md` explains how players can verify checksums and provenance.

## CI

`.github/workflows/ci.yml` runs on pushes to `main` and on pull requests. It configures and builds the Windows Release target and checks that the expected runtime output was staged next to `ab3d1.exe`.

## Optional Windows Code Signing

Signing is not required for the beta release workflow. To enable it later, add these repository or environment secrets:

- `WINDOWS_SIGNING_CERT_BASE64`: Base64-encoded `.pfx` signing certificate.
- `WINDOWS_SIGNING_CERT_PASSWORD`: Password for the `.pfx` certificate.

When both secrets are present, the release workflow signs `build/Release/ab3d1.exe` before packaging it.

## Security Notes

Release publishing only runs for tags in the `retro-foundry/Alien-Breed-HD` repository. The workflow uses the built-in `GITHUB_TOKEN` and grants write permissions only to the publish job. Keep beta tags protected in GitHub repository settings so only trusted maintainers can create or move release tags.
