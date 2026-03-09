## Contribution Workflow

1. **Fork** the repository.
2. **Create a branch**: `git checkout -b feat/your-feature-name`
3. **Commit your changes**: `git commit -m "feat: add new feature"`
4. **Push to your Fork**: `git push origin feat/your-feature-name`
5. **Create a Pull Request** targeting the `main` branch of the upstream repository.

## Commit Message Convention

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

Plaintext

```
<type>[optional scope]: <description>

[optional body]

[optional footer(s)]
```

### Type Definitions

- **feat**: A new feature.
- **fix**: A bug fix.
- **docs**: Documentation only changes.
- **refactor**: A code change that neither fixes a bug nor adds a feature.
- **test**: Adding missing tests or correcting existing tests.
- **chore**: Changes to the build process or auxiliary tools and libraries.

### Example

Plaintext

```
feat(auth): add JWT token validation

- Implement token verification
- Add token refresh endpoint

Closes #123
```

## Code Review Process

- All PRs must receive at least **one approval** before merging.
- All **CI checks** must pass.
- All review comments must be addressed/resolved.
- We use **"Squash and merge"** for all PRs.

## Release Process

Only maintainers are authorized to release new versions: `make release VERSION=1.0.0`
