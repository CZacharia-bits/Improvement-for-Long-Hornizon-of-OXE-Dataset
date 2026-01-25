# Project Guidelines

## Contribution Workflow

Every chunk of work follows this process:

### 1. Create a GitHub Issue

Before starting work, create an issue describing what you're doing:

```bash
gh issue create --title "Short description of the work" --body "Detailed description of what needs to be done"
```

This creates a trackable record of the work and allows for discussion before implementation.

### 2. Create a Branch

Create a branch from `master` with a descriptive name:

```bash
git checkout master
git pull origin master
git checkout -b <your-name>/<short-description>
```

**Branch naming convention:**
- `andrew/add-feature-x`
- `andrew/fix-bug-in-loader`
- `andrew/update-readme`

### 3. Make Your Changes

- Keep commits focused and atomic
- Write clear commit messages describing what changed and why
- Test your changes locally before pushing

### 4. Push and Create a PR

```bash
git push -u origin <branch-name>
gh pr create --title "Description of changes" --body "## Summary
- Change 1
- Change 2

## Test plan
- [ ] How to verify the changes work

Fixes #<issue-number>"
```

**PR guidelines:**
- Reference the issue with `Fixes #N` to auto-close it on merge
- Include a summary of changes
- Add a test plan so reviewers know how to verify

### 5. Review and Merge

- Address any review feedback
- Once approved, merge via GitHub
- Delete the branch after merging

---

## Quick Reference

| Step | Command |
|------|---------|
| Create issue | `gh issue create --title "..." --body "..."` |
| Create branch | `git checkout -b name/description` |
| Push branch | `git push -u origin branch-name` |
| Create PR | `gh pr create --title "..." --body "..."` |
| List open PRs | `gh pr list` |
| View PR status | `gh pr view <number>` |

---

## Project Setup

```bash
make install        # Install dependencies
make visualize-all  # Verify everything works
```

## Running Tests

```bash
make visualize      # Test visualization on default dataset
make inspect        # Test dataset inspection
make evaluate       # Test evaluation framework
```
