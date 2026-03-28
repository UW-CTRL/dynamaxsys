# dynamaxsys

A common dynamical systems library written in JAX, designed to be used for various JAX projects requiring a dynamical system.

### Install

Install the repo:

```pip install dynamaxsys```

Alternatively, if you want to develop on the code base, please fork this repo and make pull requests as needed.

```pip install -e .```

### Release

This package uses Git tags and `setuptools-scm` for versioning, so releases should be cut from tags instead of editing a version string by hand.

Preview the next patch release tag:

```bash
make release-next
```

Create the next patch, minor, or major tag:

```bash
make release-patch
make release-minor
make release-major
```

Create and push the next patch, minor, or major tag:

```bash
make release-patch-push
make release-minor-push
make release-major-push
```

Build and validate release artifacts:

```bash
make check
```

Upload the built artifacts after the tag is pushed:

```bash
make publish
```

The release script intentionally fails if the working tree is dirty unless you use the dry-run target.


