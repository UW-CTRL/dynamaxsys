.PHONY: help clean build check publish release-next release-patch release-minor release-major release-patch-push release-minor-push release-major-push

PYTHON ?= python3

help:
	@echo "Available targets:"
	@echo "  make clean               Remove build artifacts"
	@echo "  make build               Build sdist and wheel"
	@echo "  make check               Build and validate release artifacts"
	@echo "  make publish             Upload dist/* with twine"
	@echo "  make release-next        Show the next patch tag"
	@echo "  make release-patch       Create the next patch tag"
	@echo "  make release-minor       Create the next minor tag"
	@echo "  make release-major       Create the next major tag"
	@echo "  make release-patch-push  Create and push the next patch tag"
	@echo "  make release-minor-push  Create and push the next minor tag"
	@echo "  make release-major-push  Create and push the next major tag"

clean:
	rm -rf build dist *.egg-info dynamaxsys.egg-info

build: clean
	@$(PYTHON) -m build

check: build
	@$(PYTHON) -m twine check dist/*

publish:
	@$(PYTHON) -m twine upload dist/*

release-next:
	@$(PYTHON) scripts/release_tag.py patch --dry-run --allow-dirty

release-patch:
	@$(PYTHON) scripts/release_tag.py patch

release-minor:
	@$(PYTHON) scripts/release_tag.py minor

release-major:
	@$(PYTHON) scripts/release_tag.py major

release-patch-push:
	@$(PYTHON) scripts/release_tag.py patch --push

release-minor-push:
	@$(PYTHON) scripts/release_tag.py minor --push

release-major-push:
	@$(PYTHON) scripts/release_tag.py major --push