PYTHON = ./tmp/.venv/bin/python
UV     = uv

dependencies:
	@which uv > /dev/null 2>&1 || curl -Ls https://astral.sh/uv/install.sh | sh

./tmp/.venv/.installed: uv.lock pyproject.toml
	-mkdir tmp
	UV_PROJECT_ENVIRONMENT=./tmp/.venv $(UV) sync
	touch $@

clean:
	@echo "This will delete the dataset and trained data too, if do not want to \
	delete all press ctrl + c or ctrl + z"
	@echo "Sleeping for 10 seconds..."
	@sleep 10
	rm -rf ./tmp ./core/runs ./runs yolo26n.pt

install: dependencies ./tmp/.venv/.installed

run: install
	$(PYTHON) ./core/yolo.py $(ARGS)

.PHONY: clean dependencies install run