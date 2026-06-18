PYTHON = ./tmp/.venv/bin/python
UV     = uv

dependencies:
	@which uv > /dev/null 2>&1 || curl -Ls https://astral.sh/uv/install.sh | sh

./tmp/.venv/.installed: uv.lock pyproject.toml
	UV_PROJECT_ENVIRONMENT=./tmp/.venv $(UV) sync
	touch $@

clean:
	rm -rf ./tmp/.venv ./tmp/yolo26x.pt ./tmp/yolov8n.pt

clean-all: clean
	@echo "This will delete the dataset and trained data too, if do not want to \
	delete all press ctrl + c or ctrl + z"
	@echo "Sleeping for 5 seconds..."
	@sleep 5
	rm -rf ./tmp/dataset ./core/runs ./runs

install: dependencies ./tmp/.venv/.installed

run: install
	$(PYTHON) ./core/yolo.py $(ARGS)

.PHONY: clean clean-all dependencies install run

#=============================
# Treino padrão
#make run
# Dataset customizado, salvando a cada 5 épocas
#make run ARGS="--epochs 2 --save-every 1 --batch 2 --workers 0 --model yolov8n.pt"
# Retomar de um checkpoint
#make run ARGS="--epochs 2 --save-every 1 --batch 2 --workers 0 --model ./core/runs/train/weights/last.pt"