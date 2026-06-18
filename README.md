# Animal Detection — YOLO Fine-tuning

Projeto para fine-tuning de modelos YOLO em datasets customizados, com ambiente reproduzível via `uv` e automação via `make`.

## Estrutura do projeto, Requisitos e Dependências principais

```
.
├── core/
│   ├── yolo.py           # Script principal de treino
│   └── runs/             # Checkpoints
├── runs/                 # Resultados
├── tmp/                  # Diretório temporário (gerado pelo make) 
│   ├── .venv/            # Ambiente virtual
│   ├── dataset/          # Dataset de treino (inserido pelo usuário)
│   │   └── data.yaml
│   └── yolo26n.pt        # Modelo base (baixado automaticamente)
├── pyproject.toml        # Dependências do projeto
├── uv.lock               # Lock file (não edite manualmente)
└── makefile
```

---

- Python 3.10
- [`uv`](https://github.com/astral-sh/uv) (instalado automaticamente pelo `make`)
- CPU ou GPU (CUDA)

---

| Pacote          | Descrição                        |
|-----------------|----------------------------------|
| `ultralytics`   | Framework YOLO                   |
| `torch`         | PyTorch                          |
| `torchvision`   | Utilitários de visão computacional |
| `onnxruntime`   | Inferência do modelo exportado   |
| `onnxslim`      | Otimização do modelo ONNX        |

---

## Instalação

```bash
make install
```

Isso instala o `uv` se necessário, cria o virtualenv em `./tmp/.venv` e instala todas as dependências fixadas no `uv.lock`.

## Treino

### Uso básico

Antes de começar a treinar, o usuário precisa inserir no diretório `/tmp` o dataset de treinamento, caso o nome do dataset seja diferente é necessário utilizar o argumento `--data` para apontar para o dataset.

```bash
make run
```

Usa os valores padrão definidos por padrão.

---

### Uso avançado

```bash
make run ARGS="--epochs 50 --save-every 25 --model yolo26n.pt"
```

> Este exemplo mostra três dos parâmetros de entrada que podem ser alterados.

---

### Argumentos

| Argumento    | Padrão            | Descrição                                              |
|--------------|-------------------|--------------------------------------------------------|
| `--data`     | `dataset/data.yaml` | Caminho para o arquivo de configuração do dataset    |
| `--epochs`   | `100`             | Número total de épocas de treino                       |
| `--save-every`| `10`             | Salvar checkpoint a cada N épocas                      |
| `--model`    | `yolo26n.pt`      | Modelo base para fine-tuning                           |
| `--imgsz`    | `640`             | Tamanho da imagem de entrada (pixels)                  |
| `--resume`   | `0`               | Retomar treinamento                                    |
| `--batch`    | `2`               | Tamanho do batch (reduza se faltar memória RAM)        |
| `--workers`  | `0`               | Workers do dataloader (use 0 em CPU)                   |
| `--no-amp`   | —                 | Desativa mixed precision (automático em CPU)           |

---

## Retomar de um checkpoint

Os checkpoints são salvos automaticamente em:

```
core/runs/train/weights/
├── epoch_10.pt
├── epoch_20.pt
├── ...
├── last.pt
└── best.pt
```

>O arquivo `best.pt` contém o melhor modelo ao longo do treino. O modelo final é exportado para ONNX ao término.

No caso de treino continuo de dataset:

```bash
make run ARGS="--model runs/train/weights/last.pt --resume 1"
```

Para treinar outro dataset:

```bash
make run ARGS="--data dataset2/data.yaml --model runs/train/weights/best.pt"
```

## Dispositivo

O script detecta automaticamente se há GPU disponível:

- **GPU**: usa a placa com mais memória livre e habilita mixed precision (AMP)
- **CPU**: desativa AMP

## Limpeza

```bash
# Remove ambiente virtual, dataset e runs de treino
make clean
```

> `make clean` aguarda 10 segundos antes de deletar — pressione `Ctrl+C` para cancelar.
