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
│   └── yolo26x.pt        # Modelo base (baixado automaticamente)
├── .gitignore            # ---
├── LICENSE               # ---
├── makefile              # Automação básica
├── pyproject.toml        # Dependências do projeto
├── README                # ---
├── uv.lock               # Lock file (não edite manualmente)
└── yolo26n.pt            # Utilizado para teste AMP automatico
```

---

- Python >=3.10,<3.14
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

### Argumentos¹

| Argumento    | Padrão            | Descrição                                              |
|--------------|-------------------|--------------------------------------------------------|
| `--data`     | `dataset/data.yaml` | Caminho para o arquivo de configuração do dataset    |
| `--epochs`   | `100`             | Número total de épocas de treino                       |
| `--save-every`| `10`             | Salvar checkpoint a cada N épocas                      |
| `--model`    | `yolo26x.pt`      | Modelo base para fine-tuning                           |
| `--optimizer` | `auto`           | Otimizador                                             |
| `--imgsz`    | `640`             | Tamanho da imagem de entrada (pixels)                  |
| `--resume`   | `0`               | Retomar treinamento                                    |
| `--batch`    | `-1`              | Tamanho do batch (auto 60% de uso)                     |
| `--workers`  | `4`               | Workers do dataloader (use 4 em CPU)                   |
| `--no-amp`   | —                 | Desativa mixed precision (automático em CPU)           |

> ¹ Em caso de duvidas, veja a documentação oficial dos parâmetros em [`ultralytics`](https://docs.ultralytics.com/usage/cfg#train-settings).

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

> O arquivo `best.pt` contém o melhor modelo ao longo do treino. O modelo final é exportado para ONNX ao término.

No caso de treino continuo de dataset, retomando de onde parou:

```bash
make run ARGS="--model runs/train/weights/last.pt --resume True"
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

> **Cuidado**, esse comando **vai limpar o projeto**. Aguarde 10 segundos antes de deletar — pressione `Ctrl+C` ou `Ctrl+Z` para cancelar.

## Anotações

> 1 - Esse script é uma automação básica para fine-tuning e não foi testado de forma intensa, caso encontre algum erro, sinta-se livre modificalo e adapta-lo para o seu caso.

> 2 - Para os parâmetros, foram escolhidos apenas os julgados essenciais da documentação oficial para uso genérico.

---

## Licença

MIT
