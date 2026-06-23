from ultralytics import YOLO
from multiprocessing import freeze_support
import argparse
import torch
import os

# Base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Default arguments,
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data",       type=str, default=os.path.join(BASE_DIR, "tmp", "dataset", "data.yaml"))
    p.add_argument("--epochs",     type=int, default=100)
    p.add_argument("--save_every", type=int, default=10, dest="save_every")
    p.add_argument("--model",      type=str, default=os.path.join(BASE_DIR, "tmp", "yolo26x.pt"))
    p.add_argument("--optimizer",  type=str, default="auto")
    p.add_argument("--project",    type=str, default=os.path.join(BASE_DIR, "core", "runs"))
    p.add_argument("--batch",      type=int, default=-1)
    p.add_argument("--workers",    type=int, default=max(1, (os.cpu_count()-2)))
    p.add_argument("--imgsz",      type=int, default=640)
    p.add_argument("--exist_ok",   type=int, default=0, choices=[0, 1])
    p.add_argument("--resume",     type=int, default=0, choices=[0, 1])
    p.add_argument("--amp",        type=int, default=1, choices=[0, 1])
    return p.parse_args()

############################################################################

def get_device(args):
    if not torch.cuda.is_available():
        args.amp = 0
        return "cpu"

    best_gpu, max_free = 0, 0
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        free = props.total_memory - torch.cuda.memory_allocated(i)
        print(f"GPU {i}: {props.name} — livre: {free / 1024**3:.1f} GB")
        if free > max_free:
            max_free = free
            best_gpu = i

    print(f"Usando GPU {best_gpu}")
    return best_gpu

def fine_tuning(args):
    freeze_support()

    device = get_device(args)

    model = YOLO(args.model)

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=device,
            optimizer=args.optimizer,
            name="train",
            save=True,
            exist_ok=bool(args.exist_ok),
            resume=bool(args.resume),
            batch=args.batch,
            workers=args.workers,
            save_period=args.save_every,
            amp=bool(args.amp),
            project=args.project,
        )
    except Exception as e:
        print(f"Erro: {e}")
        raise

    metrics = model.val()
    path = model.export(format="onnx")
    print(f"Modelo exportado: {path}")

if __name__ == "__main__":
    args = parse_args()
    fine_tuning(args)
