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
    p.add_argument("--save-every", type=int, default=10, dest="save_every")
    p.add_argument("--model",      type=str, default="yolo26x.pt")
    p.add_argument("--optimizer",  type=str, default="auto")
    p.add_argument("--project",    type=str, default=os.path.join(BASE_DIR, "core", "runs"))
    p.add_argument("--batch",      type=int, default=-1)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--imgsz",      type=int, default=640)
    p.add_argument("--exist_ok",   type=bool, default=False)
    p.add_argument("--resume",     type=bool, default=False)
    p.add_argument("--amp",        action="store_true", default=True)
    p.add_argument("--no-amp",     action="store_false", dest="amp")
    return p.parse_args()

############################################################################

def get_device(args):
    if not torch.cuda.is_available():
        args.amp = False
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

    model = YOLO(os.path.join(BASE_DIR, "tmp", args.model))

    try:
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            device=device,
            optimizer=args.optimizer,
            name="train",
            save=True,
            exist_ok=args.exist_ok,
            resume=args.resume,
            batch=args.batch,
            workers=args.workers,
            save_period=args.save_every,
            amp=args.amp,
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
