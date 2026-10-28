"""
將 PyTorch 權重 (.pt) 轉為 NumPy (.npz)，網頁推論不需安裝 torch

  pip install torch   # 僅轉檔時需要
  python scripts/convert_model.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "models" / "model_S2plusMonth.pt"
DST = ROOT / "models" / "model_S2plusMonth.npz"


def main():
    state = torch.load(SRC, map_location="cpu")
    arrays = {k.removeprefix("module.").replace(".", "__"): v.numpy() for k, v in state.items()}
    np.savez_compressed(DST, **arrays)
    print(f"{DST.name}：{', '.join(f'{k}{tuple(v.shape)}' for k, v in arrays.items())}")


if __name__ == "__main__":
    sys.exit(main())
