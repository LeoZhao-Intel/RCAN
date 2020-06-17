import torch

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

torch.manual_seed(args.seed)
checkpoint = utility.checkpoint(args)

if checkpoint.ok:
    loader = data.Data(args)
    model = model.Model(args, checkpoint)

    # export_onnx(h, w, batch)
    model.export_onnx(180, 160, 1)
    model.export_onnx(180, 320, 1)
    model.export_onnx(360, 320, 1)

