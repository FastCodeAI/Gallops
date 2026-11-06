"""CPU shape check of the trainer's model, inputs and loss, while the GPU is full."""
import os, sys, time
import numpy as np, torch
import train_b2d as TB
D2, G2 = TB.D2, TB.G2
torch.set_num_threads(8); dev = 'cpu'
t0 = time.time(); T = D2.Tables('train', 4); print('frames', len(T), round(time.time() - t0, 1), 's', flush=True)
ag, cfg = TB.build(None, dev); ag.train(); print('built', round(time.time() - t0, 1), flush=True)
n = sum(p.numel() for p in ag._pad_model.parameters()); nb = sum(p.numel() for p in ag._pad_model._backbone.parameters()); print('params', n, 'backbone', nb)
ds = D2.Samples(T); b = D2.collate([ds[i] for i in (3, 40, 100)]); print({k: tuple(v.shape) for k, v in b.items() if torch.is_tensor(v)}, 'teachers', [len(x) for x in b['teachers']])
print('ego', b['ego'][0, 0].tolist())
tg = TB.targets_of(b, dev); pred = ag.forward(TB.features(b, T, dev)); print({k: (tuple(v.shape) if torch.is_tensor(v) else (len(v) if isinstance(v, list) else v)) for k, v in pred.items()}, flush=True)
ld = ag.pad_loss(tg, pred, cfg); print({k: round(float(v), 4) for k, v in ld.items()}, round(time.time() - t0, 1), flush=True)
ld['loss'].backward(); g = [p for p in ag._pad_model.parameters() if p.grad is not None]; print('params with grad', sum(p.numel() for p in g), 'of', n)
opt = ag.get_optimizers(); print('groups', [(len(q['params']), q['lr']) for q in opt.param_groups])
print('SMOKE_OK')
