"""Training loop (identical for the main experiments, the ablation and the single-source runs)."""
import time

import numpy as np
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader

from . import config as C
from .config import BS, CFG, DEVICE, EPOCHS, LR
from .data import DS
from .decoding import decode_free
from .evaluate import evaluate, infer
from .model import Net, loss_fn
from .utils import set_seed


@torch.no_grad()
def _val_free_nme(model, ds):
    model.eval()
    K, v = ds.K, []
    for b in DataLoader(ds, 32, shuffle=False, num_workers=2):
        P = decode_free(infer(model, b["img"].to(DEVICE), False)[:, :K])
        G = b["pts"].numpy()
        gl = np.linalg.norm(G[:, :, 0] - G[:, :, 1], axis=-1)
        d = np.minimum(np.linalg.norm(P - G, axis=-1).sum(-1), np.linalg.norm(P - G[:, :, ::-1], axis=-1).sum(-1))
        v += list((d / (2 * gl + 1e-6)).mean(1))
    return float(np.median(v))


def train_model(tr, anat, seed, augment=True, verbose=True, mask_head=True, frac=1.0, select="constrained"):
    """Train one model on dataframe `tr` (the subject-grouped 10% validation split is made here).

    Recipe: AdamW (wd 1e-4), OneCycle peak LR 3e-4, batch 16, 40 epochs, grad-clip 5.0, AMP.
    The checkpoint is the epoch with the lowest validation median NME after a 10-epoch warm-up.

    Options for the supplementary studies (defaults reproduce the main experiments):
      mask_head  False trains without the auxiliary mask channel (no constrained decoding possible);
      frac       <1 keeps a random subset of training *subjects* (validation split unchanged);
      select     "constrained" (main runs) or "free": which decoding's validation NME picks the checkpoint
                 (the 2x2 / size-shift study uses "free" for every variant so selection is identical).

    Returns (model with best weights loaded, best_epoch, train_minutes, n_train).
    """
    set_seed(seed)
    c = CFG[anat]
    K = len(c["meas"])
    use_aux = bool(c["aux"] and mask_head)
    nc = K + int(use_aux)
    gi, vi = next(GroupShuffleSplit(1, test_size=.1, random_state=seed).split(tr, groups=tr.group))
    if frac < 1:
        g = tr.group.iloc[gi]
        u = np.array(sorted(g.unique()))
        keep = set(np.random.RandomState(seed).choice(u, max(2, int(round(frac * len(u)))), replace=False))
        gi = gi[g.isin(keep).values]
    dtr = DS(tr.iloc[gi], anat, True)
    if not augment:
        dtr.train = False          # the entire augmentation ablation: everything else identical
    dva = DS(tr.iloc[vi], anat, False)
    TL = DataLoader(dtr, BS, shuffle=True, num_workers=3, drop_last=True, persistent_workers=True)
    model = Net(nc).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=C.WEIGHT_DECAY)
    sch = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=EPOCHS * len(TL))
    amp = DEVICE == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    warmup = max(10, EPOCHS // 4)
    best, bst, be, t0 = 1e9, None, 0, time.time()
    for ep in range(EPOCHS):
        model.train()
        tl = 0
        for b in TL:
            x, h = b["img"].to(DEVICE), b["hm"].to(DEVICE)
            opt.zero_grad()
            with torch.autocast("cuda", enabled=amp):
                o = model(x)
            loss = loss_fn(o.float(), h, K, use_aux)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP)
            scaler.step(opt)
            scaler.update()
            sch.step()
            tl += loss.item()
        v = _val_free_nme(model, dva) if select == "free" else evaluate(model, dva, False).nme.median()
        if ep + 1 >= warmup and v < best:
            best, be = v, ep + 1
            bst = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        if verbose and (ep + 1) % 10 == 0:
            print(f"   ep{ep + 1:02d} loss {tl / len(TL):.3f} valNME(med) {v:.4f}")
    if bst is None:
        bst = {k: t.detach().cpu().clone() for k, t in model.state_dict().items()}
        be = EPOCHS
    model.load_state_dict(bst)
    return model, be, (time.time() - t0) / 60, len(dtr)
