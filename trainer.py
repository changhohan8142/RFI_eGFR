#!/usr/bin/env python
# eGFR regression from retinal fundus images — DDP trainer (single-task)
#
# Trains a ViT-based model (backbone + linear eGFR head) using PyTorch DDP.
# Supports three fine-tuning modes: LoRA, partial (last N blocks), and linear probe.
# Data is split 7:1:2 by patient ID; val MSE is used for early stopping and LR halving.
#
# Usage (example, 2 GPUs):
#   torchrun --nproc_per_node=2 trainer.py --arch dinov3 --ft lora --lora_rank 4 --ft_blks full
#
# Key arguments:
#   --arch        : backbone architecture (retfound, retfound_dinov2, mae, openclip, dinov2, dinov3)
#   --ft          : fine-tuning mode (linear, partial, lora)
#   --csv         : master CSV with columns pngfilename, gfr_updated, PAT_ID, STDY_AGE
#   --img_dir     : directory containing AutoMorph M0 fundus PNGs
#   --good_dir    : directory containing AutoMorph M1 good-quality PNGs (used as inclusion filter)
#   --out_dir     : output directory for inference CSVs and performance summary
#   --split_dir   : output directory for train/val/test split CSVs
#   --root_dir    : output directory for model checkpoints


import os, random, warnings, argparse
from pathlib import Path
from functools import partial
import time

import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm, trange

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data as udata
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
import torchvision.transforms as tfms

from sklearn.metrics import roc_auc_score

from models.encoder import build_model

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_DEVICE_ORDER",             "PCI_BUS_ID")
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT",      "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING","1")
os.environ.setdefault("NCCL_TIMEOUT",                  "7200")
ImageFile.LOAD_TRUNCATED_IMAGES = True

GLOBAL_SEED = 43


# -------------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------------
def set_seed(seed=GLOBAL_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark     = True


@torch.no_grad()
def concat_all_gather(tensor):
    tensors_gather = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(tensors_gather, tensor, async_op=False)
    return torch.cat(tensors_gather, dim=0)


def pearson_r(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def compute_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)

    def safe_auroc(thr):
        y_bin = (y_true < thr).astype(int)
        try:
            return float(roc_auc_score(y_bin, thr - y_pred))
        except Exception:
            return np.nan

    return dict(
        mse          = float(np.mean((y_true - y_pred) ** 2)),
        pearson_r    = pearson_r(y_true, y_pred),
        auroc_under60= safe_auroc(60),
        auroc_under90= safe_auroc(90),
        mean_pred    = float(y_pred.mean()),
        mean_true    = float(y_true.mean()),
    )


# -------------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------------
class EGFRImageDS(Dataset):
    """Fundus image dataset for eGFR regression.

    Returns (image_tensor, egfr_target, sample_index).
    """

    def __init__(self, df_part, img_sz, kind, hflip_p=0.5, strong_aug=False):
        self.df  = df_part.reset_index(drop=True)
        norm     = tfms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        if kind == "train":
            t = ([tfms.RandomResizedCrop((img_sz, img_sz)),
                  tfms.AutoAugment(),
                  tfms.RandomHorizontalFlip(p=hflip_p)]
                 if strong_aug else
                 [tfms.Resize((img_sz, img_sz)),
                  tfms.RandomHorizontalFlip(p=hflip_p)])
            self.tfm = tfms.Compose(t + [tfms.ToTensor(), norm])
        else:
            self.tfm = tfms.Compose([tfms.Resize((img_sz, img_sz)), tfms.ToTensor(), norm])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r   = self.df.iloc[idx]
        img = self.tfm(Image.open(r.pngfilename).convert("RGB"))
        return img, torch.tensor(float(r.gfr_updated), dtype=torch.float32), idx


# -------------------------------------------------------------------------
# Model — ViT backbone + single eGFR regression head
# -------------------------------------------------------------------------
class EgfrOnlyModel(nn.Module):
    """Wraps build_model() backbone with a single linear eGFR head."""

    def __init__(self, base_model, feat_dim=None, init_mu=None):
        super().__init__()
        self.base = base_model

        D = (feat_dim
             or getattr(self.base, "num_features", None)
             or getattr(self.base, "embed_dim", None)
             or getattr(getattr(self.base, "head", nn.Identity()), "in_features", None)
             or 768)

        self.head_egfr = nn.Linear(D, 1)
        if init_mu is not None:
            with torch.no_grad():
                self.head_egfr.bias.data.fill_(init_mu)
                self.head_egfr.weight.data.zero_()

    def forward_features(self, x):
        feats = (self.base.forward_features(x)
                 if hasattr(self.base, "forward_features")
                 else self.base(x))

        if isinstance(feats, dict):
            preferred = ("x_norm_clstoken", "cls_token", "pooled", "last_hidden_state",
                         "features", "x_norm_patchtokens", "patch_tokens", "tokens")
            chosen = next((feats[k] for k in preferred if k in feats and feats[k] is not None), None)
            if chosen is None:
                chosen = next((v for v in feats.values() if torch.is_tensor(v)), None)
            if chosen is None:
                raise TypeError("Could not extract tensor features from backbone dict output.")
            feats = chosen

        if torch.is_tensor(feats) and feats.ndim == 3:
            feats = feats[:, 0]
        return feats

    def forward(self, x):
        return {"egfr": self.head_egfr(self.forward_features(x)).squeeze(1)}


# -------------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------------
class EGFRTrainer:
    def __init__(self, args):
        self.args = args
        self._init_distributed_mode()
        set_seed()
        self._load_csv_and_split()
        self._build_loaders()
        self._build_model_opt()
        self._init_io()

    def _init_distributed_mode(self):
        dist.init_process_group(backend="nccl")
        dist.barrier()
        self.world_size  = dist.get_world_size()
        self.global_rank = dist.get_rank()
        self.local_rank  = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(self.local_rank)
        self.device     = torch.cuda.current_device()
        self.enable_amp = self.args.enable_amp
        self.scaler     = torch.cuda.amp.GradScaler(enabled=self.enable_amp)

    def _load_csv_and_split(self):
        args = self.args

        if self.global_rank == 0:
            os.makedirs(args.out_dir,   exist_ok=True)
            os.makedirs(args.split_dir, exist_ok=True)
        dist.barrier()

        df = pd.read_csv(args.csv, low_memory=False)

        def sanitize(fn):
            b, e = os.path.splitext(str(fn).strip())
            return b.replace(".", "_") + e

        df["pngfilename"] = args.img_dir + df["pngfilename"].apply(sanitize)

        # keep only images present in img_dir
        valid_png = {f for f in os.listdir(args.img_dir) if f.lower().endswith(".png")}
        df = df[df["pngfilename"].apply(lambda p: os.path.basename(p) in valid_png)].reset_index(drop=True)

        # keep only images that passed AutoMorph M1 quality filter
        good_png = {f for f in os.listdir(args.good_dir) if f.lower().endswith(".png")}
        df = df[df["pngfilename"].apply(lambda p: os.path.basename(p) in good_png)].reset_index(drop=True)

        # coerce numeric columns and drop rows missing required fields
        for col in ("gfr_updated", "STDY_AGE"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if "STDY_DT" in df.columns:
            df["STDY_DT"] = pd.to_datetime(df["STDY_DT"], errors="coerce")

        df = df.dropna(subset=["gfr_updated", "pngfilename", "PAT_ID"]).reset_index(drop=True)

        # patient-level 7:1:2 split
        pat_ids = df["PAT_ID"].dropna().astype(str).unique()
        rng = np.random.RandomState(GLOBAL_SEED)
        rng.shuffle(pat_ids)
        n_tr = int(round(len(pat_ids) * 0.7))
        n_va = int(round(len(pat_ids) * 0.1))

        tr_pats = set(pat_ids[:n_tr])
        va_pats = set(pat_ids[n_tr:n_tr + n_va])
        te_pats = set(pat_ids[n_tr + n_va:])

        self.df_train = df[df.PAT_ID.astype(str).isin(tr_pats)].copy()
        self.df_val   = df[df.PAT_ID.astype(str).isin(va_pats)].copy()
        self.df_test  = df[df.PAT_ID.astype(str).isin(te_pats)].copy()
        self.out_dir  = args.out_dir

        if self.global_rank == 0:
            self.df_train.to_csv(os.path.join(args.split_dir, "train.csv"), index=False)
            self.df_val.to_csv(  os.path.join(args.split_dir, "val.csv"),   index=False)
            self.df_test.to_csv( os.path.join(args.split_dir, "test.csv"),  index=False)
            for name, d in [("train", self.df_train), ("val", self.df_val), ("test", self.df_test)]:
                print(f"[split] {name}: N_images={len(d)}, N_patients={d['PAT_ID'].astype(str).nunique()}")

        # training-set mean used for eGFR head bias initialization
        self.egfr_mu = float(self.df_train["gfr_updated"].mean())

    def _name_suffix(self):
        if self.args.ft == "lora":
            return f"{self.args.arch}_{self.args.ft}_rank{self.args.lora_rank}_ft{self.args.ft_blks}"
        if self.args.ft == "partial":
            return f"{self.args.arch}_{self.args.ft}_ft{self.args.ft_blks}"
        return f"{self.args.arch}_{self.args.ft}"

    def _build_loaders(self):
        dist_sampler = partial(udata.distributed.DistributedSampler,
                               num_replicas=self.world_size, rank=self.global_rank)

        self.train_dataset = EGFRImageDS(self.df_train, self.args.img_size, "train",
                                         hflip_p=self.args.hflip_p, strong_aug=self.args.strong_aug)
        self.val_dataset   = EGFRImageDS(self.df_val,   self.args.img_size, "valid")
        self.test_dataset  = EGFRImageDS(self.df_test,  self.args.img_size, "test")

        self.train_sampler = dist_sampler(dataset=self.train_dataset, shuffle=True,  drop_last=True)
        self.val_sampler   = dist_sampler(dataset=self.val_dataset,   shuffle=False)
        self.test_sampler  = dist_sampler(dataset=self.test_dataset,  shuffle=False)

        self.train_loader = udata.DataLoader(
            self.train_dataset, sampler=self.train_sampler,
            batch_size=self.args.batch_size, num_workers=self.args.num_workers,
            pin_memory=True, drop_last=True,
            persistent_workers=(self.args.num_workers > 0),
        )
        for attr, ds, sampler in [("val_loader",  self.val_dataset,  self.val_sampler),
                                   ("test_loader", self.test_dataset, self.test_sampler)]:
            setattr(self, attr, udata.DataLoader(
                ds, sampler=sampler, batch_size=4,
                num_workers=self.args.eval_workers, pin_memory=False, persistent_workers=False,
            ))

    def _build_model_opt(self):
        base  = build_model(self.args).cuda()
        model = EgfrOnlyModel(base_model=base, init_mu=self.egfr_mu).cuda()

        self.model = DistributedDataParallel(
            model, device_ids=[self.device],
            find_unused_parameters=True, static_graph=False,
        )

        head_params = list(self.model.module.head_egfr.parameters())
        for p in head_params:
            p.requires_grad = True
        body_params = [p for p in self.model.module.base.parameters() if p.requires_grad]

        param_groups = []
        if head_params:
            param_groups.append({"params": head_params, "lr": self.args.lr_head, "weight_decay": 0.0})
        if body_params:
            param_groups.append({"params": body_params, "lr": self.args.lr_body,
                                  "weight_decay": self.args.weight_decay})
        if not param_groups:
            param_groups = [{"params": [p for p in self.model.parameters() if p.requires_grad],
                             "lr": self.args.lr_head, "weight_decay": self.args.weight_decay}]

        self.optimizer  = optim.AdamW(param_groups)
        self.huber_main = nn.SmoothL1Loss(beta=5)

    def _init_io(self):
        self.desc = f"{self.args.arch}_{self.args.ft}"
        if self.args.ft == "lora":
            self.desc += f"_rank_{self.args.lora_rank}_ft_{self.args.ft_blks}"
        elif self.args.ft == "partial":
            self.desc += f"_ft_{self.args.ft_blks}"
        self.log_dir = Path(self.args.root_dir) / self.desc
        if self.global_rank == 0:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_path = self.log_dir / "ckpt.pth.tar"

    def save(self):
        if self.global_rank == 0:
            tmp = self.ckpt_path.with_suffix(".pt.tmp")
            torch.save(self.model.module.state_dict(), tmp, _use_new_zipfile_serialization=False)
            os.replace(tmp, self.ckpt_path)

    def load(self):
        ckpt = torch.load(self.ckpt_path, map_location=f"cuda:{self.device}")
        self.model.module.load_state_dict(ckpt, strict=True)

    # -------------------------------------------------------------------------
    # Train / Eval
    # -------------------------------------------------------------------------
    def train_one_epoch(self, epoch):
        self.train_sampler.set_epoch(epoch)
        self.model.train()
        accum = torch.tensor([0.0], device=self.device)

        for x, y, _ in tqdm(self.train_loader, desc=f"train[{epoch}]",
                             total=len(self.train_loader), disable=self.global_rank != 0):
            self.optimizer.zero_grad(set_to_none=True)
            x = x.cuda(self.local_rank, non_blocking=True)
            y = y.cuda(self.local_rank, non_blocking=True)

            with torch.amp.autocast(device_type="cuda", enabled=self.enable_amp):
                out  = self.model(x)
                loss = self.huber_main(out["egfr"], y)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            accum += loss.detach()

        mean_loss = (concat_all_gather(accum).mean() / len(self.train_loader)).item()
        if self.global_rank == 0:
            print(f"Epoch {epoch}: train loss {mean_loss:.4f} | lr={self.optimizer.param_groups[0]['lr']:.2e}")

    @torch.no_grad()
    def _eval_loader(self, loader, sampler, tag):
        self.model.eval()
        sampler.set_epoch(0)
        preds_l, trues_l, idxs_l = [], [], []

        for x, y, idx in tqdm(loader, desc=tag, total=len(loader), disable=self.global_rank != 0):
            x = x.cuda(self.local_rank, non_blocking=True)
            y = y.cuda(self.local_rank, non_blocking=True)
            with torch.amp.autocast(device_type="cuda", enabled=self.enable_amp):
                p = self.model(x)["egfr"]
            preds_l.append(p.detach().cpu())
            trues_l.append(y.detach().cpu())
            idxs_l.append(idx.detach().cpu())

        to_np = lambda lst: torch.cat(lst).numpy() if lst else np.zeros(0, dtype=np.float32)
        payload  = {"preds": to_np(preds_l), "trues": to_np(trues_l), "idxs": to_np(idxs_l)}
        gathered = [None] * self.world_size if self.global_rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)

        m = ordered_true = ordered_pred = None
        if self.global_rank == 0:
            all_preds = np.concatenate([g["preds"] for g in gathered])
            all_trues = np.concatenate([g["trues"] for g in gathered])
            all_idxs  = np.concatenate([g["idxs"]  for g in gathered])

            if all_idxs.size > 0:
                n = int(all_idxs.max()) + 1
                ordered_pred, ordered_true = np.empty(n), np.empty(n)
                ordered_pred[all_idxs] = all_preds
                ordered_true[all_idxs] = all_trues
            else:
                ordered_pred, ordered_true = all_preds, all_trues

            m = compute_metrics(ordered_true, ordered_pred)
            print(f"\t{tag}: MSE {m['mse']:.2f} | r {m['pearson_r']:.3f} "
                  f"| AUROC<60 {m['auroc_under60']:.3f} | AUROC<90 {m['auroc_under90']:.3f} "
                  f"| mean_pred {m['mean_pred']:.2f} | mean_true {m['mean_true']:.2f}")

        m_list = [m]
        dist.broadcast_object_list(m_list, src=0)
        return m_list[0], ordered_true, ordered_pred

    def train(self):
        best_val_mse = np.inf
        stalls       = 0
        for epoch in trange(self.args.epochs, desc="Epochs", disable=self.global_rank != 0):
            self.train_one_epoch(epoch)
            val_m, _, _ = self._eval_loader(self.val_loader, self.val_sampler, "VAL")

            if val_m["mse"] < (best_val_mse - 1e-1):
                best_val_mse = val_m["mse"]
                stalls = 0
                self.save()
            else:
                stalls += 1
                if stalls == self.args.lr_halve_patience:
                    for g in self.optimizer.param_groups:
                        g["lr"] *= 0.5
                    if self.global_rank == 0:
                        print("  ↳ LR halved")
                if stalls >= self.args.early_stop_patience:
                    if self.global_rank == 0:
                        print("  ↳ Early stop.")
                    break

    def inference(self):
        self.load()
        val_m,  val_true,  val_pred  = self._eval_loader(self.val_loader,  self.val_sampler,  "VAL-final")
        test_m, test_true, test_pred = self._eval_loader(self.test_loader, self.test_sampler, "TEST-final")

        if self.global_rank == 0:
            tag       = self._name_suffix()
            base_cols = ["PAT_ID", "pngfilename", "STDY_DT", "STDY_AGE", "gfr_updated"]

            for name, df_split, preds, trues in [("val",  self.df_val,  val_pred,  val_true),
                                                  ("test", self.df_test, test_pred, test_true)]:
                if preds is None or trues is None:
                    continue
                df_out = df_split.reset_index(drop=True).copy()
                cols   = [c for c in base_cols if c in df_out.columns]
                df_out = df_out.assign(pred_egfr=preds, true_egfr=trues)[cols + ["pred_egfr", "true_egfr"]]
                df_out.to_csv(Path(self.out_dir) / f"infout_{name}_{tag}_egfr_trainer.csv", index=False)

            row = dict(
                version="egfr_trainer_single_task",
                name_tag=tag, arch=self.args.arch, ft=self.args.ft,
                lora_rank=(self.args.lora_rank if self.args.ft == "lora" else None),
                ft_blks=(self.args.ft_blks if self.args.ft in ["lora", "partial"] else None),
                batch_size=self.args.batch_size, best_ckpt=str(self.ckpt_path.name),
                val_mse=val_m["mse"],   val_pearson_r=val_m["pearson_r"],
                val_auroc_under60=val_m["auroc_under60"],
                val_auroc_under90=val_m["auroc_under90"],
                val_mean_pred=val_m["mean_pred"],   val_mean_true=val_m["mean_true"],
                test_mse=test_m["mse"], test_pearson_r=test_m["pearson_r"],
                test_auroc_under60=test_m["auroc_under60"],
                test_auroc_under90=test_m["auroc_under90"],
                test_mean_pred=test_m["mean_pred"], test_mean_true=test_m["mean_true"],
            )
            perf_csv = Path(self.out_dir) / "model_performance_summary_egfr_trainer.csv"
            pd.concat([pd.read_csv(perf_csv), pd.DataFrame([row])] if perf_csv.exists()
                      else [pd.DataFrame([row])]).to_csv(perf_csv, index=False)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="eGFR DDP trainer — single task")
    p.add_argument("--arch",    type=str, default="retfound",
                   choices=["retfound", "retfound_dinov2", "mae", "openclip", "dinov2", "dinov3"])
    p.add_argument("--ft",      type=str, default="lora",
                   choices=["linear", "partial", "lora"])
    p.add_argument("--lora_rank", type=int,   default=4)
    p.add_argument("--ft_blks",   type=str,   default="4")
    # ---- paths (update to your environment) ----
    p.add_argument("--csv",       type=str, default="/path/to/master.csv")
    p.add_argument("--img_dir",   type=str, default="/path/to/AutoMorph/M0/images/")
    p.add_argument("--good_dir",  type=str, default="/path/to/AutoMorph/M1/Good_quality/")
    p.add_argument("--out_dir",   type=str, default="./models_egfr")
    p.add_argument("--split_dir", type=str, default="./splits_egfr")
    p.add_argument("--root_dir",  type=str, default="./ckpts_egfr")
    # ---- training ----
    p.add_argument("--img_size",   type=int,   default=448)
    p.add_argument("--epochs",     type=int,   default=100)
    p.add_argument("--batch_size", type=int,   default=2)
    p.add_argument("--num_workers",   type=int, default=4)
    p.add_argument("--eval_workers",  type=int, default=0)
    p.add_argument("--lr_head",       type=float, default=1e-2)
    p.add_argument("--lr_body",       type=float, default=2e-3)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--enable_amp",    action="store_true")
    p.add_argument("--lr_halve_patience",  type=int, default=4)
    p.add_argument("--early_stop_patience",type=int, default=7)
    p.add_argument("--strong_aug",    action="store_true", default=True)
    p.add_argument("--hflip_p",       type=float, default=0.5)
    p.add_argument("--huber_delta",   type=float, default=5.0)

    args = p.parse_args()
    args.ft_blks = int(args.ft_blks) if args.ft_blks.isdigit() else args.ft_blks.lower()

    trainer = EGFRTrainer(args)
    trainer.train()
    time.sleep(10)
    trainer.inference()


if __name__ == "__main__":
    main()