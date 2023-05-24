import argparse

# Set up logger
import os
import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "1"
os.environ["AUTOGRAPH_VERBOSITY"] = "1"


logger = logging.getLogger()
logger.handlers = []
ch = logging.StreamHandler()
formatter = logging.Formatter(
    fmt="%(asctime)s (%(levelname)s): %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")

import numpy as np
import yaml
import string
import ast
import random
import time
from datetime import datetime

from gemnet.model.gemnet import GemNet
from gemnet.training.trainer import Trainer
from gemnet.training.metrics import Metrics, BestMetrics
from gemnet.training.data_container import DataContainer
from gemnet.training.data_provider import DataProvider
import process

import torch
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument('--num_spherical', type=int, default=7)
parser.add_argument('--num_radial', type=int, default=6)
parser.add_argument('--num_blocks', type=int, default=4)
parser.add_argument('--emb_size_atom', type=int, default=128)
parser.add_argument('--emb_size_edge', type=int, default=128)
parser.add_argument('--emb_size_trip', type=int, default=64)
parser.add_argument('--emb_size_quad', type=int, default=32)
parser.add_argument('--emb_size_rbf', type=int, default=16)
parser.add_argument('--emb_size_cbf', type=int, default=16)
parser.add_argument('--emb_size_sbf', type=int, default=32)
parser.add_argument('--emb_size_bil_trip', type=int, default=64)
parser.add_argument('--emb_size_bil_quad', type=int, default=32)


parser.add_argument('--num_before_skip', type=int, default=1)
parser.add_argument('--num_after_skip', type=int, default=1)
parser.add_argument('--num_concat', type=int, default=1)
parser.add_argument('--num_atom', type=int, default=2)


parser.add_argument('--cutoff', type=float, default=5.0)
parser.add_argument('--int_cutoff', type=float, default=10.0)
parser.add_argument('--triplets_only', action='store_true', default=False)
parser.add_argument('--direct_forces', action='store_true', default=False)

parser.add_argument('--mve', action='store_true', default=False)
parser.add_argument('--loss', type=str, default='rmse')
parser.add_argument('--forces_coupled', action='store_true', default=False)
parser.add_argument('--envelope_exponent', type=int, default=5)
parser.add_argument('--extensive', action='store_false', default=True)

parser.add_argument('--rho_force', type=float, default=0.999)
parser.add_argument('--ema_decay', type=float, default=0.999)
parser.add_argument('--weight_decay', type=float, default=0.000002)

parser.add_argument('--learning_rate', type=float, default=1e-3)
parser.add_argument('--decay_steps', type=int, default=45000)
parser.add_argument('--decay_rate', type=float, default=0.01)
parser.add_argument('--staircase', action='store_true', default=False)
parser.add_argument('--decay_patience', type=int, default=50000)
parser.add_argument('--decay_factor', type=float, default=0.5)
parser.add_argument('--decay_cooldown', type=int, default=50000)
parser.add_argument('--agc', action='store_true', default=False)
parser.add_argument('--grad_clip_max', type=float, default=10.0)
parser.add_argument('--decay', action='store_false', default=True)

parser.add_argument('--tfseed', type=int, default=1234)
parser.add_argument('--data_seed', type=int, default=42)
parser.add_argument('--scale_file', type=str, default='scaling_factors.json')
parser.add_argument('--comment', type=str, default='GemNet')
parser.add_argument('--model_name', type=str, default='')
parser.add_argument('--output_init', type=str, default='HeOrthogonal')
parser.add_argument('--logdir', type=str, default='logs')

parser.add_argument('--patience', type=int, default=50000)
parser.add_argument('--evaluation_interval', type=int, default=1)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--warmup_steps', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_steps', type=int, default=1500000)
parser.add_argument('--n_pts', type=int, default=100000)
parser.add_argument('--molecule', type=str, default="benzene")
parser.add_argument('--loss_type', type=str, default="sliced")
parser.add_argument('--n_vectors', type=int, default=3)

parser.add_argument('--adv_training', action='store_true', default=False)
parser.add_argument('--adv_alpha', type=float, default=0.5)
parser.add_argument('--adv_eps', type=float, default=0.5)
parser.add_argument('--adv_lambda', type=float, default=0.5)

parser.add_argument('--save_model', action='store_true', default=False)
parser.add_argument('--ccsd', action='store_true', default=False)
parser.add_argument('--revised', action='store_true', default=False)
parser.add_argument('--classic', action='store_true', default=False)
args = parser.parse_args()

num_spherical = args.num_spherical
num_radial = args.num_radial
num_blocks = args.num_blocks
emb_size_atom = args.emb_size_atom
emb_size_edge = args.emb_size_edge
emb_size_trip = args.emb_size_trip
emb_size_quad = args.emb_size_quad
emb_size_rbf = args.emb_size_rbf
emb_size_cbf = args.emb_size_cbf
emb_size_sbf = args.emb_size_sbf
num_before_skip = args.num_before_skip
num_after_skip = args.num_after_skip
num_concat = args.num_concat
num_atom = args.num_atom
emb_size_bil_quad = args.emb_size_bil_quad
emb_size_bil_trip = args.emb_size_bil_trip
triplets_only = args.triplets_only
forces_coupled = args.forces_coupled
direct_forces = args.direct_forces
mve = args.mve
cutoff = args.cutoff
int_cutoff = args.int_cutoff
envelope_exponent = args.envelope_exponent
extensive = args.extensive
output_init = args.output_init
scale_file = args.scale_file
data_seed = args.data_seed
# dataset = args.dataset
test_dataset = None
logdir = args.logdir
loss = args.loss
tfseed = args.tfseed
num_steps = args.num_steps
rho_force = args.rho_force
ema_decay = args.ema_decay
weight_decay = args.weight_decay
grad_clip_max = args.grad_clip_max
agc = args.agc
decay_patience = args.decay_patience
decay_factor = args.decay_factor
decay_cooldown = args.decay_cooldown
batch_size = args.batch_size
evaluation_interval = args.evaluation_interval
patience = args.patience
save_interval = args.save_interval
learning_rate = args.learning_rate
warmup_steps = args.warmup_steps
decay_steps = args.decay_steps
decay_rate = args.decay_rate
staircase = args.staircase
#restart = args.restart
restart = None
comment = args.comment

torch.manual_seed(tfseed)

logging.info("Start training")
num_gpus = torch.cuda.device_count()
cuda_available = torch.cuda.is_available()
logging.info(f"Available GPUs: {num_gpus}")
logging.info(f"CUDA Available: {cuda_available}")
if num_gpus == 0:
    logging.warning("No GPUs were found. Training is run on CPU!")
if not cuda_available:
    logging.warning("CUDA unavailable. Training is run on CPU!")


extension = ".pth"
save_dir = "model_dir"
log_path_model = f"{save_dir}/model{args.model_name}{extension}"

logging.info("Initialize model")
model = GemNet(
    num_spherical=num_spherical,
    num_radial=num_radial,
    num_blocks=num_blocks,
    emb_size_atom=emb_size_atom,
    emb_size_edge=emb_size_edge,
    emb_size_trip=emb_size_trip,
    emb_size_quad=emb_size_quad,
    emb_size_rbf=emb_size_rbf,
    emb_size_cbf=emb_size_cbf,
    emb_size_sbf=emb_size_sbf,
    num_before_skip=num_before_skip,
    num_after_skip=num_after_skip,
    num_concat=num_concat,
    num_atom=num_atom,
    emb_size_bil_quad=emb_size_bil_quad,
    emb_size_bil_trip=emb_size_bil_trip,
    num_targets=2 if mve else 1,
    triplets_only=triplets_only,
    direct_forces=direct_forces,
    forces_coupled=forces_coupled,
    cutoff=cutoff,
    int_cutoff=int_cutoff,
    envelope_exponent=envelope_exponent,
    activation="swish",
    extensive=extensive,
    output_init=output_init,
    scale_file=scale_file,
)
# push to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train = {}
logging.info("Load dataset")


if args.revised:
    dataset = "data/{}_train_gem_revised_{}.npz".format(args.molecule, args.n_pts)
elif args.classic:
    dataset = "data/{}_train_gem_classic.npz".format(args.molecule)
else:
    dataset = "data/{}_train_gem_{}.npz".format(args.molecule, args.n_pts)


data_container = DataContainer(
    dataset, cutoff=cutoff, int_cutoff=int_cutoff, triplets_only=triplets_only
)

num_train = len(data_container)
logging.info(f"Training data size (unsup): {num_train}")
data_provider = DataProvider(
    data_container,
    num_train,
    0,
    batch_size,
    seed=data_seed,
    shuffle=True,
    random_split=False,
)


# Initialize datasets
train["dataset_iter"] = data_provider.get_dataset("train")

logging.info("Prepare training")
# Initialize trainer
trainer = Trainer(
    model,
    learning_rate=learning_rate,
    decay_steps=decay_steps,
    decay_rate=decay_rate,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    ema_decay=ema_decay,
    decay_patience=decay_patience,
    decay_factor=decay_factor,
    decay_cooldown=decay_cooldown,
    grad_clip_max=grad_clip_max,
    rho_force=rho_force,
    mve=mve,
    loss=loss,
    staircase=staircase,
    agc=agc,
)

logging.info("Restoring model and trainer")
model_checkpoint = torch.load(log_path_model)
model.load_weights(model_checkpoint["model"])
print("Loaded", args.n_pts // args.batch_size)

pred_energy, pred_force, real_pos, real_charges, real_Ns = [], [], [], [], []
real_forces = []
for i in range(args.n_pts // args.batch_size):
    E, F, R, Z, N, real_F = trainer.predict_from_numpy(train["dataset_iter"])
    pred_energy.append(E)
    pred_force.append(F)
    real_pos.append(R)
    real_charges.append(Z)
    real_Ns.append(N)
    real_forces.append(real_F)
    if i % 100 == 0:
        print("Step: {}".format(i))
        print("Psuedo Label MAE: {}".format(np.abs(F - real_F).mean()))

if args.revised:
    np.savez_compressed("data/{}_pretrain_revised_pred_{}_{}.npz".format(args.molecule, args.model_name, args.n_pts), E=np.concatenate(pred_energy), F=np.concatenate(real_forces),
                        F_pred=np.concatenate(pred_force), R=np.concatenate(real_pos), Z=np.concatenate(real_charges), N=np.concatenate(real_Ns))
elif args.classic:
    np.savez_compressed("data/{}_pretrain_classic_pred.npz".format(args.molecule), E=np.concatenate(pred_energy), F_real=np.concatenate(real_forces),
                        F_pred=np.concatenate(pred_force), R=np.concatenate(real_pos), Z=np.concatenate(real_charges), N=np.concatenate(real_Ns))
else:
    np.savez_compressed("data/{}_pretrain_pred.npz".format(args.molecule), E=np.concatenate(pred_energy), F_real=np.concatenate(real_forces),
                        F_pred=np.concatenate(pred_force), R=np.concatenate(real_pos), Z=np.concatenate(real_charges), N=np.concatenate(real_Ns))


