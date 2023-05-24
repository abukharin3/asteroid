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
parser.add_argument('--decay_steps', type=int, default=1200000)
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

parser.add_argument('--dataset', type=str, default='data/benzene_train_gem.npz')
parser.add_argument('--val_dataset', type=str, default='data/benzene_val_gem.npz')
parser.add_argument('--num_val', type=int, default=0)

parser.add_argument('--patience', type=int, default=50000)
parser.add_argument('--evaluation_interval', type=int, default=100)
parser.add_argument('--save_interval', type=int, default=100)
parser.add_argument('--warmup_steps', type=int, default=10000)
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_steps', type=int, default=15000000)
parser.add_argument('--n_labels', type=int, default=200)
parser.add_argument('--n_pts', type=int, default=100000)
parser.add_argument('--molecule', type=str, default="benzene")
parser.add_argument('--loss_type', type=str, default="sliced")
parser.add_argument('--n_vectors', type=int, default=1)

parser.add_argument('--load_model', action='store_true', default=False)
parser.add_argument('--save_model', action='store_true', default=False)
parser.add_argument('--pretrain', action='store_true', default=False)
parser.add_argument('--data_parallel', action='store_true', default=False)
parser.add_argument('--ccsd', action='store_true', default=False)
parser.add_argument('--revised', action='store_true', default=False)
parser.add_argument('--decay_rate_lam', type=float, default=0.9995)
parser.add_argument('--noise_magnitude', type=float, default=1)
parser.add_argument('--sigma', type=float, default=0.5)
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
val_dataset = args.val_dataset
test_dataset = None
num_val = args.num_val
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

# Used for creating a "unique" id for a run (almost impossible to generate the same twice)
def id_generator(
	size=6, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits
):
	return "".join(random.SystemRandom().choice(chars) for _ in range(size))

# A unique directory name is created for this run based on the input
if (restart is None) or (restart == "None"):
	directory = (
		logdir
		+ "/"
		+ str(args.molecule)
		+ "_"
		+ str(comment)
		+ "_sm"
		+ str(args.loss_type)
		+ "semi" 
		+ "_" + str(args.num_spherical)
		+ "_" + str(args.num_blocks)
		+ "_" + str(args.num_radial)
		+ "_" + str(args.emb_size_atom)
		+ "_" + str(args.direct_forces)
		+ "_" + str(args.n_pts)
		+ "_pretrain"
	)
else:
	directory = restart

logging.info(f"Directory: {directory}")
logging.info("Create directories")

if not os.path.exists(directory):
	os.makedirs(directory, exist_ok=True)

best_dir = os.path.join(directory, "best")
if not os.path.exists(best_dir):
	os.makedirs(best_dir)
log_dir = os.path.join(directory, "logs")
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

save_dir = "model_dir"
if not os.path.exists(log_dir):
	os.makedirs(log_dir)

extension = ".pth"
log_path_model = f"{save_dir}/model{args.model_name}{extension}"
log_path_training = f"{save_dir}/model{args.comment}_finetune{extension}"
best_path_model = f"{save_dir}/model{args.model_name}{extension}"


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_sup = {}
validation = {}
test = {}

logging.info("Load dataset")

if args.ccsd:
	dataset_sup = "data/{}_train_gem_ccsd_{}.npz".format(args.molecule, args.n_labels)
	val_dataset = "data/{}_val_gem_ccsd_{}.npz".format(args.molecule, args.n_labels)
	test_dataset = "data/{}_test_gem_ccsd_{}.npz".format(args.molecule, args.n_labels)
	if not os.path.exists(dataset_sup):
		process.process_ccsd(args.molecule, args.n_labels)
elif args.revised:
	dataset_sup = "data/{}_train_gem_sup_revised_{}_{}.npz".format(args.molecule, args.n_labels, args.n_pts)
	val_dataset = 'data/{}_val_gem_revised_{}.npz'.format(args.molecule, args.n_pts)
	test_dataset = 'data/{}_test_gem_revised_{}.npz'.format(args.molecule, args.n_pts)
	if not os.path.exists(dataset_sup):
		process.get_dataset(args.molecule, args.n_labels, args.n_pts, revised=True)
else:
	dataset_sup = "data/{}_train_gem_sup_{}_{}.npz".format(args.molecule, args.n_labels, args.n_pts)
	val_dataset = 'data/{}_val_gem_{}.npz'.format(args.molecule, args.n_pts)
	test_dataset = 'data/{}_test_gem_{}.npz'.format(args.molecule, args.n_pts)

data_container_sup = DataContainer(
	dataset_sup, cutoff=cutoff, int_cutoff=int_cutoff, triplets_only=triplets_only
)

# Initialize DataProvider
num_train_sup = len(data_container_sup)
logging.info(f"Training data size (sup): {num_train_sup}")
data_provider_sup = DataProvider(
	data_container_sup,
	num_train_sup,
	0,
	batch_size,
	seed=data_seed,
	shuffle=True,
	random_split=True,
)

# Initialize validation datasets
val_data_container = DataContainer(
	val_dataset,
	cutoff=cutoff,
	int_cutoff=int_cutoff,
	triplets_only=triplets_only,
)
if num_val == 0:
	num_val = len(val_data_container)
logging.info(f"Validation data size: {num_val}")
val_data_provider = DataProvider(
	val_data_container,
	0,
	num_val,
	batch_size,
	seed=data_seed,
	shuffle=True,
	random_split=True,
)

test_data_container = DataContainer(
		test_dataset,
		cutoff=cutoff,
		int_cutoff=int_cutoff,
		triplets_only=triplets_only,
)

test_data_provider = DataProvider(
	test_data_container,
	0,
	num_val,
	batch_size,
	seed=data_seed,
	shuffle=True,
	random_split=True,
)

# Initialize datasets
train_sup["dataset_iter"] = data_provider_sup.get_dataset("train")
validation["dataset_iter"] = val_data_provider.get_dataset("val")
test["dataset_iter"] = test_data_provider.get_dataset("val")

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

# Initialize metrics
train_sup["metrics"] = Metrics("train", trainer.tracked_metrics)
validation["metrics"] = Metrics("val", trainer.tracked_metrics)
test["metrics"] = Metrics("test", trainer.tracked_metrics)

# Save/load best recorded loss (only the best model is saved)
metrics_best = BestMetrics(best_dir, validation["metrics"])

# Set up checkpointing
# Restore latest checkpoint
if args.load_model:
	logging.info("Restoring model and trainer")
	print("saving", args.save_model)
	model_checkpoint = torch.load(log_path_model)
	model.load_weights(model_checkpoint["model"])
else:
	logging.info("Freshly initialize model")
metrics_best.inititalize()
step_init = 0
epoch=1
summary_writer = SummaryWriter(log_dir)

steps_per_epoch = int(np.ceil(num_train_sup / batch_size))

best_val = 10 ** 32
for step in range(step_init + 1, num_steps + 1):
	# Perform training step
	trainer.train_on_batch(train_sup['dataset_iter'], train_sup['metrics'], args, step=epoch)

	# Check performance on the validation set
	if step % evaluation_interval == 0:
		# Save backup variables and load averaged variables
		trainer.save_variable_backups()
		trainer.load_averaged_variables()

		for i in range(len(val_data_container) // batch_size):
			trainer.test_on_batch(test['dataset_iter'], test["metrics"])

		for i in range(len(test_data_container) // batch_size):
			trainer.test_on_batch(validation['dataset_iter'], validation["metrics"])

		if validation['metrics'].result()['force_mae_val'] < best_val and args.save_model:
			best_val = validation['metrics'].result()['force_mae_val']
			torch.save({"model": model.state_dict()}, log_path_training)

		# write to summary writer
		metrics_best.write(summary_writer, step)

		epoch = step // steps_per_epoch
		train_metrics_res = train_sup["metrics"].result(append_tag=False)
		val_metrics_res = validation["metrics"].result(append_tag=False)
		test_metrics_res = test["metrics"].result(append_tag=False)
		metrics_strings = [
			f"{key}: train={train_metrics_res[key]:.6f}, val={val_metrics_res[key]:.6f}, test={test_metrics_res[key]:.6f}"
			for key in validation["metrics"].keys
		]
		logging.info(
			f"{step}/{num_steps} (epoch {epoch}): " + "; ".join(metrics_strings)
		)

		# decay learning rate on plateau
		if args.decay:
			trainer.decay_maybe(validation["metrics"].loss)

		train_sup["metrics"].write(summary_writer, step)
		validation["metrics"].write(summary_writer, step)
		test["metrics"].write(summary_writer, step)
		train_sup["metrics"].reset_states()
		validation["metrics"].reset_states()
		test["metrics"].reset_states()

		# Restore backup variables
		trainer.restore_variable_backups()




