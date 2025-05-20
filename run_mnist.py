from absl import app
from absl import flags
from absl import logging

import jax
from ml_collections import config_flags
from flax.training import checkpoints, train_state
from flax import linen as nn
import optax
import jax.numpy as jnp
from functools import partial
from utils import TrainableModel, SDTrainState
from sd_loss import sd_2nd_cdf, mean_risk
# replay buffer
import flashbax as fbx
from collections import defaultdict

import train
import prepare 
from time import sleep

flags.DEFINE_string('workdir', 'tmp/mnist', 'Directory to store model data.')
config_flags.DEFINE_config_file(
    'config',
    'configs/default_mnist.py',
    'File path to the training hyperparameter configuration.',
    lock_config=True,
)
FLAGS = flags.FLAGS

import numpy as np
import torch
import torch.utils.data as data
import torchvision
from torchvision.datasets import MNIST
from torchvision import transforms

from os.path import abspath

# Transformations applied on each image => bring them into a numpy array and normalize between -1 and 1
def image_to_numpy(img):
    img = np.array(img, dtype=np.float32)
    img = (img / 255. - 0.5) / 0.5
    return img

# We need to stack the batch elements as numpy arrays
def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def get_dataloader(config):
  train_dataset = MNIST(root=config.dataset_path, train=True, transform=image_to_numpy, download=True) #  60,000 examples, @ batch=128 -> 468 batches

  test_set = MNIST(root=config.dataset_path, train=False, transform=image_to_numpy, download=True) # 10,000 examples, @ batch=128 -> 78 batches
  # pin_memory: If True, the data loader will copy tensors into CUDA pinned memory before returning them.
  train_loader = data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=numpy_collate, pin_memory=False)

  test_loader  = data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False, drop_last=False, collate_fn=numpy_collate, 
            pin_memory=jax.default_backend()!="cpu")
  
  return train_loader, test_loader


class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x = jnp.expand_dims(x, axis=3)
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.Dense(features=256)(x)
    x = nn.relu(x)
    x = nn.Dense(features=10)(x)
    return x

class Trainer(TrainableModel):
  def __init__(self, config):
    super(Trainer, self).__init__(config)
    self.rng = jax.random.key(config.seed)
    self.model=CNN()
    self.state = None
    # Item Buffer is a simple buffer that stores individual items. 
    # It is useful for storing data that is independent of each other, such as 
    # (observation, action, reward, discount, next_observation) tuples, or entire episodes.
    self.buffer = fbx.make_item_buffer(**config.buffer_args) # create replay buffer, sample size is batch_size
    self.create_fn()

  def create_train_state(self, batch):
    """Creates initial `TrainState`."""
    """
    When we add parameters, optimiser states, and a bunch of other metrics to the return call of train_step 
    it gets a bit unwieldy to handle all the state. 
    It could get worse if we later need a more complex state. 
    One solution would be to return a namedtuple so we can at least package the state together somewhat. 
    However, Flax provides its own solution, flax.training.train_state.TrainState, which has some extra functions 
    that make updating the combined state (model and optimiser state) easier.
    """
    self.rng, init_rng = jax.random.split(self.rng)
    imgs, labels = batch
    """
    init takes as first argument either a single PRNGKey, or a dictionary mapping variable collections names to their PRNGKeys, 
    and will call method (which is the module’s __call__ function by default) passing *args and **kwargs, and returns a dictionary 
    of initialized variables.
    If you pass a single PRNGKey, Flax will use it to feed the 'params' RNG stream. If you want to use a different RNG stream 
    or need to use multiple streams, you can pass a dictionary mapping each RNG stream name to its corresponding PRNGKey to init.
    """
    params = self.model.init(init_rng, imgs)['params']
    tx = optax.sgd(self.config.learning_rate, self.config.momentum)
    state = SDTrainState.create(apply_fn=self.model.apply, params=params, tx=tx)

    loss, _ = self.eval_step(state, batch)
    buffer_state = self.buffer.init(loss[0])
    buffer_state = self.buffer.add(buffer_state, loss[1:])
    state = state.replace(buffer_state=buffer_state)
    
    state = state.replace(xepoch=-1)
    state = state.replace(xstep=-1)
    
    self.state = state

  def create_fn(self):
  # Creates jit compatible train_step and eval_step functions

    def calc_batch_loss(params, batch):
      imgs, labels = batch
      logits = self.model.apply({'params': params}, imgs)
      one_hot = jax.nn.one_hot(labels, 10)
      batch_loss = optax.softmax_cross_entropy(logits=logits, labels=one_hot)
      acc = jnp.mean(jnp.argmax(logits, -1) == labels)
      metrics = {'ce_loss': jnp.mean(batch_loss), 'accuracy': acc, 'batch_loss': batch_loss}
      return batch_loss, metrics
    
    # Calculates the loss of the batch, including adjustments, if needed, for SD
    def calc_final_loss(batch_loss, batch_ref=None):
      ce_loss = jnp.mean(batch_loss)
      if self.config.loss == 'standard':
        loss = ce_loss
      elif self.config.loss == 'mean_risk':
        loss = mean_risk(batch_loss)
      elif self.config.loss == 'sd_2nd_cdf':
        loss = sd_2nd_cdf(-batch_loss, -batch_ref) # in sd_loss, get_utility=False
      return loss

    # Training function
    def train_step(state, batch, rng=None):

      def loss_fn(params):
        batch_loss, metrics = calc_batch_loss(params, batch) # cross-entropy over batch, standard classifier
        batch_ref = None
        if self.config.loss == 'sd_2nd_cdf':
          batch_ref = self.buffer.sample(state.buffer_state, rng)['experience'] # samples batch_size from replay buffer
        final_loss = calc_final_loss(batch_loss, batch_ref)
        return final_loss, metrics

      # loss_fn should return a scalar (which includes arrays with shape () but not arrays with shape (1,) etc.)
      (final_loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

      state = state.apply_gradients(grads=grads)
      new_buffer_state = self.buffer.add(state.buffer_state, metrics['batch_loss']) # batch_loss are the losses of each example in the batch
      state = state.replace(buffer_state=new_buffer_state) # update the buffer
      return state, metrics
    
    def eval_step(state, batch):
      batch_loss, metrics = calc_batch_loss(state.params, batch)
      return batch_loss, metrics
    
    self.train_step = jax.jit(train_step)
    self.eval_step = jax.jit(eval_step)

  @staticmethod
  def format_log(epoch, train_metrics, test_metrics=None):
    if test_metrics is not None:
      return 'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f, test_loss: %.4f, test_accuracy: %.2f' \
      % (epoch, train_metrics['ce_loss'], train_metrics['accuracy'] * 100, test_metrics['ce_loss'], test_metrics['accuracy'] * 100)
    else:
       return 'epoch:% 3d, train_loss: %.4f, train_accuracy: %.2f' \
      % (epoch, train_metrics['ce_loss'], train_metrics['accuracy'] * 100)
      
def main(argv):
  if len(argv) > 1: # argv[0] is script name
    raise app.UsageError('Too many command-line arguments.')
 
  logging.info('JAX process: %d / %d', jax.process_index(), jax.process_count())
  logging.info('JAX local devices: %r', jax.local_devices())

  # orbax fills the log with info logs which cannot be suppressed.
  # Suppress info logs and use warnings to output your info
  logging.set_verbosity(logging.WARNING)
 
  DATASET_PATH = FLAGS.workdir
  config = FLAGS.config

  seed = 0
  config.seed = seed
  torch.manual_seed(config.seed)
  checkpoint_dir = abspath("./checkpoints/mnist/seed_"+str(seed))  # now absolute
  """
  Trainer() initializes jax RNG.
  prepare() -> maybe_restore_checkpoint() 
    -> setup() to initialize everything to defaults
    -> manager.restore() to restore state (including RNG)
  """
  manager, trainer, config = prepare.prepare(config, Trainer(config), get_dataloader, checkpoint_dir)
  # test saving a checkpoint to skip waiting for epoch completion 
  if True:
      train_metrics = defaultdict(list)
      train_metrics['epoch'] = 0
      test_metrics = defaultdict(list)
      test_metrics['epoch'] = 0
      train.save_checkpoint(manager, trainer, config, 0, 0, train_metrics, test_metrics)
      sleep(3) # let checkpointer complete in background
      exit()
  
  train.train_and_evaluate(config, trainer, manager, get_dataloader, FLAGS.workdir)


  # for seed in range(10):
  #   config.seed = seed
  #   torch.manual_seed(config.seed)
  #   checkpoint_dir = abspath("./checkpoints/mnist/seed_"+str(seed))  # now absolute
  #   for loss in ['standard', 'sd_2nd_cdf']:
  #     config.loss = loss
  #     manager, trainer, config = prepare.prepare(config, Trainer(config), get_dataloader, checkpoint_dir)
  #     train.train_and_evaluate(config, trainer, manager, get_dataloader, FLAGS.workdir)

if __name__ == '__main__':
  flags.mark_flags_as_required(['config', 'workdir'])
  app.run(main)
