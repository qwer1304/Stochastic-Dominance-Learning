from absl import logging
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collections
from collections import defaultdict
import numpy as np
import optax
from utils import TrainableModel, to_plain_dict, flatten_and_convert_metrics
import pickle
## Progress bar
from tqdm.auto import tqdm
from pathlib import Path
import orbax.checkpoint as ocp

def update_metrics_batch(epoch_metrics, batch_metrics):
  for metric in batch_metrics:
    if metric == 'batch_loss':
      epoch_metrics['sample_loss'] += list(batch_metrics[metric])
    else:
      epoch_metrics[metric].append(batch_metrics[metric])
  return

def update_metrics_epoch(epoch_metrics):
  for metric in epoch_metrics:
    if metric == 'sample_loss':
      continue
      # sample_loss = epoch_metrics[metric]
      # mean_sl = np.mean(sample_loss)
      # mean_abs_dev = np.mean(np.abs(sample_loss - mean_sl))
      # mean_abs_dev_med = 0.5*np.mean(np.abs(sample_loss - np.median(sample_loss)))
      # epoch_metrics['mean_abs_dev'] = mean_abs_dev
      # epoch_metrics['mean_abs_dev_med'] = mean_abs_dev_med
    else:
      epoch_metrics[metric] = np.mean(epoch_metrics[metric])
  return

# ================================
# SAVE CHECKPOINT
# ================================
def save_checkpoint(manager, trainer, config, epoch, step, train_metrics, test_metrics):
          
      metrics = flatten_and_convert_metrics({'train': train_metrics, 'test': test_metrics})
      
      config_dict = to_plain_dict(config)
      
      manager.save(
            step,
            args=ocp.args.Composite(
              state=ocp.args.StandardSave(trainer.state),
              rng=ocp.args.JaxRandomKeySave(trainer.rng),
              config=ocp.args.JsonSave(config_dict),
            ),
            metrics=metrics
      )
      #manager.save(step, args=ocp.args.StandardSave(trainer.state), metrics=metrics)

def train_and_evaluate(
    config: ml_collections.ConfigDict, trainer: TrainableModel, manager, get_dataloader, workdir: str
) -> train_state.TrainState:
  """Execute model training and evaluation loop.

  Args:
    config: Hyperparameter configuration for training and evaluation.

  Returns:
    The train state (which includes the `.params`).
  """
  train_loader, test_loader = get_dataloader(config)

  #trainer.create_train_state(next(iter(train_loader)))

  # Over epochs:
  start_epoch = trainer.state.xepoch + 1 # xepoch was the last epoch completed in checkpoint, so start from next
  step = trainer.state.xstep + 1 # ditto for step
  for epoch in range(start_epoch, config.num_epochs):

    # Perform training
    train_metrics = defaultdict(list)
    train_metrics['epoch'] = epoch
    # Over batches in each epoch:
    # This corresponds to Algorithm 1, with the computation of u^* on Line 6 done with Algorithm 2 
    # and sampling of X_{theta_t}^N with a replay buffer 
    for batch in tqdm(train_loader, desc=f"Epoch {epoch}", leave=False):
      # Generate 2 independent pseudorandom values given a single key.
      trainer.rng, rng = jax.random.split(trainer.rng)
      # Do one train step
      trainer.state, metrics = trainer.train_step(trainer.state, batch, rng) # in run_mnist
      update_metrics_batch(train_metrics, metrics)
      step += 1
    update_metrics_epoch(train_metrics)      

    # Perform evaluation
    test_metrics = None
    eval_flag = (epoch % 1 == 0) or (epoch == config.num_epochs)
    if eval_flag:
      test_metrics = defaultdict(list)
      test_metrics['epoch'] = epoch
      for batch in tqdm(test_loader, desc=f"Epoch {epoch}", leave=False):
        _, metrics = trainer.eval_step(trainer.state, batch) # in run_mnist
        update_metrics_batch(test_metrics, metrics)
      update_metrics_epoch(test_metrics)      
        
    logging.warning(trainer.format_log(epoch, train_metrics, test_metrics))

    if config.save_log and eval_flag:
      train_metrics['method'] = config.loss
      test_metrics['method'] = config.loss
      filename = config.result_path+'%s_log.pkl'%(config.task)
      all_train_metrics = defaultdict(list)
      all_test_metrics = defaultdict(list)
      if Path(filename).is_file():
        with open(filename, 'rb') as f:
          try:
            all_train_metrics = pickle.load(f)
            all_test_metrics = pickle.load(f)
          except EOFError:
            break
      for metric in train_metrics:
        if metric == 'sample_loss': continue
        all_train_metrics[metric].append(train_metrics[metric])
      for metric in test_metrics:
        if metric == 'sample_loss': continue
        all_test_metrics[metric].append(test_metrics[metric])
      with open(filename, 'wb') as f:
        pickle.dump(all_train_metrics, f)
        pickle.dump(all_test_metrics, f)
        
    trainer.state = trainer.state.replace(xepoch=epoch, xstep=step)
    if epoch % config.checkpoint_every_epochs == 0:
      # Save
      save_checkpoint(manager, trainer, config, epoch, step, train_metrics, test_metrics)

