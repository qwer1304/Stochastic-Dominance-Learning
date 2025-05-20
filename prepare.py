import ml_collections
from utils import TrainableModel, to_config_dict_recursive
from flax.training import orbax_utils
import orbax.checkpoint as ocp
import logging
from orbax.checkpoint.logging import AbstractLogger  # base class


# ================================
# CHECKPOINT MANAGER
# ================================
def create_ckpt_manager(ckpt_dir):

    def setup_logger(level=logging.INFO):
        class QuietLogger(AbstractLogger):
            def log_entry(self, msg, *args, **kwargs):
                # log_level: 'INFO', 'WARNING', etc.
                #if log_level in ['INFO', 'WARNING', 'ERROR']:  # only show WARNING and above
                #logging.info(msg, *args, **kwargs)
                return
       
        orbax_logger = QuietLogger()
        return orbax_logger

    options = ocp.CheckpointManagerOptions(max_to_keep=3, prevent_write_metrics=False, 
        best_fn=lambda metrics: 0)  
    # Create a CheckpointManager with the correct directory and target structure
    logger = setup_logger()
    manager = ocp.CheckpointManager(
        ckpt_dir,
        logger=logger,
        options=options,
        metadata={'version': 1.1, 'lang': 'en'}
    )
    return manager

# ================================
# SETUP
# ================================
def setup(config: ml_collections.ConfigDict, trainer: TrainableModel, get_dataloader):

    train_loader, test_loader = get_dataloader(config)
    trainer.create_train_state(next(iter(train_loader)))
    return trainer

# ================================
# MAYBE RESTORE FROM CHECKPOINT
# ================================
def maybe_restore_checkpoint(manager, config, trainer, get_dataloader):
    trainer =  setup(config, trainer, get_dataloader)
    if manager.latest_step() is not None:
        # Restore
        #args = ocp.args.StandardRestore(item=trainer.state)
        args=ocp.args.Composite(
            state=ocp.args.StandardRestore(item=trainer.state),
            rng=ocp.args.JaxRandomKeyRestore(),
            config=ocp.args.JsonRestore(item=config),
        )
        restored_data = manager.restore(manager.latest_step(),args=args)
        
        trainer.state = restored_data['state']
        trainer.rng = restored_data['rng']
        config = to_config_dict_recursive(restored_data['config'])       
        #state =  manager.restore(manager.latest_step(),args=args)
        #train.state = state
    return trainer, config

def prepare(config, trainer, get_dataloader, checkpoint_dir):

    manager = create_ckpt_manager(checkpoint_dir)
       
    trainer, config = maybe_restore_checkpoint(manager, config, trainer, get_dataloader)
    return manager, trainer, config
