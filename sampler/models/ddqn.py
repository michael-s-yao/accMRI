from models.buffer import ReplayBuffer
from models.policy import DDQNPolicy


class DDQNPolicyTrainer:
    def __init__(
        self,
        num_steps: int,
        obs_shape: Union[tuple, torch.Size],
        action_buffer_size: int,
        capacity: int = 1000,
        batch_size: int = 2,
        burn_in: int = 100,
        max_lines: int = 8,
        seed: Optional[int] = None,
        filepath: Optional[str] = None
    ):
        """
        Args:
            num_steps: number of sampling steps to simulate in a single
                training step.
            obs_shape: the shape of the tensors representing kspace (BCHW2).
            action_buffer_size: the number of actions that can be stored.
            capacity: how many transitions can be stored. After capcity is
                reached, early transitions are overwritten in FIFO fashion.
            batch_size: batch_size returned by the replay buffer.
            burn_in: threshold for replay buffer size below which the memory
                will return None. Used for burn-in period before training.
            max_lines: the maximum number of lines that can be acquired per
                acquisition step.
            seed: optional random seed.
            filepath: optional .pt file to read from to populate the replay
                buffer. If provided, the other inputs are ignored and
                parameters are automatically set from the .pt file.
        """
        self.num_steps = num_steps
        self.memory = ReplayBuffer(
            obs_shape,
            action_buffer_size,
            capacity,
            batch_size,
            burn_in,
            max_lines,
            seed,
            filepath
        )
        self.policy = DDQNPolicy(
            max_lines,
            in_chans,
            self.memory,
            chans=chans,
            num_layers=num_layers,
            seed=seed,
            reconstruction_size=reconstruction_size,
            eps_thresh=eps_thresh,
            q_thresh=q_thresh
        )
        self.target_nn = DDQNPolicy(
            max_lines,
            in_chans,
            None,
            chans=chans,
            num_layers=num_layers,
            seed=seed,
            reconstruction_size=reconstruction_size,
            eps_thresh=eps_thresh,
            q_thresh=q_thresh
        )

    def _train_policy(self):
        for _  in range(num_steps):

