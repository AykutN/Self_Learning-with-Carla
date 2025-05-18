import torch
import numpy as np
import torchvision.transforms as T


class ReplayBuffer:
    def __init__(self, buffer_size: int, batch_size: int, device, state_shape) -> None:
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.max_size = buffer_size
        self.device = device

        self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size,) + state_shape, dtype=np.float32)
        self.action = np.zeros((self.max_size, 1), dtype=np.int64)
        self.next_state = np.zeros((self.max_size,) + state_shape, dtype=np.float32)
        self.reward = np.zeros((self.max_size, 1), dtype=np.float32)
        self.done = np.zeros((self.max_size, 1), dtype=np.float32)

    def add(self, state, action, reward, next_state, done) -> None:
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.reward[self.ptr] = reward
        self.next_state[self.ptr] = next_state
        self.done[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.max_size
        self.crt_size = min(self.crt_size + 1, self.max_size)

    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.done[ind]).to(self.device)
        )

    def __len__(self):
        return self.crt_size