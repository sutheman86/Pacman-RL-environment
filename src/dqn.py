import torch
import numpy as np
from torch._prims_common import check
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

class DuelingQNet(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DuelingQNet, self).__init__()
        self.layers = {
                "layer1": nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
                "layer2": nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                "layer3": nn.Conv2d(64, 64, kernel_size=3, stride=1),
        }
        print(self.layers["layer1"])
        self.conv = nn.Sequential(
            self.layers["layer1"],
            nn.ReLU(),
            self.layers["layer2"],
            nn.ReLU(),
            self.layers["layer3"],
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)  # Output a single value V(s)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)  # Output advantages A(s, a)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        value = self.value_stream(conv_out)
        advantage = self.advantage_stream(conv_out)

        # Combine value and advantage to get Q-values
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

class QNet(nn.Module):
    # TODO(Lab-4): Q-Network architecture.
    def __init__(self, input_shape, n_actions):
        super(QNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions),
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        out = self.fc(conv_out)
        return out

class DeepQNetwork():
    def __init__(
        self,
        n_actions,
        input_shape,
        qnet,
        device,
        learning_rate=2e-4,
        reward_decay=0.99,
        replace_target_iter=1000,
        memory_size=10000,
        batch_size=32,
    ):
        # initialize parameters
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.lr = learning_rate
        self.gamma = reward_decay
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.device = device
        self.learn_step_counter = 0
        self.init_memory()

        # Network
        self.qnet_eval = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target = qnet(self.input_shape, self.n_actions).to(self.device)
        self.qnet_target.eval()
        self.optimizer = optim.RMSprop(self.qnet_eval.parameters(), lr=self.lr)
        self.episode = 0
        self.qval = 0

    def choose_action(self, state, epsilon=0):
        # 將狀態轉換為 FloatTensor 並增加 batch 維度
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        actions_value = self.qnet_eval.forward(state)
        self.qval = actions_value
        if np.random.uniform() > epsilon:  # greedy
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()[0]
        else:  # random
            action = np.random.randint(0, self.n_actions)
        return action, actions_value

    def learn(self):
        # 替换目标网络参数
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.qnet_target.load_state_dict(self.qnet_eval.state_dict())

        # 随机采样经验池中的一个批次
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        b_s = torch.FloatTensor(self.memory["s"][sample_index]).to(self.device)
        b_a = torch.LongTensor(self.memory["a"][sample_index]).to(self.device)
        b_r = torch.FloatTensor(self.memory["r"][sample_index]).to(self.device)
        b_s_ = torch.FloatTensor(self.memory["s_"][sample_index]).to(self.device)
        b_d = torch.FloatTensor(self.memory["done"][sample_index]).to(self.device)

        # DQN 和 DDQN 两种方式
        q_curr_eval = self.qnet_eval(b_s).gather(1, b_a)
        q_next_target = self.qnet_target(b_s_).detach()
        q_next_eval = self.qnet_eval(b_s_).detach()

        next_state_values = q_next_target.gather(1, q_next_eval.max(1)[1].unsqueeze(1))  # DDQN
        q_curr_recur = b_r + (1 - b_d) * self.gamma * next_state_values

        # 损失计算
        self.loss = F.smooth_l1_loss(q_curr_eval, q_curr_recur)

        # 反向传播和优化
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()
        self.learn_step_counter += 1
        return self.loss.detach().cpu().numpy()



    def init_memory(self):
        # 初始化经验池
        self.memory = {
            "s": np.zeros((self.memory_size, *self.input_shape)),
            "a": np.zeros((self.memory_size, 1)),
            "r": np.zeros((self.memory_size, 1)),
            "s_": np.zeros((self.memory_size, *self.input_shape)),
            "done": np.zeros((self.memory_size, 1)),
        }

    def store_transition(self, s, a, r, s_, d):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        index = self.memory_counter % self.memory_size
        self.memory["s"][index] = s
        self.memory["a"][index] = np.array(a).reshape(-1, 1)
        self.memory["r"][index] = np.array(r).reshape(-1, 1)
        self.memory["s_"][index] = s_
        self.memory["done"][index] = np.array(d).reshape(-1, 1)
        self.memory_counter += 1

    def save_load_model(self, op, path="save", fname="qnet.pt"):
        if not os.path.exists(path):
            os.makedirs(path)
        file_path = os.path.join(path, fname)

        if op == "save":
            # 保存模型狀態、優化器狀態、學習步驟和經驗池計數
            checkpoint = {
                'qnet_eval_state_dict': self.qnet_eval.state_dict(),
                'qnet_target_state_dict': self.qnet_target.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'learn_step_counter': self.learn_step_counter,
                'memory_counter': self.memory_counter,
                'episode': self.episode,
            }
            torch.save(checkpoint, file_path)
            print(f"Model saved successfully at {file_path}")

        elif op == "load":
            try:
                # 加載模型狀態、優化器狀態、學習步驟和經驗池計數
                checkpoint = torch.load(file_path, map_location=self.device)

                # 檢查是否包含所有必需的鍵
                required_keys = ['qnet_eval_state_dict', 'qnet_target_state_dict', 'optimizer_state_dict']
                missing_keys = [key for key in required_keys if key not in checkpoint]

                if missing_keys:
                    raise KeyError(f"Missing keys in checkpoint: {missing_keys}")

                # 加載各部分的狀態
                self.qnet_eval.load_state_dict(checkpoint['qnet_eval_state_dict'])
                self.qnet_target.load_state_dict(checkpoint['qnet_target_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # 選擇性地加載學習計數
                self.learn_step_counter = checkpoint.get('learn_step_counter', 0)
                self.memory_counter = checkpoint.get('memory_counter', 0)
                self.episode = checkpoint.get('episode', 0)

                print("Model loaded successfully from", file_path)
                return {'learn_step_counter': self.learn_step_counter, 'episode': self.episode}

            except FileNotFoundError:
                print(f"No saved model found at {file_path}, starting fresh.")
            except KeyError as e:
                print(f"Error loading model: {e}")

            # 如果未成功加載模型或發生錯誤，返回初始狀態
        return {'learn_step_counter': 0, 'episode': 0} 
