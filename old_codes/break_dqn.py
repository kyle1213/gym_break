import gymnasium as gym
import numpy as np
import random
import copy
import datetime
import torch
import torch.nn.functional as F
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from collections import deque
import torchvision.transforms as transforms


# DQN을 위한 파라미터 값 세팅
state_size = [84, 84, 1]
action_size = 4

load_model = False
train_mode = True

batch_size = 128
mem_maxlen = 100000
discount_factor = 0.99
learning_rate = 0.0001

run_step = 10000000 if train_mode else 0
test_step = 0 if train_mode else 10000
train_start_step = 50000
target_update_step = 5000

print_interval = 10
save_interval = 100

epsilon_eval = 0
epsilon_init = 0.1 if train_mode else epsilon_eval
epsilon_min = 0.
explore_step = run_step * 0.1
epsilon_delta = 0
#epsilon_delta = (epsilon_init - epsilon_min) / (1 * explore_step) if train_mode else 0.


# 모델 저장 및 불러오기 경로
date_time = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
save_path = f"./saved_models/DQN/{date_time}"
load_path = f"./saved_models/DQN/20231126203925"

# 연산 장치
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Transforms:
    def to_gray(frame1, frame2=None):
        gray_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.CenterCrop((175,150)),
            transforms.Resize((84, 84)),
            transforms.ToTensor()
        ])
        # Subtract one frame from the other to get sense of ball and paddle direction
        if frame2 is not None:
            new_frame = gray_transform(frame1) - 0.5*frame2
        else:
            new_frame = gray_transform(frame1)
        return new_frame


def init_gym_env(env_path):
    env = gym.make(env_path, render_mode='human')
    env.metadata['render_fps'] = 60
    state = env.reset()[0]
    processed_state = Transforms.to_gray(state)

    return env, processed_state


class DQN(torch.nn.Module):
    def __init__(self, **kwargs):
        super(DQN, self).__init__(**kwargs)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32,
                                     kernel_size=(8, 8), stride=(4, 4))
        dim1 = ((state_size[0] - 8) // 4 + 1, (state_size[1] - 8) // 4 + 1)
        self.conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64,
                                     kernel_size=(4, 4), stride=(2, 2))
        dim2 = ((dim1[0] - 4) // 2 + 1, (dim1[1] - 4) // 2 + 1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=64,
                                     kernel_size=(3, 3), stride=(1, 1))
        dim3 = ((dim2[0] - 3) // 1 + 1, (dim2[1] - 3) // 1 + 1)

        self.flat = torch.nn.Flatten()
        self.fc1 = torch.nn.Linear(64 * dim3[0] * dim3[1], 512)
        self.q = torch.nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.shape[0], -1)
        x = F.relu(self.fc1(x))

        return self.q(x)


# DQNAgent 클래스 -> DQN 알고리즘을 위한 다양한 함수 정의
class DQNAgent:
    def __init__(self):
        self.network = DQN().to(device)
        self.target_network = copy.deepcopy(self.network)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=mem_maxlen)
        self.epsilon = epsilon_init
        self.writer = SummaryWriter(save_path)

        if load_model == True:
            print(f"... Load Model from {load_path}/ckpt ...")
            checkpoint = torch.load(load_path + '/ckpt', map_location=device)
            self.network.load_state_dict(checkpoint["network"])
            self.target_network.load_state_dict(checkpoint["network"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

    # Epsilon greedy 기법에 따라 행동 결정
    def get_action(self, state, training=True):
        #  네트워크 모드 설정
        self.network.train(training)
        epsilon = self.epsilon if training else epsilon_eval
        # 랜덤하게 행동 결정
        if epsilon > random.random():
            qs = torch.Tensor([0, 0, 0, 0])
            qs[np.random.randint(0, action_size, size=1)[0]] = 1
        # 네트워크 연산에 따라 행동 결정
        else:
            qs = self.network(torch.FloatTensor(state).unsqueeze(dim=0).to(device))
        return qs

    # 리플레이 메모리에 데이터 추가 (상태, 행동, 보상, 다음 상태, 게임 종료 여부)
    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # 학습 수행
    def train_model(self):
        batch = random.sample(self.memory, batch_size)
        state = np.stack([b[0] for b in batch], axis=0)
        action = np.stack([b[1] for b in batch], axis=0)
        reward = np.stack([b[2] for b in batch], axis=0)
        next_state = np.stack([b[3] for b in batch], axis=0)
        done = np.stack([b[4] for b in batch], axis=0)

        state, action, reward, next_state, done = map(lambda x: torch.FloatTensor(x).to(device),
                                                      [state, action, reward, next_state, done])

        q = self.network(state).max(1, keepdims=True).values
        action = torch.argmax(action.unsqueeze(1), dim=1, keepdim=True)
        q = q.gather(1, action)

        with torch.no_grad():
            next_q = self.target_network(next_state)
            target_q = reward.unsqueeze(1) + next_q.max(1, keepdims=True).values * ((1 - done) * discount_factor)
        loss = F.huber_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # 엡실론 감소
        self.epsilon = max(epsilon_min, self.epsilon - epsilon_delta)

        return loss.item()

    # 타겟 네트워크 업데이트
    def update_target(self):
        self.target_network.load_state_dict(self.network.state_dict())

    # 네트워크 모델 저장
    def save_model(self):
        print(f"... Save Model to {save_path}/ckpt ...")
        torch.save({
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, save_path + '/ckpt')

    # 학습 기록
    def write_summary(self, score, loss, epsilon, step):
        self.writer.add_scalar("run/score", score, step)
        self.writer.add_scalar("model/loss", loss, step)
        self.writer.add_scalar("model/epsilon", epsilon, step)


os.environ['KMP_DUPLICATE_LIB_OK']='True'
if __name__ == '__main__':
    # env init
    env, state = init_gym_env('ALE/Breakout-v5')

    # agent init
    agent = DQNAgent()

    # values init
    scores, episode, losses = [], 0, []
    score = 0
    lives = 5

    for step in range(run_step + test_step):
        if step == run_step:
            if train_mode:
                agent.save_model()
            print("TEST START")
            train_mode = False
        action = torch.argmax(agent.get_action(state, train_mode)).cpu().numpy()

        old_state = state
        state, reward, done, _, info = env.step(action)
        state = Transforms.to_gray(state, old_state)
        env.render()

        score += reward

        if train_mode:
            agent.append_sample(old_state, action, reward, state, [done])

        if train_mode and step > max(batch_size, train_start_step):
            # 학습 수행
            loss = agent.train_model()
            losses.append(loss)

            # 타겟 네트워크 업데이트
            if step % target_update_step == 0:
                agent.update_target()

        if done and train_mode:
            episode += 1
            state = env.reset()[0]
            state = Transforms.to_gray(state)
            lives = 5
            scores.append(score)
            score = 0

            # 게임 진행 상황 출력 및 텐서 보드에 보상과 손실함수 값 기록
            if episode % print_interval == 0:
                mean_score = np.mean(scores)
                mean_loss = np.mean(losses)
                agent.write_summary(mean_score, mean_loss, agent.epsilon, step)
                actor_losses, critic_losses, scores, losses = [], [], [], []

                print(f"{episode} Episode / Step: {step} / Score: {mean_score:.2f} / " + \
                      f"Loss: {mean_loss:.4f} / Epsilon: {agent.epsilon:.4f}")

            # 네트워크 모델 저장
            if train_mode and episode % save_interval == 0:
                agent.save_model()
        elif done and not train_mode:
            print('scores: ', scores)



    env.close()