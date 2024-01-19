#!/usr/bin/env python

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper
import cv2
import psutil
import torch

import torch.nn as nn
import torch.optim as optim
import random
import collections
import math
import os

class DQNNet(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(DQNNet, self).__init__()
        # Definan aqui la arquitectura de la CNN
        self.input_shape = input_shape
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        self.feature_size = self.feature_size()
        self.fc_layers = nn.Sequential(
            nn.Linear(self.feature_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)  # flatten
        return self.fc_layers(x)

    def feature_size(self):
        return self.conv_layers(torch.zeros(1, *self.input_shape)).view(1, -1).size(1)
    
    def load_weights(self, weights_path):
        self.load_state_dict(torch.load(weights_path))

class DQN:

    def guardar(self):
        torch.save(self.net.state_dict(), 'duckies_M_pes_v1.pth')

    def __init__(self, input_shape, num_actions, learning_rate=1e-4, device=torch.device("cpu")):
        self.net = DQNNet(input_shape, num_actions)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learning_rate)
        self.memory = collections.deque(maxlen=100000)
        self.gamma = 0.8  # Factor de descuento
        self.epsilon = 1.0  # Para la exploracion -> Greedy
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.995
        self.batch_size = 32  # Ajustado a 128
        self.device = device
        self.num_actions = num_actions


    def select_action(self, state):
        print("self.epsilon",self.epsilon)
        #if random.random() < self.epsilon:
        if 2 < self.epsilon:
            random_accion=random.randrange(self.num_actions)
            print("Paso Ramdom", random_accion)
            return random_accion
        else:
            state = torch.tensor(np.expand_dims(state, axis=0), dtype=torch.float32)
            q_values = self.net(state)
            print("Paso RED",torch.argmax(q_values).item())
            return torch.argmax(q_values).item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        if (len(self.memory) % 1000)==0:
            print("Guardado")
            dqn.guardar()
    
    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_np      = np.array(states, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        actions_np     = np.array(actions)

        rewards_np     = np.array(rewards)
        dones_np       = np.array(dones)
        states      = torch.tensor(states_np)
        next_states = torch.tensor(next_states_np).permute(0, 3, 1, 2)
        actions     = torch.tensor(actions_np, dtype=torch.int64)   
        rewards     = torch.tensor(rewards_np, dtype=torch.float32)
        dones       = torch.tensor(dones_np, dtype=torch.uint8)

        q_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        next_q_values = self.net(next_states).max(1)[0]
        expected_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay    

class Navegacion:
    # Parametros para el detector de lineas blancas
    white_filter_1 = np.array([0, 0, 0])
    white_filter_2 = np.array([180, 48, 255])
    numero=0
    

    # Filtros para el detector de lineas amarillas
    yellow_filter_1 = np.array([26, 62, 156])
    yellow_filter_2 = np.array([31, 255, 243])


    window_filter_name = "filtro"

    def __init__(self, map_name):
        self.numero=0
        self.env = DuckietownEnv(
            seed=1,
            map_name=map_name,
            draw_curve=False,
            draw_bbox=False,
            domain_rand=False,
            frame_skip=1,
            distortion=False,
        )

        self.env.reset()
        self.env.render()
        self.key_handler = key.KeyStateHandler()
        self.env.unwrapped.window.push_handlers(self.key_handler)

        self.next_action = np.array([0.0, 0.0])

        self.env.unwrapped.cur_pos = np.array([0.78, 0, 0.92], dtype=np.float64)
        self.env.unwrapped.cur_angle = np.array([-np.pi / 2], dtype=np.float64)

    def calculate_duckie_center(self, white_pixels, yellow_pixels):
        if white_pixels is not None:
            duckie_center = np.mean(white_pixels[:, 0, :], axis=0)
        elif yellow_pixels is not None:
            duckie_center = np.mean(yellow_pixels[:, 0, :], axis=0)
        else:
            duckie_center = None

        return duckie_center

    def update(self, dt=None):
        action = self.next_action

        if self.key_handler[key.UP]:
            action[0] += 0.44
        if self.key_handler[key.DOWN]:
            action[0] -= 0.44
        if self.key_handler[key.LEFT]:
            action[1] += 1
        if self.key_handler[key.RIGHT]:
            action[1] -= 1
        if self.key_handler[key.SPACE]:
            action = np.array([0, 0])

        if self.key_handler[key.LSHIFT]:
            action *= 1.5

        print("Accion Movimiento", action)

        obs, reward, done, info = self.env.step(action)
        
        action = dqn.select_action(obs.transpose(2,0,1))
        action_list = [0, 0]
        if action == 0:
            action_list =np.array([0.2, 0.0])  # Avanzar
        elif action == 1:
            action_list= np.array([0.1, +0.02])  # Girar a la derecha
        elif action == 2:
            action_list= np.array([0.1, -0.02]) # Girar a la izquierda
        elif action == 3:
            action_list= np.array([0.3, +0.8])  # Diagonal derecha (avanzar y girar ligeramente a la derecha)
        elif action == 4:
            action_list= np.array([0.3, -0.8]) # Diagonal izquierda (avanzar y girar ligeramente a la izquierda)
        elif action == 5:
            action_list= np.array([0.4, 0.0]) # Rapido adelante


        next_observation, reward, done, info = self.env.step(tuple(action_list))
        
        if self.numero % 32 == 0:
            #dqn.learn()# comenta esta linea para no aprender y activa est linea 75 #if 2 < self.epsilon: y comenta la anterior dejar de aprender y dejar andar solo
            print("Sin aprender")

        self.numero=self.numero+1

        self.next_action = tuple(action_list)

        self.env.render()
        posicion_reward=self.line_follower(obs)


        if action == 0:
            if posicion_reward == 0:
                reward=10
            elif posicion_reward == 1:
                reward=10
            elif posicion_reward == 2:
                reward=-2
            elif posicion_reward == 3:
                reward=-2
            elif posicion_reward == 4:
                reward=-8
            elif posicion_reward == 5:
                reward=-8
            elif posicion_reward == 8:
                reward=-8
        elif action == 1:
            if posicion_reward == 0:
                reward=-2
            elif posicion_reward == 1:
                reward=10
            elif posicion_reward == 2:
                reward=-2
            elif posicion_reward == 3:
                reward=10
            elif posicion_reward == 4:
                reward=-2
            elif posicion_reward == 5:
                reward=-10
            elif posicion_reward == 8:
                reward=-8
        elif action == 2:
            if posicion_reward == 0:
                reward=-2
            elif posicion_reward == 1:
                reward=-2
            elif posicion_reward == 2:
                reward=10
            elif posicion_reward == 3:
                reward=-2
            elif posicion_reward == 4:
                reward=10
            elif posicion_reward == 5:
                reward=-8
            elif posicion_reward == 8:
                reward=-10        
        elif action == 3:
            if posicion_reward == 0:
                reward=-2
            elif posicion_reward == 1:
                reward=10
            elif posicion_reward == 2:
                reward=-2
            elif posicion_reward == 3:
                reward=10
            elif posicion_reward == 4:
                reward=-2
            elif posicion_reward == 5:
                reward=10
            elif posicion_reward == 8:
                reward=8
        elif action == 4:
            if posicion_reward == 0:
                reward=-2
            elif posicion_reward == 1:
                reward=-2
            elif posicion_reward == 2:
                reward=10
            elif posicion_reward == 3:
                reward=-2
            elif posicion_reward == 4:
                reward=10
            elif posicion_reward == 5:
                reward=-8
            elif posicion_reward == 8:
                reward=-10  
        elif action == 5:
            if posicion_reward == 0:
                reward=10
            elif posicion_reward == 1:
                reward=10
            elif posicion_reward == 2:
                reward=-2
            elif posicion_reward == 3:
                reward=-2
            elif posicion_reward == 4:
                reward=-8
            elif posicion_reward == 5:
                reward=-8
            elif posicion_reward == 8:
                reward=-8

        

        if done:
            reward=-100
            print("RESET")
            self.env.reset()

        dqn.store_transition(obs.transpose(2,0,1), action, reward, next_observation, done)
    
    def line_follower(self, observation):
 
        kernel = np.ones((4,4),np.uint8)
        
        puntosYline=self.get_line(observation,self.yellow_filter_1,self.yellow_filter_2,line_color='yellow')
 
        l1x1=puntosYline[0]
        l1y1=puntosYline[1]
        l1x2=puntosYline[2]
        l1y2=puntosYline[3]
        par1=(l1x1,l1y1)
        par2=(l1x2,l1y2)

        puntosWline=self.get_line(observation,self.white_filter_1,self.white_filter_2,'white')

        l2x1=puntosWline[0]
        l2y1=puntosWline[1]
        l2x2=puntosWline[2]
        l2y2=puntosWline[3]
        par3=(l2x1,l2y1)
        par4=(l2x2,l2y2)
               
        x_intersection, y_intersection=find_intersection(par1, par2, par3, par4)

        reward=1.0
        slope=0.0
        if x_intersection is not None and   y_intersection is not None: 

            point1=x_intersection, y_intersection
            point2=self.env.unwrapped.cur_pos[0],self.env.unwrapped.cur_pos[2]
            slope, intercept=line_equation(point1, point2)
            
            if ( not math.isnan(slope)):  #esto pasa cuando el amarillo es un punto, por lo tanto me alejo del amarillo
                if(slope<0.3 and slope>-0.3):
                    
                    if(slope>0):
                        #print("cero, pero mayor que cero")
                        #retonar=np.array([0.2, -0.001])
                        reward = 0 
                        
                    elif (slope <0):
                        #print("cero, pero menor que cero")
                        #retonar=np.array([0.2, +0.001])
                        reward = 1
                else:
                    #retonar=np.array([0.0, 0.0])
                    if(slope>0):
                        #print("mas que cero, pero mayor que cero")
                        #retonar=np.array([0.1, -0.01])
                        reward = 2
                    elif (slope <0):
                        #print("cero, pero menor que cero")
                        #retonar=np.array([0.1, +0.01])
                        reward = 3
            else:
                #retonar=np.array([0.1, -0.1])
                reward = 4

        else:
            #print("x o y es none")
            if(( (par1==par2==(0,0) and (par3!=(0,0) and par4!=(0,0)))) ): #amarillo, linea izquieda es cero
                #print("no veo amarillo")
                #retonar=np.array([0.01, 0.1])
                reward = 5

            if(par3==par4==(0,0)and (par1!=(0,0) and par2!=(0,0))): #blanco, linea derecha es cero
                #print("no veo blanco")
                #retonar=np.array([0.01, -0.1])
                reward = 6

        return reward
 
    @staticmethod
    def get_line(
        observation,
        filter_1,
        filter_2,
        line_color,
        erode_kernel=None,
        dilate_kernel=None,
        erosion_iterations=None,
        dilate_iterations=None,
    ):

        converted = cv2.cvtColor(observation, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(converted, filter_1, filter_2)
        segment_image = cv2.bitwise_and(converted, converted, mask=mask)
        filtered_image = cv2.cvtColor(segment_image, cv2.COLOR_HSV2BGR)

        # opening
        if erode_kernel is not None:
            filtered_image = cv2.erode(
                filtered_image, erode_kernel, iterations=erosion_iterations
            )

        if dilate_kernel is not None:
            filtered_image = cv2.dilate(
                filtered_image, dilate_kernel, iterations=dilate_iterations
            )

        gray_lines = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2GRAY)

        cdstP = np.copy(gray_lines)

        edges = cv2.Canny(gray_lines, 50, 200, None, 3)
        lines = cv2.HoughLinesP(gray_lines, 1, np.pi / 180, 150, None, 0, 0)

        xP1=0
        yP1=0
        xP2=0
        yP2=0
        if lines is not None:
            acum=0
            for i in range(0, len(lines)):
                l = lines[i][0]
                _xP1=l[0]
                _yP1=l[1]
                _xP2=l[2]
                _yP2=l[3]
                d=np.sqrt((_xP1-_xP2)**2 + (_yP1-_yP2)**2  )
                acum+=d
                xP1+=l[0]*d
                yP1+=l[1]*d
                xP2+=l[2]*d
                yP2+=l[3]*d
            if acum > 0:
                xP1=int(xP1/acum)             
                yP1=int(yP1/acum)
                xP2=int(xP2/acum)
                yP2=int(yP2/acum)

        cv2.imshow(line_color, filtered_image)

        return (xP1, yP1, xP2, yP2)

def line_equation(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    if x2 - x1 == 0:
        slope = np.inf
        intercept = x1
    else:
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1

    return slope, intercept

def find_intersection(point1_line1, point2_line1, point1_line2, point2_line2):

    if(point1_line1==point2_line1 or point1_line2==point2_line2):
        return None, None
    slope1, intercept1 = line_equation(point1_line1, point2_line1)
    slope2, intercept2 = line_equation(point1_line2, point2_line2)

    if slope1 == slope2:
        print("Las líneas son paralelas y no se intersectan.")
        return None, None
    else:
        x_intersection = (intercept2 - intercept1) / (slope1 - slope2)
        y_intersection = slope1 * x_intersection + intercept1
        return x_intersection, y_intersection

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--map-name", default="bigger_loop")
    args = parser.parse_args()
    # Crear una instancia de DQN y establecer el dispositivo
    app = Navegacion(args.map_name)
    dqn = DQN(input_shape=(3, 640, 480), num_actions=6, device=torch.device("cpu"))  # Resolución reducida

    weights_path = 'duckies_M_pes_v1.pth'
    existe = os.path.isfile(weights_path)

    if existe:
        print("El archivo existe.")
        dqn.net.load_weights(weights_path)
    else:
        print("El archivo no existe.")

    pyglet.clock.schedule_interval(app.update, 0.5 / app.env.unwrapped.frame_rate)
    pyglet.app.run()
    app.env.close()
