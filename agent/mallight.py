from . import RLAgent
from common.registry import Registry
import numpy as np
import os
import random
from collections import OrderedDict, deque
import gym
import scipy.sparse as sp

from generator.lane_vehicle import LaneVehicleGenerator
from generator.intersection_phase import IntersectionPhaseGenerator
import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data, Batch
from torch_geometric.utils import add_self_loops


@Registry.register_model('mallight')
class MalLightAgent(RLAgent):
    #  TODO: test multiprocessing effect on agents or need deep copy here
    def __init__(self, world, rank):
        super().__init__(world, world.intersection_ids[rank])
        """
        multi-agents in one model-> modify self.action_space, self.reward_generator, self.ob_generator here
        """
        #  general setting of world and model structure
        # TODO: different phases matching

        #QCY: test running DConv
        # testFunc(Registry.mapping['world_mapping']['graph_setting'].graph)

        self.buffer_size = Registry.mapping['trainer_mapping']['setting'].param['buffer_size']
        self.replay_buffer = deque(maxlen=self.buffer_size)

        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph
        self.world = world
        self.sub_agents = len(self.world.intersections)
        # TODO: support dynamic graph later
        self.edge_idx = torch.tensor(self.graph['sparse_adj'].T, dtype=torch.long)  # source -> target

        #  model parameters
        self.phase = Registry.mapping['model_mapping']['setting'].param['phase']
        self.one_hot = Registry.mapping['model_mapping']['setting'].param['one_hot']
        self.model_dict = Registry.mapping['model_mapping']['setting'].param

        #  get generator for MalLightAgent
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for MalLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(world, inter, ["pressure"], average="all", negative=True)   #QCY: i change here
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  get queue generator for MalLightAgent
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        #  get delay generator for MalLightAgent
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"],
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # TODO: add irregular control of signals in the future
        self.action_space = gym.spaces.Discrete(len(self.world.intersections[0].phases))

        if self.phase:
            # TODO: irregular ob and phase in the future
            if self.one_hot:
                self.ob_length = self.ob_generator[0][1].ob_length + len(self.world.intersections[0].phases)
            else:
                self.ob_length = self.ob_generator[0][1].ob_length + 1
        else:
            self.ob_length = self.ob_generator[0][1].ob_length

        self.get_attention = Registry.mapping['logger_mapping']['setting'].param['attention']
        # train parameters
        self.rank = rank
        self.gamma = Registry.mapping['model_mapping']['setting'].param['gamma']
        self.grad_clip = Registry.mapping['model_mapping']['setting'].param['grad_clip']
        self.epsilon = Registry.mapping['model_mapping']['setting'].param['epsilon']
        self.epsilon_decay = Registry.mapping['model_mapping']['setting'].param['epsilon_decay']
        self.epsilon_min = Registry.mapping['model_mapping']['setting'].param['epsilon_min']
        self.learning_rate = Registry.mapping['model_mapping']['setting'].param['learning_rate']
        self.vehicle_max = Registry.mapping['model_mapping']['setting'].param['vehicle_max']
        self.batch_size = Registry.mapping['model_mapping']['setting'].param['batch_size']

        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_network()
        self.criterion = nn.MSELoss(reduction='mean')
        self.optimizer = optim.RMSprop(self.model.parameters(),
                                       lr=self.learning_rate,
                                       alpha=0.9, centered=False, eps=1e-7)

    def reset(self):
        observation_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ['lane_count'], in_only=True, average=None)
            observation_generators.append((node_idx, tmp_generator))
        sorted(observation_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.ob_generator = observation_generators

        #  get reward generator for MalLightAgent
        rewarding_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, average='all', negative=True)
            rewarding_generators.append((node_idx, tmp_generator))
        sorted(rewarding_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.reward_generator = rewarding_generators

        #  phase generator
        phasing_generators = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = IntersectionPhaseGenerator(self.world, inter, ['phase'],
                                                       targets=['cur_phase'], negative=False)
            phasing_generators.append((node_idx, tmp_generator))
        sorted(phasing_generators, key=lambda x: x[0])  # now generator's order is according to its index in graph
        self.phase_generator = phasing_generators

        # queue metric
        queues = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_waiting_count"],
                                                 in_only=True, negative=False)
            queues.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(queues, key=lambda x: x[0])
        self.queue = queues

        # delay metric
        delays = []
        for inter in self.world.intersections:
            node_id = inter.id if 'GS_' not in inter.id else inter.id[3:]
            node_idx = self.graph['node_id2idx'][node_id]
            tmp_generator = LaneVehicleGenerator(self.world, inter, ["lane_delay"],
                                                 in_only=True, average="all", negative=False)
            delays.append((node_idx, tmp_generator))
        # now generator's order is according to its index in graph
        sorted(delays, key=lambda x: x[0])
        self.delay = delays

    def get_ob(self):
        x_obs = []  # sub_agents * lane_nums,
        for i in range(len(self.ob_generator)):
            x_obs.append((self.ob_generator[i][1].generate()) / self.vehicle_max)
        # construct edge information.
        length = set([len(i) for i in x_obs])
        if len(length) == 1: # each intersections may has  different lane nums
            x_obs = np.array(x_obs, dtype=np.float32)
        else:
            x_obs = [np.expand_dims(x,axis=0) for x in x_obs]
        return x_obs

    def get_reward(self):
        # TODO: test output
        rewards = []  # sub_agents
        for i in range(len(self.reward_generator)):
            rewards.append(self.reward_generator[i][1].generate())
        rewards = np.squeeze(np.array(rewards, dtype=np.float32)) * 12
        return rewards

    def get_phase(self):
        # TODO: test phase output onehot/int
        phase = []  # sub_agents
        for i in range(len(self.phase_generator)):
            phase.append((self.phase_generator[i][1].generate()))
        phase = (np.concatenate(phase)).astype(np.int8)
        # phase = np.concatenate(phase, dtype=np.int8)
        return phase

    def get_queue(self):
        """
        get delay of intersection
        return: value(one intersection) or [intersections,](multiple intersections)
        """
        queue = []
        for i in range(len(self.queue)):
            queue.append((self.queue[i][1].generate()))
        tmp_queue = np.squeeze(np.array(queue, dtype=np.float32))
        queue = np.sum(tmp_queue, axis=1 if len(tmp_queue.shape)==2 else 0)
        return queue

    def get_delay(self):
        delay = []
        for i in range(len(self.delay)):
            delay.append((self.delay[i][1].generate()))
        delay = np.squeeze(np.array(delay, dtype=np.float32))
        return delay # [intersections,]

    def get_action(self, ob, phase, test=False):
        """
        input are np.array here
        # TODO: support irregular input in the future
        :param ob: [agents, ob_length] -> [batch, agents, ob_length]
        :param phase: [agents] -> [batch, agents]
        :param test: boolean, exploit while training and determined while testing
        :return: [batch, agents] -> action taken by environment
        """
        if not test:
            if np.random.rand() <= self.epsilon:
                return self.sample()
        observation = torch.tensor(ob, dtype=torch.float32)

        edge = self.edge_idx
        dp = Data(x=observation, edge_index=edge)
        # TODO: not phase not used
        if self.get_attention:
            # TODO: collect attention matrix later
            actions = self.model(x=dp.x.cuda(), edge_index=dp.edge_index.cuda(), train=False)   #QCY: add cuda()
            att = None
            # QCY: add cpu() to move actions to host memory
            actions = actions.clone().detach().cpu().numpy()
            if len(actions.shape)==3:
                actions = np.squeeze(actions,axis=0)
            return np.argmax(actions, axis=1), att  # [batch, agents], [batch, agents, nv, neighbor]
        else:
            actions = self.model(x=dp.x.cuda(), edge_index=dp.edge_index.cuda(), train=False)    #QCY: add cuda()
            # QCY: add cpu() to move actions to host memory
            actions = actions.clone().detach().cpu().numpy()
            if len(actions.shape) == 3:
                actions = np.squeeze(actions,axis=0)
            return np.argmax(actions, axis=1)  # [batch, agents] TODO: check here

    def sample(self):
        return np.random.randint(0, self.action_space.n, self.sub_agents)

    def _build_model(self):
        model = MallightNet(self.ob_length, self.action_space.n, **self.model_dict)

        # QCY: load the model to cuda
        # model = model.to(self.device)
        model = model.cuda()

        return model

    def remember(self, last_obs, last_phase, actions, actions_prob, rewards, obs, cur_phase, done, key):
        self.replay_buffer.append((key, (last_obs, last_phase, actions, rewards, obs, cur_phase)))

    def _batchwise(self, samples):
        # load onto tensor

        batch_list = []
        batch_list_p = []
        actions = []
        rewards = []
        for item in samples:
            dp = item[1]
            state = torch.tensor(dp[0], dtype=torch.float32)
            batch_list.append(Data(x=state, edge_index=self.edge_idx))

            state_p = torch.tensor(dp[4], dtype=torch.float32)
            batch_list_p.append(Data(x=state_p, edge_index=self.edge_idx))
            rewards.append(dp[3])
            actions.append(dp[2])
        batch_t = Batch.from_data_list(batch_list)
        batch_tp = Batch.from_data_list(batch_list_p)
        # TODO reshape slow warning
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.long)
        if self.sub_agents > 1:
            rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
            actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here
        # rewards = rewards.view(rewards.shape[0] * rewards.shape[1])
        # actions = torch.tensor(np.array(actions), dtype=torch.long)
        # actions = actions.view(actions.shape[0] * actions.shape[1])  # TODO: check all dimensions here

        return batch_t, batch_tp, rewards, actions

    def train(self):
        samples = random.sample(self.replay_buffer, self.batch_size)
        b_t, b_tp, rewards, actions = self._batchwise(samples)

        # QCY: move data to GPU
        # b_tp = b_tp.to(self.device)
        # b_t = b_t.to(self.device)
        b_tp = b_tp.cuda()
        b_t = b_t.cuda()
        # print("b_t.x.shape",b_t.x.shape)
        rewards = rewards.cuda()
        # print("rewards:",rewards.shape)
        actions = actions.cuda()
        # print("actions:",actions.shape)

        out = self.target_model(x=b_tp.x, edge_index=b_tp.edge_index, train=False)
        out = torch.reshape(out,[self.batch_size*self.sub_agents,8])
        # print("out.shape:",out.shape)
        target = rewards + self.gamma * torch.max(out, dim=1)[0]
        target_f = self.model(x=b_t.x, edge_index=b_t.edge_index, train=False)
        target_f = torch.reshape(target_f,[self.batch_size*self.sub_agents,8])
        # print("target_f.shape:",target_f.shape)

        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        loss = self.criterion(self.model(x=b_t.x, edge_index=b_t.edge_index, train=True), target_f)

        # QCY: move loss to GPU
        # loss = loss.to(self.device)
        loss = loss.cuda()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.clone().detach().cpu().numpy()

    def update_target_network(self):
        weights = self.model.state_dict()
        self.target_model.load_state_dict(weights)

    def load_model(self, e):
        model_name = os.path.join(Registry.mapping['logger_mapping']['path'].path,
                                  'model', f'{e}_{self.rank}.pt')
        self.model = self._build_model()
        self.model.load_state_dict(torch.load(model_name))
        self.target_model = self._build_model()
        self.target_model.load_state_dict(torch.load(model_name))

        # QCY: put model to GPU
        self.model = self.model.cuda()
        self.target_model = self.target_model.cuda()

    def save_model(self, e):
        path = os.path.join(Registry.mapping['logger_mapping']['path'].path, 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        model_name = os.path.join(path, f'{e}_{self.rank}.pt')
        torch.save(self.target_model.state_dict(), model_name)


class MallightNet(nn.Module):
    def __init__(self, input_dim, output_dim, **kwargs):
        super(MallightNet, self).__init__()
        self.model_dict = kwargs
        self.action_space = gym.spaces.Discrete(output_dim)
        self.features = input_dim
        self.module_list = nn.ModuleList()
        self.embedding_MLP = Embedding_MLP(self.features, layers=self.model_dict.get('NODE_EMB_DIM')).cuda()
        #QCY: put embedding MLP to GPU
        self.graph = Registry.mapping['world_mapping']['graph_setting'].graph
        self.support = buildSupport(self.graph).cuda()
        self.length = int((max(self.graph["node_id2idx"].values())+1)**0.5)
        self.diffusionConvNet = DiffusionGraphConv(self.support, 128, self.length**2, 100, 128)

        for i in range(self.model_dict.get('N_LAYERS')):
            block = MultiHeadAttModel(d=self.model_dict.get('INPUT_DIM')[i],
                                      dv=self.model_dict.get('NODE_LAYER_DIMS_EACH_HEAD')[i],
                                      d_out=self.model_dict.get('OUTPUT_DIM')[i],
                                      nv=self.model_dict.get('NUM_HEADS')[i],
                                      suffix=i).cuda()
            self.module_list.append(block)

        # output_dict = OrderedDict()
        output_dict = nn.Sequential()   #QCY: I change here

        if len(self.model_dict['OUTPUT_LAYERS']) != 0:
            # TODO: dubug this branch
            for l_idx, l_size in enumerate(self.model_dict['OUTPUT_LAYERS']):
                # name = f'output_{l_idx}'
                if l_idx == 0:
                    h = nn.Linear(block.d_out, l_size)
                else:
                    h = nn.Linear(self.model_dict.get('OUTPUT_LAYERS')[l_idx - 1], l_size)
                # output_dict.update({name: h})
                output_dict.append(h)
                # name = f'relu_{l_idx}'
                # output_dict.update({name: nn.ReLU})
                output_dict.append(nn.ReLU())
            out = nn.Linear(self.model_dict['OUTPUT_LAYERS'][-1], self.action_space.n)
        else:
            out = nn.Linear(block.d_out, self.action_space.n)
        # name = f'output'
        # output_dict.update({name: out})
        output_dict.append(out)
        self.output_layer = output_dict

    def forward(self, x, edge_index, train=True):
        h = self.embedding_MLP.forward(x, train)
        # TODO: implement att

        if train:
            for mdl in self.module_list:
                h = mdl.forward(h, edge_index, train)
            h = self.diffusionConvNet(h)
            h = self.output_layer(h)
        else:
            with torch.no_grad():
                for mdl in self.module_list:
                    h = mdl.forward(h, edge_index, train)
                h = self.diffusionConvNet(h)
                h = self.output_layer(h)
        return h


class Embedding_MLP(nn.Module):
    def __init__(self, in_size, layers):
        super(Embedding_MLP, self).__init__()
        # constructor_dict = OrderedDict()
        constructor_dict = nn.Sequential()   #QCY: I change here

        for l_idx, l_size in enumerate(layers):
            # name = f"node_embedding_{l_idx}"
            if l_idx == 0:
                h = nn.Linear(in_size, l_size)
                # constructor_dict.update({name: h})
                constructor_dict.append(h)
            else:
                h = nn.Linear(layers[l_idx - 1], l_size)
                # constructor_dict.update({name: h})
                constructor_dict.append(h)
            # name = f"n_relu_{l_idx}"
            # constructor_dict.update({name: nn.ReLU()})
            constructor_dict.append(nn.ReLU())
        # self.embedding_node = nn.Sequential(constructor_dict)
        self.embedding_node = constructor_dict

    def _forward(self, x):
        x = self.embedding_node(x)
        return x

    def forward(self, x, train=True):
        if train:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


class MultiHeadAttModel(MessagePassing):
    """
    inputs:
        In_agent [bacth,agents,128]
        In_neighbor [agents, neighbor_num]
        l: number of neighborhoods (in my code, l=num_neighbor+1,because l include itself)
        d: dimension of agents's embedding
        dv: dimension of each head
        dout: dimension of output
        nv: number of head (multi-head attention)
    output:
        -hidden state: [batch,agents,32]
        -attention: [batch,agents,neighbor]
    """
    def __init__(self, d, dv, d_out, nv, suffix):
        super(MultiHeadAttModel, self).__init__(aggr='add')
        self.d = d
        self.dv = dv
        self.d_out = d_out
        self.nv = nv
        self.suffix = suffix
        # target is center
        self.W_target = nn.Linear(d, dv * nv)
        self.W_source = nn.Linear(d, dv * nv)
        self.hidden_embedding = nn.Linear(d, dv * nv)
        self.out = nn.Linear(dv, d_out)
        self.att_list = []
        self.att = None

    def _forward(self, x, edge_index):
        # TODO: test batch is shared or not

        # x has shape [N, d], edge_index has shape [E, 2]
        edge_index, _ = add_self_loops(edge_index=edge_index)
        aggregated = self.propagate(x=x, edge_index=edge_index)  # [16, 16]
        out = self.out(aggregated)
        out = F.relu(out)  # [ 16, 128]
        #self.att = torch.tensor(self.att_list)
        return out

    def forward(self, x, edge_index, train=True):
        if train:
            return self._forward(x, edge_index)
        else:
            with torch.no_grad():
                return self._forward(x, edge_index)

    def message(self, x_i, x_j, edge_index):
        h_target = F.relu(self.W_target(x_i))
        h_target = h_target.view(h_target.shape[:-1][0], self.nv, self.dv)
        agent_repr = h_target.permute(1, 0, 2)

        h_source = F.relu(self.W_source(x_j))
        h_source = h_source.view(h_source.shape[:-1][0], self.nv, self.dv)

        neighbor_repr = h_source.permute(1, 0, 2)  # [nv, E, dv]
        index = edge_index[1]  # which is target

        e_i = torch.mul(agent_repr, neighbor_repr).sum(-1)  # [5, 64]
        max_node = torch_scatter.scatter_max(e_i, index=index)[0]  # [5, 16]
        max_i = max_node.index_select(1, index=index)  # [5, 64]
        ec_i = torch.add(e_i, -max_i)
        ecexp_i = torch.exp(ec_i)
        norm_node = torch_scatter.scatter_sum(ecexp_i, index=index)  # [5, 16]
        normst_node = torch.add(norm_node, 1e-12)  # [5, 16]
        normst_i = normst_node.index_select(1, index)  # [5, 64]

        alpha_i = ecexp_i / normst_i  # [5, 64]
        alpha_i_expand = alpha_i.repeat(self.dv, 1, 1)
        alpha_i_expand = alpha_i_expand.permute((1, 2, 0))  # [5, 64, 16]

        hidden_neighbor = F.relu(self.hidden_embedding(x_j))
        hidden_neighbor = hidden_neighbor.view(hidden_neighbor.shape[:-1][0], self.nv, self.dv)
        hidden_neighbor_repr = hidden_neighbor.permute(1, 0, 2)  # [5, 64, 16]
        out = torch.mul(hidden_neighbor_repr, alpha_i_expand).mean(0)

        # TODO: attention ouput in the future
        # self.att_list.append(alpha_i)  # [64, 16]
        return out

    def get_att(self):
        if self.att is None:
            print('invalid att')
        return self.att

class DiffusionGraphConv(nn.Module):
    def __init__(self, support, input_dim, num_nodes, max_diffusion_step, output_dim, bias_start=1.414):
        super(DiffusionGraphConv, self).__init__()
        self.num_matrices = 1 * max_diffusion_step + 1  # Don't forget to add for x itself.
        input_size = input_dim
        self._num_nodes = num_nodes
        self._max_diffusion_step = max_diffusion_step
        self.output_size = output_dim
        self.support = support
        self.weight = nn.Parameter(torch.FloatTensor(size=(input_size*self.num_matrices, output_dim)))
        self.biases = nn.Parameter(torch.FloatTensor(size=(output_dim,)))
        nn.init.xavier_normal_(self.weight.data, gain=1.414)   #QCY: change from 1.414 to 100, then back
        nn.init.constant_(self.biases.data, val=bias_start)

    @staticmethod
    def _concat(x, x_):
        x_ = torch.unsqueeze(x_, 0)
        return torch.cat([x, x_], dim=0)

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward(self, inputs, bias_start=1.414):
        """
        Diffusion Graph convolution with graph matrix
        :param inputs:
        :param state:
        :param output_size:
        :param bias_start:
        :return:
        """
        # print(inputs.shape)
        if len(inputs.shape) == 2:
            if inputs.shape[0] % 256 == 0:
                batch_size = 256
            else:
                batch_size = 1
        else:
            # Reshape input and state to (batch_size, num_nodes, input_dim/state_dim)
            batch_size = inputs.shape[0]
        inputs = torch.reshape(inputs, (batch_size, self._num_nodes, -1))
        input_size = inputs.shape[2]
        # dtype = inputs.dtype

        x = inputs
        x0 = torch.transpose(x, dim0=0, dim1=1)
        x0 = torch.transpose(x0, dim0=1, dim1=2)  # (num_nodes, total_arg_size, batch_size)
        x0 = torch.reshape(x0, shape=[self._num_nodes, input_size * batch_size])
        x = torch.unsqueeze(x0, dim=0)

        if self._max_diffusion_step == 0:
            pass
        else:
            x1 = torch.sparse.mm(self.support, x0)
            x = self._concat(x, x1)
            for k in range(2, self._max_diffusion_step + 1):
                x2 = 2 * torch.sparse.mm(self.support, x1) - x0
                x = self._concat(x, x2)
                x1, x0 = x2, x1

        x = torch.reshape(x, shape=[self.num_matrices, self._num_nodes, input_size, batch_size])
        x = torch.transpose(x, dim0=0, dim1=3)  # (batch_size, num_nodes, input_size, order)
        x = torch.reshape(x, shape=[batch_size * self._num_nodes, input_size * self.num_matrices])

        x = torch.matmul(x, self.weight)  # (batch_size * self._num_nodes, output_size)
        x = torch.add(x, self.biases)
        # Reshape res back to 2D: (batch_size, num_node, state_dim) -> (batch_size, num_node * state_dim)
        return torch.reshape(x, [batch_size*self._num_nodes, self.output_size])

def build_sparse_matrix(L):
    """
    build pytorch sparse tensor from scipy sparse matrix
    reference: https://stackoverflow.com/questions/50665141
    :return:
    """
    shape = L.shape
    i = torch.LongTensor(np.vstack((L.row, L.col)).astype(int))
    v = torch.FloatTensor(L.data)
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))

def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx

def buildSupport(graph):
    adj_m = graph["sparse_adj"]
    adj_mat = np.zeros((16, 16)).astype(np.float32)
    for i, j in adj_m:
        adj_mat[i, j] = 300
        adj_mat[j, i] = 300
    return build_sparse_matrix(calculate_random_walk_matrix(adj_mat).T)

def testFunc(graph):
    adj_m = graph["sparse_adj"]
    length = int((max(graph["node_id2idx"].values())+1)**0.5)

    #创建adj_matrix与supports
    adj_mat = np.zeros((16, 16)).astype(np.float32)
    for i,j in adj_m:
        adj_mat[i,j] = 300
        adj_mat[j,i] = 300
    print("adj_mat.shape:",adj_mat.shape)
    print("adj_mat:",adj_mat)
    support = build_sparse_matrix(calculate_random_walk_matrix(adj_mat).T)   #QCY: should add .cuda() later
    # print("supports.shape:",support.shape)

    #supports, input_dim, num_nodes, max_diffusion_step, output_dim, bias_start=0.0
    net = DiffusionGraphConv(support, 128, length**2, 100, 128)

    #inputs
    inputs = np.random.randint(1,100,[256,16,128])
    inputs = torch.from_numpy(inputs.astype("float32"))
    print("inputs size:", inputs.shape)


    device = torch.device("cpu")
    model = net.to(device)

    output = model(inputs)

    #output = torch.reshape(output, [1, 49, 30])
    print("output size:", output.shape)

    torch.set_printoptions(profile="full")
    output = torch.reshape(output,[256,length**2,128])
    print("reshaped output size:",output.shape)
    # print(output[0,1,:])
    exit(1)
