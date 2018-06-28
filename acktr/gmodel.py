import torch.nn as nn
import torch
import torch.nn.functional as F

from glf.common.containers import OrderedSet

class G(nn.Module):
    
    def __init__(self, cnn, batches=None, key_size=128, g_size=20):
        super(G, self).__init__()

        if batches==None:
            batches = []

        self.g_key = OrderedSet(batches)

        self.g = nn.ParameterList([nn.Parameter(torch.randn(key_size, g_size)) for gs in self.g_key])

        self._batches = batches

        self.key_size = key_size
        self.g_size = g_size

        wn = nn.utils.weight_norm

        self.cnn = cnn
        self.output_size = cnn.output_size
        self.state_size = cnn.state_size
        latent_size = cnn.output_size

        #key generation neural network
        self.key_gen = nn.Sequential(
            wn(nn.Linear(latent_size,latent_size)),
            nn.ELU(),
            wn(nn.Linear(latent_size,latent_size)),
            nn.ELU(),
            wn(nn.Linear(latent_size,key_size)))

        self.fuse = nn.Linear(latent_size+key_size,latent_size)

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda
    
    @property
    def batches(self):
        return self._batches

    def set_batches(self, batches):
        g_set = len(self.g) > 0

        self._batches = batches

        for game_state in OrderedSet(batches):
            self.add_game_state(game_state)

        if not g_set:
            # parameterlist does not register as cuda 
            # if initialized as empty
            if self.is_cuda:
                self.g.cuda()

    def add_game_state(self, game_state):
        if game_state not in self.g_key:
            self.g_key.add(game_state)
            self.g.append(nn.Parameter(torch.randn(self.key_size, self.g_size)))

    def gbatch(self):

        glist = []
        key_list = list(self.g_key)

        for game_state in self._batches:
            if game_state in self.g_key:
                ind = key_list.index(game_state)
                glist.append(self.g[ind])
            else:
                raise KeyError('{} not found'.format(game_state))

        return torch.stack(glist)
    
    def forward(self, inputs, states, masks):
        
        _, x, states = self.cnn(inputs, states, masks)

        n_batch = len(self.batches)

        if x.size(0) % n_batch != 0:
            raise ValueError('expected {} to be a multiple of {}'.format(x.size(0),n_batch))

        n_step = x.size(0) // n_batch

        n_batch = n_step * n_batch
        #generate a look-up key for the G memory. add an extra book-keeping dimension
        x = x.view(n_batch,-1)
        key = self.key_gen(x).unsqueeze(1).view(n_batch,1,self.key_size)

        #get stacked g's
        gstack = self.gbatch().repeat(n_step,1,1)

        #dot the key into memory
        weights = torch.bmm(key,gstack).view(n_batch,self.g_size) 
        #take softmax to get weights for each entry in the memory (the lookup)
        weights = F.softmax(weights,dim=1).view(n_batch,1,self.g_size) 

        #do a weighted sum of all of the columns
        value = self.g_size*F.adaptive_avg_pool1d(gstack*weights.expand(n_batch,self.key_size,self.g_size),1)
        value = value.view(n_batch,self.key_size)

        #combine the result of the lookup with the latent state vector
        combination = torch.cat([x,value],dim=1)
        x = self.fuse(combination)

        #return policy decision
        return self.cnn.critic_linear(x), x, states
