import torch.nn as nn
import torch
import torch.nn.functional as F

from glf.common.containers import OrderedSet

class G(nn.Module):
    
    def __init__(self, cnn, games=None, key_size=128, g_size=20, one_g = False):
        super(G, self).__init__()

        if games==None:
            games = []

        if not one_g:
            self._gkey = OrderedSet(games)
            self._gmat = nn.ParameterList([nn.Parameter(torch.randn(key_size, g_size)) for gs in self._gkey])
        else:
            self._gkey = OrderedSet(['default'])
            self._gmat = nn.ParameterList([nn.Parameter(torch.randn(key_size, g_size)) for gs in self._gkey])

        self._batches = []

        self.key_size = key_size
        self.g_size = g_size

        self.one_g = one_g

        wn = nn.utils.weight_norm

        self.cnn = cnn
        self.output_size = cnn.output_size
        self.state_size = cnn.state_size

        #key generation neural network
        self.key_gen = nn.Sequential(
            wn(nn.Linear(cnn.output_size,cnn.output_size)),
            nn.ELU(),
            wn(nn.Linear(cnn.output_size,cnn.output_size)),
            nn.ELU(),
            wn(nn.Linear(cnn.output_size,key_size)))

        self.fuse = nn.Linear(cnn.output_size+key_size,cnn.output_size)

    @property
    def is_cuda(self):
        return any(p.is_cuda for p in self.parameters())
    
    @property
    def batches(self):
        return self._batches

    def set_batches(self, batches):

        self._batches = batches

        for game_state in OrderedSet(batches):
            self.add_game_state(game_state)

        # parameterlist does not register as cuda 
        # if initialized as empty
        if self.is_cuda:
            self._gmat.cuda()

    def add_game_state(self, game_state):
        if not self.one_g:
            if game_state not in self._gkey:
                self._gkey.add(game_state)
                self._gmat.append(nn.Parameter(torch.randn(self.key_size, self.g_size)))

    def gbatch(self):

        glist = []
        key_list = list(self._gkey)

        for game_state in self._batches:
            if game_state in self._gkey:
                ind = key_list.index(game_state)
                glist.append(self._gmat[ind])
            elif 'default' in self._gkey:
                ind = key_list.index('default')
                glist.append(self._gmat[ind])
            else:
                raise KeyError('{} not found'.format(game_state))

        return torch.stack(glist)
    
    def forward(self, inputs, states, masks):
        
        _, x, states = self.cnn(inputs, states, masks)

        n_batch = len(self.batches)

        if n_batch:

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
