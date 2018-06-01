class PolicyPlus(nn.Module):
    """
    The model is a convolutional neural network. The input is the state/observation and
    the output is the quality of each action in the given state.
    """
    
    def __init__(self,g_size=10,latent_size=64,key_size=64,num_input_frames=1):
        super(PolicyPlus, self).__init__()

        #this is the original code
        self.conv1 = nn.Conv2d(num_input_frames*3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)
        self.state = nn.Linear(29600, latent_size)

        self.key_size = key_size


        #key generation neural network
        self.key_gen = nn.Sequential(nn.Linear(latent_size,latent_size),self.BatchNorm1d(latent_size),nn.Relu(),nn.Linear(latent_size,latent_size),
            self.BatchNorm1d(latent_size),nn.Relu(),nn.Linear(latent_size,key_size))

        #policy neural network
        self.policy_gen = nn.Sequential(nn.Linear(latent_size+key_size,latent_size),self.BatchNorm1d(latent_size),nn.Relu(),nn.Linear(latent_size,latent_size),
            self.BatchNorm1d(latent_size),nn.Relu(),nn.Linear(latent_size,10))
    
    def image_features(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.state(x.view(x.size(0), -1))

    def forward(self,x,G=None):
        latent_vector = self.image_features(x)

        combination = torch.cat([latent_vector,torch.ones(x.size(0),self.key_size).to(x.device)],dim=1)

        if G is not None:

            #generate a look-up key for the G memory. add an extra book-keeping dimension
            key = self.key_gen(latent_vector).unsqueeze_(1)

            #add a batch dimension -- just a little bit of dimensional resizing
            G_expand = G.expand((key.size(0),key.size(1),G.size(1)))

            #dot the key into memory
            weights = torch.bmm(key,G_memory).view(-1,G.size(1)) 
            #take softmax to get weights for each entry in the memory (the lookup)
            weights = F.softmax(weights,dim=1).view(-1,1,G.size(1)) 

            #do a weighted sum of all of the columns
            value = torch.sum(G_expand*weights,dim=2).view(-1,G.size(1))

            #combine the result of the lookup with the latent state vector
            combination = torch.cat([latent_vector,value],dim=1)


        #return policy decision
        return self.policy_gen(combination)


