import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        input_shape,
        num_classes,
        hidden_sizes=[15, 15, 15],
        activation="relu",
        bias=False,
        skip_connections=None
    ):

        super(MLP, self).__init__()
        self.num_classes = num_classes
        self.layers = []
        self.activation = activation
        self.input_shape = input_shape

        #TODO: allow multi-channel data
        #ch, w, h = self.input_shape
        #self.input_size = ch * w * h

        self.layers_size = hidden_sizes
        self.layers_size.insert(0, self.input_shape)
        self.layers_size.append(num_classes)

        self.num_layers = len(self.layers_size)
        self.modulispace_dimension = None
        self.skip_connections = skip_connections

        for idx in range(len(self.layers_size) - 1):
            self.layers.append(
                nn.Linear(self.layers_size[idx], self.layers_size[idx + 1], bias=bias)
            )

            if idx == len(self.layers_size) - 2:
                break

            if self.skip_connections is None:
                self.layers.append(self.get_activation_fn()())

        if self.skip_connections is not None:
            self.activation_fn = self.get_activation_fn()()

        self.net = nn.Sequential(*self.layers)
        self._init()
        self._get_moduli_space_dimension()

    def forward(self, x):
        if self.skip_connections is None:
            x = self.net(x)
            return x

        if not isinstance(self.skip_connections, list):
            TypeError('skip_connections should be a list of tuples, it is not a list')

        if not isinstance(self.skip_connections[0], tuple):
            TypeError('skip_connections should be a list of tuples, the elements are not tuples')

        for begin, end in self.skip_connections:
            if not begin < end:
                TypeError('The tuples in skip_connections should be ordered in increasing order')

        previous = 0

        for i in range(self.num_layers):
            x = self.activation_fn(self.layers[i](x))

            for begin, end in self.skip_connections:
                if i == begin:
                    previous = x
                elif i == end:
                    x += previous

    def get_weights(self):
        w = []
        for m in self.modules():
            if isinstance(m, nn.Linear):
                w.append(m.weight.data)
        return w

    def get_activation_fn(self):
        act_name = self.activation.lower()
        activation_fn_map = {
            "relu": nn.ReLU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "leakyrelu": nn.LeakyReLU,
            "prelu": nn.PReLU,
            "sigmoid": nn.Sigmoid,
        }
        if act_name not in activation_fn_map.keys():
            raise ValueError("Unknown activation function name : ")
        return activation_fn_map[act_name]

    def _init(self):
        def init_func(m):
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight.data)

        self.apply(init_func)

    def _get_moduli_space_dimension(self):
        #TODO: maybe check this
        dimension = self.layers_size[0]*self.layers_size[1]
        for i in range(1, len(self.layers_size)-1):
            dimension += self.layers_size[i]*self.layers_size[i+1]
            dimension -= self.layers_size[i]

        self.modulispace_dimension = dimension
