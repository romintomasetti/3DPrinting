import tensorflow
import numpy

def getClusterModelParams(model_state,activation):
    """
    Helps defining the Cluster Neural Netwok structure.
    Parameters
    ----------
    model_state : list of int
        model_state[0] : number of pairs "Function/Context"
        model_state[1] : number of layers in each Function/Context network
        model_state[2] : number of neurons in each layer
    activation : str
        Activation function
    """
    model_params = {}

    model_params["pairs"] = model_state[0]

    layers = [
        [
            model_state[1] for y in range(2)
        ] for x in range(model_params["pairs"])
    ]


    model_params["layers"] = layers

    neurons = [
        [
            [
                model_state[2]for z in range(model_state[1])
            ] for y in range(2)
        ] for x in range(model_params["pairs"])
    ]

    model_params["neurons"] = neurons

    activations = [
        [
            [
                activation for z in range(model_state[1])
            ] for y in range(2)
        ] for x in range(model_params["pairs"])
    ]

    model_params['activation'] = activations

    return model_params

class ClusterNetworkModel:

    def __init__(
        self,
        FC_num_layers = 0,
        FC_num_neurons = [],
        FC_activations = [],
        Cluster_input_size = -1,
        number_of_pairs_FuncContext = 3,
        number_of_layers            = [[2,2],[2,2],[2,2]],
        number_of_neurons           = [
            [[5,5],
            [5,5]],
            [[5,5],
            [5,5]],
            [[5,5],
            [5,5]]
        ],
        activation_function = [
            [["tanh","tanh"],
            ["tanh","tanh"]],
            [["tanh","tanh"],
            ["tanh","tanh"]],
            [["tanh","tanh"],
            ["tanh","tanh"]]
        ],
        verbose = True) -> None:
        """
        Initialize a Neural network model of the type "cluster".
        Parameters
        ----------
        FC_num_layers int
            Number of Fully-Connected layers.
        FC_num_neurons : list of int
            Number of neurons in each Fully-Connected Layer.
        Cluster_input_size : int
            Input size of the cluster network.
        number_of_pairs_FuncContext : int
            Number of Function network/Context network pairs.
        number_of_layers : list
            Number of layers in each Context/Function network.
        number_of_neurons : list
            Number of neurons in each layer.
        """

        assert isinstance(FC_num_layers,int)
        self.FC_num_layers = FC_num_layers

        assert len(FC_num_neurons) == FC_num_layers
        self.FC_num_neurons = FC_num_neurons

        assert len(FC_activations) == FC_num_layers
        self.FC_activations = FC_activations

        # Number of pairs "Function network / context network":
        assert isinstance(number_of_pairs_FuncContext,int)
        self.number_of_pairs_FuncContext = number_of_pairs_FuncContext

        # Number of layers in each 'sub-network':
        assert len(number_of_layers) == number_of_pairs_FuncContext
        assert all(len(x) == 2 for x in number_of_layers)
        self.number_of_layers = number_of_layers

        # Number of neurons in each layer:
        assert len(number_of_neurons) == number_of_pairs_FuncContext
        assert all(len(x) == 2 for x in number_of_neurons)
        for i in range(len(number_of_neurons)):
            for j in range(len(number_of_neurons[i])):
                assert len(number_of_neurons[i][j]) == number_of_layers[i][j]
        self.number_of_neurons = number_of_neurons

        # Activation function of each layer:
        assert len(activation_function) == number_of_pairs_FuncContext
        assert all(len(x) == 2 for x in activation_function)
        for i in range(len(activation_function)):
            for j in range(len(activation_function[i])):
                assert len(activation_function[i][j]) == number_of_layers[i][j]
        self.activation_function = activation_function

        # Verbosity
        self.verbose = verbose

        print(Cluster_input_size)
        assert Cluster_input_size >= 0
        self.Cluster_input_size = Cluster_input_size

        # Ensure matching of sizes
        if self.FC_num_layers > 0 and self.number_of_pairs_FuncContext > 0:
            assert self.FC_num_neurons[-1] == self.Cluster_input_size

    def getModel(self):
        return self.model

    def PrintIf(self,*args):
        if self.verbose:
            print(*args)

    def CreateFullyConnectedNetwork(
        self,
        type_net,
        input,
        layers,
        neurons,
        activation,
        pair
    ):

        for layer in range(layers-1):

            # Xavier Standard Deviation
            xavier_stddev = numpy.sqrt(2/(neurons[layer] + neurons[layer+1]))

            # Initialize weights
            weights = tensorflow.Variable(
                tensorflow.truncated_normal(
                        [neurons[layer], neurons[layer+1]],
                        stddev=xavier_stddev,
                        dtype = tensorflow.float32
                ),
                dtype=tensorflow.float32,
                name=type_net + str(pair) + "_weights_" + str(layer))

            biases = tensorflow.Variable(
                tensorflow.zeros(
                    [1,neurons[layer+1]],
                    dtype=tensorflow.float32
                ),
                dtype=tensorflow.float32,
                name=type_net + str(pair) + "bias_"+str(layer))

            if layer < layers - 2:

                if activation[layer] == "tanh":
                    tf_acti_func = tensorflow.tanh
                else:
                    raise Exception("Unknown activation function " + activation[layer])

                input = tf_acti_func(
                    tensorflow.add(
                        tensorflow.matmul(
                            input,weights
                        ),
                        biases
                    )
                )

            else:

                input = tensorflow.add(
                    tensorflow.matmul(
                        input,weights
                    ),
                    biases
                )

        return input  

    def CreateModel(self,input,input_size,output_size):

        self.PrintIf("> Creating model...")

        # Create the Fully-Connected network
        if self.FC_num_layers > 0:
            with tensorflow.name_scope("FullyConnected"):
                layers  = self.FC_num_layers + 2
                neurons = self.FC_num_neurons
                neurons.insert(0,input_size)
                if self.Cluster_input_size > 0:
                    neurons.append(self.Cluster_input_size)
                else:
                    neurons.append(output_size)
                output_FullyConnected = \
                    self.CreateFullyConnectedNetwork(
                        "FuncNet",
                        input,
                        layers,
                        neurons,
                        self.FC_activations,
                        "FC"
                    )
            input = output_FullyConnected

        output_subnets = []

        # Loop over the pairs of 'function/context' networks:
        for pair in range(self.number_of_pairs_FuncContext):

            self.PrintIf("\t> Creating pair 'Function/Context' ",pair)
                
            with tensorflow.name_scope("FuncContPair_" + str(pair)):
                # Create the function network
                layers = self.number_of_layers[pair][0]
                neurons = self.number_of_neurons[pair][0]
                activation = self.activation_function[pair][0]

                neurons.insert(0,self.Cluster_input_size)
                neurons.append(1)

                layers += 2

                self.PrintIf("\t\t> Function network layers : ",layers," with neurons ",neurons)

                output_function = 0

                with tensorflow.name_scope("FunctionNetwork_" + str(pair)):
                    output_function = \
                        self.CreateFullyConnectedNetwork(
                            "FuncNet",
                            input,
                            layers,
                            neurons,
                            activation,
                            pair
                        )

                # Create the context network
                layers = self.number_of_layers[pair][1]
                neurons = self.number_of_neurons[pair][1]
                activation = self.activation_function[pair][1]

                neurons.insert(0,self.Cluster_input_size)
                neurons.append(1)

                layers += 2

                self.PrintIf("\t\t> Context network layers : ",layers," with neurons ",neurons)

                output_context = 0

                with tensorflow.name_scope("ContextNetwork_" + str(pair)):
                    output_context = \
                        self.CreateFullyConnectedNetwork(
                            "ContNet",
                            input,
                            layers,
                            neurons,
                            activation,
                            pair
                        )

                output_subnets.append(
                    tensorflow.multiply(
                        output_context,
                        output_function,
                        name = "Subnet_" + str(pair)
                    )
                )

        # Convert output_subnets to meaningfull tensorflow output
        if self.number_of_pairs_FuncContext > 0:
            with tensorflow.name_scope("out"):
                self.model = tensorflow.transpose(tensorflow.squeeze(
                    tensorflow.convert_to_tensor(
                        output_subnets,
                        name = "output",
                        dtype=tensorflow.float32
                    ),
                    axis = 2
                ))
            print(input)
            print(output_subnets[0])
        else:
            self.model = output_FullyConnected
        print(self.model)

        return self.model
