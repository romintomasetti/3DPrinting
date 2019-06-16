"""
Train a Neural Network on a dataset
"""
from ClusterNetwork import *
from NeuralNetworkTrainer import *
import json,pprint,numpy,sys,os

def read_dataset(
    filename,
    PartionningTrainingValidation,
    params_inputOrdering,
    params_outputOrdering,
    params_input_normalization
):

    input  = None
    output = None
    ordering    = None

    data = []

    lines = open(filename,"r").readlines()

    for line in lines:
        if "input" in line:
            input  = int(line.split(",")[1].split(";")[0])
        elif "output" in line:
            output = int(line.split(",")[1].split(";")[0])
        elif "ordering" in line:
            line = line.split("ordering:")[1].replace("\n","")
            ordering = line.split(",")
        elif input is not None and output is not None and ordering is not None:
            line = line.replace(";\n","")
            data.append(
                [float(val) for val in line.split(",")]
            )
        else:
            raise Exception("Something went wrong for line " + line)

    # Prepare normalization:
    tmp_values = numpy.zeros((len(data),len(params_inputOrdering)))
    for counter_pair,pair in enumerate(data):
        for counter_par,(par_in,par_norm) in enumerate(zip(params_inputOrdering,params_input_normalization)):
            tmp_values[counter_pair,counter_par] = \
                pair[ordering.index(par_norm)]
            #print(par_in,pair[ordering.index(par_in)],par_norm,pair[ordering.index(par_norm)])
    mins_norm = numpy.amin(tmp_values,axis=0)
    maxs_norm = numpy.amax(tmp_values,axis=0)
    indices_tmp = numpy.where(mins_norm == maxs_norm)
    mins_norm[indices_tmp] = 0.

    input_names = ordering[0:input]
    output_names = ordering[input:input+output]

    num_pairs = len(data)

    num_pairs_training = int(PartionningTrainingValidation * num_pairs)

    num_pairs_validation = num_pairs - num_pairs_training

    print("> Dataset contains    ",num_pairs," data pairs.")
    print("> In the training   : ",num_pairs_training)
    print("> In the validation : ",num_pairs_validation)

    indices = numpy.arange(0,num_pairs,dtype=numpy.int64)

    numpy.random.shuffle(indices)

    print("\n> Ordering inside the dataset:")
    print("> Input  : ",input_names)
    print("> Output : ",output_names)

    print("\n> Ordering inside the parameters input file:")
    print("> Input  : ",params_inputOrdering)
    print("> Output : ",params_outputOrdering)

    input_ordering  = []
    output_ordering = []

    for _,params_input in enumerate(params_inputOrdering):
        FOUND = False
        for counter_D,input_name in enumerate(input_names):
            if params_input == input_name:
                input_ordering.append(counter_D)
                FOUND = True
                break
        if not FOUND:
            raise Exception(params_input + " not found in " + " , ".join(input_names))

    for _,params_output in enumerate(params_outputOrdering):
        FOUND = False
        for counter_D,output_name in enumerate(output_names):
            if params_output == output_name:
                output_ordering.append(counter_D)
                FOUND = True
                break
        if not FOUND:
            raise Exception(params_output + " not found in " + " , ".join(output_names))

    data_input  = []
    data_output = []

    for pair in data:
        # Take the inputs, and order them as is stated in the input file
        data_input.append(
            numpy.array(pair[0:input])
        )
        data_input[-1] = data_input[-1][input_ordering]
        # Take the outputs, and order them as is stated in the input file
        data_output.append(
            numpy.array(pair[input:input+output])
        )
        data_output[-1] = data_output[-1][output_ordering]

    data_input = numpy.array(data_input)

    data_output = numpy.array(data_output)

    data_input_training   = data_input[indices[0:num_pairs_training]]

    data_input_validation = data_input[indices[num_pairs_training:num_pairs_training+num_pairs_validation]]

    data_output_training = data_output[indices[0:num_pairs_training]]

    data_output_validation = data_output[indices[num_pairs_training:num_pairs_training+num_pairs_validation]]

    print("> Size of the data sets, sorted and separated in training/validation:")
    print("> Data input Training    : ",data_input_training.shape)
    print("> Data input Validation  : ",data_input_validation.shape)
    print("> Data output Training   : ",data_output_training.shape)
    print("> Data output Validation : ",data_output_validation.shape)

    # Dataset
    dataset = {}
    dataset["size_output"]         = len(params_outputOrdering)
    dataset["size_input"]          = len(params_inputOrdering)
    dataset["Input_training"]      = data_input_training
    dataset["Expected_training"]   = data_output_training
    dataset["Input_validation"]    = data_input_validation
    dataset["Expected_validation"] = data_output_validation

    return dataset,mins_norm,maxs_norm

def TrainNetwork_workflow(
    config_file,
    config_name,
    dataset_file
):
    config_parameters = json.load(open(config_file,"r"))

    if config_name not in config_parameters:
        raise Exception(config_name + "not in [" + ",".join(config_parameters) + "]")
    
    params = config_parameters[config_name]

    pprint.pprint(params)

    assert len(params["input_names"]) == params["size_input"]
    assert len(params["output_names"]) == params["size_output"]

    assert len(params["input_names"]) == len(params["input_normalization"])

    model_state = params["contextNN_state"][0:3]
    model_state = getClusterModelParams(model_state,params["contextNN_state"][3])

    model_state["Cluster_input_size"] = params["Cluster_input_size"]
    model_state["FC_num_layers"]  = params["FC_num_layers"]
    model_state["FC_num_neurons"] = params["FC_num_neurons"]
    model_state["FC_activations"] = params["FC_activations"]

    model = ClusterNetworkModel(
        FC_num_layers               = model_state["FC_num_layers"],
        FC_num_neurons              = model_state["FC_num_neurons"],
        FC_activations              = model_state["FC_activations"],
        Cluster_input_size          = model_state["Cluster_input_size"],
        number_of_pairs_FuncContext = model_state["pairs"],#number_of_pairs_FuncContext,
        number_of_layers            = model_state["layers"],#number_of_layers,
        number_of_neurons           = model_state["neurons"],#number_of_neurons,
        activation_function         = model_state["activation"],#activation_function
        verbose                     = False
    )

    dataset,mins_norm,maxs_norm = read_dataset(
        dataset_file,
        params["PartionningTrainingValidation"],
        params["input_names"],
        params["output_names"],
        params["input_normalization"]
    )

    trainer = NeuralNetworkTrainer(
        name               = config_name,
        logs_directory     = "NetworkTrainerLogs/" + os.path.basename(config_file).split(".json")[0],
        dataset            = dataset,
        loss_function_type = params["loss_function_type"]
    )

    trainer.train(
        model,
        epochs = params["epochs"],
        mins_norm = mins_norm,
        maxs_norm = maxs_norm
    )

if __name__ == "__main__":

    project_root = os.getcwd().split("3DPrinting")[0]

    TrainNetwork_workflow(
        config_file  = os.path.join(project_root,"3DPrinting/INPUTS/NeuralNetworkConfig.json"),
        config_name  = "FullyConnectedNetwork_EcliNorm",
        dataset_file = os.path.join(project_root,"3DPrinting/OUTPUTS/GMSH_FILES/SamplingParameterSpaceTest_1.csv")
    )