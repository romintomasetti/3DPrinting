import json

import os

from CreateMaterialField import *

from SolveHomogenizationProblem import *

from TrainNeuralNetwork import *

from OptimizationOfProperties import *

from StiffnessTensor import *

from ClusterNetwork import *

def main(
    info_file,
    MaterialField_file,
    MaterialField_directory
) -> None:
    """
    This function creates material distributions, runs the homogenization process on it, 
    trains a Neural Network on the data set and finally solves an optimization problem.
    Parameters
    ----------
    info_file : str
        The name of the file containing the information to be displayed before the function starts.
    MaterialField_file : str
        The name of the JSON-formated file containing the information on the material fields to be generated.
    MaterialField_directory : str
        Directory in which all outputs are saved.
    """
    # Display brief information about the function and its goals
    with open(info_file, 'r') as fin:
        print(fin.read())

    # Check MaterialField_directory exists, create it if necessary
    if not os.path.exists(MaterialField_directory):
        os.makedirs(MaterialField_directory)
    
    # Read the material field file
    with open(MaterialField_file,'r') as fin:
        MaterialField_data = json.load(fin)

    Do_3D = True

    # List to store all the folders in which results can be found.
    folders = []

    # Create material random fields
    for MatField in MaterialField_data:
        print(50*"#")
        print("> Creating the material field ",MatField)
        try:
            tmp = MaterialField_data[MatField]["isOrthotropicTest"]
        except:
            tmp = False
        MatFieldCreator = MaterialField(
            MatField,
            MaterialField_directory,
            MaterialField_data[MatField]["Threshold"],
            MaterialField_data[MatField]["eps_11"],
            MaterialField_data[MatField]["eps_22"],
            MaterialField_data[MatField]["Alphas"],
            MaterialField_data[MatField]["Lengths"],
            MaterialField_data[MatField]["Nodes"],
            MaterialField_data[MatField]["Samples"],
            MaterialField_data[MatField]["consider_as_zero"],
            MaterialField_data[MatField]["AngleType"],
            tmp
        )
        MatFieldCreator.Create()
        gmsh_files = MatFieldCreator.ToGMSH(Do_3D = Do_3D)

        # Solve the homogenization problem for each sample of each configuration of the material random fields
        homogenization_solver = SolveHomogenization(
            "homogenization_solver"
        )

        homogenization_solver.assign_material_properties(
            MaterialField_data[MatField]["E_1"],
            MaterialField_data[MatField]["E_2"],
            MaterialField_data[MatField]["nu_1"],
            MaterialField_data[MatField]["nu_2"],
            MaterialField_data[MatField]["rho_1"],
            MaterialField_data[MatField]["rho_2"]
        )

        homogenization_solver.Create(gmsh_files,Do_3D = Do_3D)

        folders += homogenization_solver.Execute()

        homogenization_solver.Analyse()

    print(50*"#")
    print("> Results can be found in there:")
    print(folders)

    """
    Initialize the data set
    """
    # Dataset
    dataset = {}
    dataset["size_output"]         = 6
    dataset["size_input"]          = 13
    dataset["Input_training"]      = numpy.zeros((len(folders),dataset["size_input"]))
    dataset["Expected_training"]   = numpy.zeros((len(folders),dataset["size_output"]))
    dataset["Input_validation"]    = numpy.array([])
    dataset["Expected_validation"] = numpy.array([])

    for counter,folder in enumerate(folders):
        print(50*"#")
        print("> Reading folder ",folder)
        tmp = folder.split("/")[-1]
        tmp = tmp.split("Realization_")[-1].split("binary")[0]

        # Parameter file
        param_file = os.path.join(
            folder,
            "Parameters_" + tmp + "binary.csv"
        )
        if os.path.exists(param_file) is False:
            raise Exception(param_file + " not found.")
        print("> Reading parameter file ",param_file)
        params = numpy.zeros(dataset["size_input"])
        with open(param_file,"r") as fin:
            for line in fin:
                # We are not interested in the threshold
                if "threshold" in line:
                    continue
                elif "eps_11" in line:
                    params[0] = float(line.split(",")[1].split(";")[0])
                elif "eps_22" in line:
                    params[1] = float(line.split(",")[1].split(";")[0])
                elif "alpha" in line:
                    params[2] = float(line.split(",")[1].split(";")[0])
                elif "lx" in line:
                    params[3] = float(line.split(",")[1].split(";")[0])
                elif "ly" in line:
                    params[4] = float(line.split(",")[1].split(";")[0])
                elif "nx" in line:
                    tmp       = float(line.split(",")[1].split(";")[0])
                    params[5] = params[3] / tmp
                elif "ny" in line:
                    tmp = float(line.split(",")[1].split(";")[0])
                    params[6] = params[4] / tmp
                elif "E_1" in line:
                    params[7] = float(line.split(",")[1].split(";")[0])
                elif "E_2" in line:
                    params[8] = float(line.split(",")[1].split(";")[0])
                elif "nu_1" in line:
                    params[9] = float(line.split(",")[1].split(";")[0])
                elif "nu_2" in line:
                    params[10] = float(line.split(",")[1].split(";")[0])
                elif "rho_1" in line:
                    params[11] = float(line.split(",")[1].split(";")[0])
                elif "rho_2" in line:
                    params[12] = float(line.split(",")[1].split(";")[0])
                else:
                    raise Exception("Cannot parse line " + line)
                
        # Elastic tensor file
        TensorFile = os.path.join(
            folder,
            "E_0_GP_0_tangent.csv"
        )
        if os.path.exists(TensorFile) is False:
            raise Exception(TensorFile + " not found.")
        print("> Reading tensor file ",TensorFile)
        stiffness4 = StiffnessTensor(TensorFile)

        """
        Assign inputs
        """
        dataset["Input_training"][counter,:] = params

        """
        Assign outputs
        """
        output = numpy.zeros(dataset["size_output"])
        if len(output) >= 1:
            output[0] = stiffness4.get(0,0,0,0)
        if len(output) >= 2:
            output[1] = stiffness4.get(1,1,1,1)
        if len(output) >= 3:
            output[2] = stiffness4.get(2,2,2,2)
        if len(output) >= 4:
            output[3] = stiffness4.get(0,0,1,1)
        if len(output) >= 5:
            output[4] = stiffness4.get(0,0,2,2)
        if len(output) >= 6:
            output[5] = stiffness4.get(1,1,2,2)
        dataset["Expected_training"][counter,:] = output

    """
    Write data set to CSV for later use.
    """
    with open(os.path.join(MaterialField_directory,"DataSet.csv"),"w+") as fin:
        fin.write("input,%d;\n"%dataset["size_input"])
        fin.write("output,%d;\n"%dataset["size_output"])
        for row in range(dataset["Input_training"].shape[0]):
            for col in range(dataset["Input_training"].shape[1]):
                fin.write("%.5e,"%dataset["Input_training"][row,col])
            for col in range(dataset["Expected_training"].shape[1]):
                fin.write("%.5e"%dataset["Expected_training"][row,col])
                if col < dataset["Expected_training"].shape[1] - 1:
                    fin.write(",")
                else:
                    fin.write(";\n")
    """
    Initialize the Neural Network Trainer.
    """
    
    trainer = NeuralNetworkTrainer(
        name           = "NeuralNetworkTrainer",
        logs_directory = "./NetworkTrainerLogs/",
        dataset        = dataset
    )

    """
    Initialize the Network
    """
    model_state = [4,4,50]
    model_state = getClusterModelParams(model_state,"tanh")
    model = ClusterNetworkModel(
        number_of_pairs_FuncContext = model_state["pairs"],#number_of_pairs_FuncContext,
        number_of_layers            = model_state["layers"],#number_of_layers,
        number_of_neurons           = model_state["neurons"],#number_of_neurons,
        activation_function         = model_state["activation"],#activation_function
        verbose                     = False
    )

    """
    Train
    """
    trainer.train(
        model,
        epochs = 1000
    )


if __name__ == "__main__":
    main(
        info_file = "./DOC/brief_info.txt",
        MaterialField_file = "./INPUTS/MaterialField.json",
        MaterialField_directory = "./OUTPUTS/"
    )