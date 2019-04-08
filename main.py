import json

import os

from CreateMaterialField import *

from SolveHomogenizationProblem import *

from TrainNeuralNetwork import *

from OptimizationOfProperties import *

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

        homogenization_solver.Execute()

        homogenization_solver.Analyse()


if __name__ == "__main__":
    main(
        info_file = "./DOC/brief_info.txt",
        MaterialField_file = "./INPUTS/MaterialField.json",
        MaterialField_directory = "./OUTPUTS/"
    )