from GenerateData import *


if __name__ == "__main__":

    if len(sys.argv) == 2:
        do_only = sys.argv[1]
    else:
        do_only = -1

    do_computations = {
        "SamplingParametersLHS" : [
            "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/ParameterSpaceSampling.json",
            "/home/romin/NewMethodsComputational/3DPrinting/OUTPUTS/ParameterSpaceSampling",
            False,True],
        "GenerateGMSH"          : [
            "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/DomainProperties.json",
            "/home/romin/NewMethodsComputational/3DPrinting/OUTPUTS/GMSH_FILES",
            True],
        "HomogenizationProblem" : [
            "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/MaterialProperties.json",
            True],
        "GenerateDataset"       : [
            "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/DatasetParams.json",
            "/home/romin/NewMethodsComputational/3DPrinting/OUTPUTS/Datasets",
            True]
    }
    GenerateData_workflow(
        do_computations = do_computations,
        do_only = do_only
    )