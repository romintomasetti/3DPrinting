from GenerateData import *

do_computations = {
    "SamplingParametersLHS" : [
        "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/ParameterSpaceSampling.json",
        "/home/romin/NewMethodsComputational/3DPrinting/OUTPUTS/ParameterSpaceSampling",
        False,False],
    "GenerateGMSH"          : [
        "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/DomainProperties.json",
        "/home/romin/NewMethodsComputational/3DPrinting/OUTPUTS/GMSH_FILES",
        False],
    "HomogenizationProblem" : [
        "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/MaterialProperties.json",
        False],
    "GenerateDataset"       : [
        "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/DatasetParams.json",
        "/home/romin/NewMethodsComputational/3DPrinting/OUTPUTS/Datasets",
        True]
}
GenerateData_workflow(
    do_computations = do_computations
)