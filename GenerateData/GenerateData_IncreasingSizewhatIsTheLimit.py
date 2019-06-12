from GenerateData import *


if __name__ == "__main__":

    if len(sys.argv) == 2:
        do_only = sys.argv[1]
    else:
        do_only = -1

    project_root = os.getcwd().split("3DPrinting")[0]

    print("> Project root is :",project_root)

    do_computations = {
        "SamplingParametersLHS" : [
            os.path.join(project_root,"3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/ParameterSpaceSampling.json"),
            os.path.join(project_root,"3DPrinting/OUTPUTS/ParameterSpaceSampling"),
            False,True],
        "GenerateGMSH"          : [
            os.path.join(project_root,"3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/DomainProperties.json"),
            os.path.join(project_root,"3DPrinting/OUTPUTS/GMSH_FILES"),
            True],
        "HomogenizationProblem" : [
            os.path.join(project_root,"3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/MaterialProperties.json"),
            True],
        "GenerateDataset"       : [
            os.path.join(project_root,"3DPrinting/INPUTS/GenerateData_SpeedAndSizeTesting/DatasetParams.json"),
            os.path.join(project_root,"3DPrinting/OUTPUTS/Datasets"),
            True]
    }
    GenerateData_workflow(
        do_computations = do_computations,
        do_only = do_only
    )
