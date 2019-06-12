"""
Generate data:
    0) Generate LHS sampling of the parameter space -> SamplingParametersLHS
    1) Generate domain by KL sampling               -> GenerateGMSH : MaterialField
    2) Generate GMSH file                           -> GenerateGMSH : MaterialField
    3) Homogenization (cm3)                         -> SolveHomogenization
    4) Generate dataset                             -> GenerateDataset
"""
from LatinHyperCubicSampling import LatinHyperCubicSampling
from CreateMaterialField import *
from SolveHomogenizationProblem import *
import json,pprint,os,numpy,io,sys,time

Do_3D = True

def SamplingParametersLHS(param_file,out_dir,PLOT,do_computations):

    # Check directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Read parameters
    with open(param_file,"r") as fin:
        params = json.load(fin)

    # Print parameters
    pprint.pprint(params)

    file_names = {}

    # Generate sampling points for each experiment in the parameters file
    for experiment in params:
        # LHS sampling
        if do_computations:
            points = LatinHyperCubicSampling(
                params[experiment]["mins"],
                params[experiment]["maxs"],
                params[experiment]["num_pts"]
            )
        # Save points
        file_names[experiment] = os.path.join(out_dir,experiment)
        if do_computations:
            numpy.save(file_names[experiment],points)
        file_names[experiment] += ".npy"
        
        # If dimension equal to 3, plot scatter
        if len(params[experiment]["mins"]) == 3 and PLOT and do_computations:
            from mpl_toolkits.mplot3d import Axes3D
            import matplotlib.pyplot as plt
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(points[:,0], points[:,1], points[:,2], marker="o")
            ax.set_xlabel(params[experiment]["paramsName"][0])
            ax.set_ylabel(params[experiment]["paramsName"][1])
            ax.set_zlabel(params[experiment]["paramsName"][2])
            plt.show()

    return file_names

def GenerateGMSH(general_params_file,files,out_dir,do_computations):

    with open(general_params_file,"r") as fin:
        parameters_domain_all = json.load(fin)
    
    gmsh_files = {}

    # Loop over the files:
    for file in files:

        gmsh_files[os.path.join(out_dir,file)] = {}
        
        # Load file
        params = numpy.load(files[file])

        # Load domain parameters relative to the file
        domain_params = parameters_domain_all[file]

        # Loop over the samples and generate GMSH file
        print(100*"-")
        print("\t> Doing ",file)
        print("\t> Number of sampling points : ",len(params))
        start = time.time()
        for counter,point in enumerate(params):
            print("\t\t" + 50 * "*-" + "*")
            print("\t\t> Doing domain ",counter," out of ",len(params))
            print("\t\t> eps_11 %.1e | eps_22 %.1e | alpha %.1e"%(point[0],point[1],point[2]))

            text_trap = io.StringIO()
            sys.stdout = text_trap

            MatFieldCreator = MaterialField(
                "sample_" + str(counter),
                os.path.join(out_dir,file),
                domain_params["threshold"],
                [point[0]],# * numpy.ones(domain_params["num_samples"],dtype=numpy.float64),
                [point[1]],# * numpy.ones(domain_params["num_samples"],dtype=numpy.float64),
                [point[2]],# * numpy.ones(domain_params["num_samples"],dtype=numpy.float64),
                domain_params["lengths"],
                domain_params["nodes"],
                domain_params["num_samples"],
                domain_params["consider_as_zero"],
                domain_params["AngleType"],
                isOrthotropicTest = False,
                RatioHighestSmallestEig = 100.0
            )
            if do_computations:
                MatFieldCreator.Create()
            gmsh_files[os.path.join(out_dir,file)]["_sample_" + str(counter)] = \
                MatFieldCreator.ToGMSH(Do_3D = Do_3D)

            sys.stdout = sys.__stdout__

            print("\t\t> Mean generation time : ",(time.time()-start)/(counter+1))

    return gmsh_files

def HomogenizationProblem(mat_props_file,gmsh_files,do_computations):

    # Contains all the folders in with cm3 solved the homogenization problem
    folders_with_homo = {}

    # Material properties information
    material_properties = json.load(open(mat_props_file,"r"))

    # Loop over the experiments
    for experiment in gmsh_files:

        # Material properties specific to current experiment
        print(experiment)
        mat_props = material_properties[experiment.split("/")[-1]]

        # Initialization of the homogenization problem solver
        homogenization_solver = SolveHomogenization(
            "homogenization_solver"
        )

        # Assignment of material propeties
        homogenization_solver.assign_material_properties(
            mat_props["E_1"],
            mat_props["E_2"],
            mat_props["nu_1"],
            mat_props["nu_2"],
            mat_props["rho_1"],
            mat_props["rho_2"]
        )

        tmp_struct = gmsh_files[experiment]
        files = []
        for tmp in tmp_struct:
            for tmptmp in tmp_struct[tmp]:
                files.append(tmptmp)

        homogenization_solver.Create(
            files,
            Do_3D = Do_3D
        )

        folders_with_homo[experiment] = homogenization_solver.Execute(do_computations)

        homogenization_solver.Analyse()

    return folders_with_homo

def GenerateDataset(folders,params_file,do_computations,out_dir):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Read parameters of the datasets
    parameters = json.load(open(params_file,"r"))

    # Loop over the datasets:
    for datasetname in parameters:

        print(50*"#")
        print("> Generating dataset ",datasetname)

        folderdataname = parameters[datasetname]["datafolder"]

        # Loop over the experiments (folders) to find the right one
        FOUND = False
        for experiment in folders:
            if experiment.split("/")[-1] != folderdataname:
                continue
            else:
                FOUND = True
                folders_forDataset = folders[experiment]
                break
        if not FOUND:
            raise Exception("Experimental data " + folderdataname + " not found in list.")

        # Dataset
        dataset = {}
        # Output is all the elts of the elastic stiffness tensor
        dataset["size_output"]         = parameters[datasetname]["sizeoutput"]
        # Input is eps_11 eps_22 alpha lx ly nx ny E1 E2 nu1 nu2 rho1 rho2
        dataset["size_input"]          = parameters[datasetname]["sizeinput"]

        dataset["Input_training"]      = \
            numpy.zeros(
                (
                    len(folders_forDataset),
                    dataset["size_input"]
                )
            )

        dataset["Expected_training"]   = \
            numpy.zeros(
                (
                    len(folders_forDataset),
                    dataset["size_output"]
                )
            )

        dataset["Input_validation"]    = []

        dataset["Expected_validation"] = []

        for counter,folder in enumerate(folders_forDataset):
            print(50*"-")
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

            # Assign input
            dataset["Input_training"][counter,:] = params

            # Assign output
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
            if len(output) >= 7:
                output[6] = stiffness4.get(0,0,1,2)
            if len(output) >= 8:
                output[7] = stiffness4.get(1,1,1,2)
            if len(output) >= 9:
                output[8] = stiffness4.get(2,2,1,2)
            if len(output) >= 10:
                output[9] = stiffness4.get(1,2,1,2)
            if len(output) >= 11:
                output[10] = stiffness4.get(0,0,0,2)
            if len(output) >= 12:
                output[11] = stiffness4.get(1,1,0,2)
            if len(output) >= 13:
                output[12] = stiffness4.get(2,2,0,2)
            if len(output) >= 14:
                output[13] = stiffness4.get(1,2,0,2)
            if len(output) >= 15:
                output[14] = stiffness4.get(0,2,0,2)
            if len(output) >= 16:
                output[15] = stiffness4.get(0,0,0,2)
            if len(output) >= 17:
                output[16] = stiffness4.get(1,1,0,1)
            if len(output) >= 18:
                output[17] = stiffness4.get(2,2,0,1)
            if len(output) >= 19:
                output[18] = stiffness4.get(1,2,0,1)
            if len(output) >= 20:
                output[19] = stiffness4.get(0,2,0,1)
            if len(output) >= 21:
                output[20] = stiffness4.get(0,1,0,1)
            dataset["Expected_training"][counter,:] = output
        
        with open(os.path.join(out_dir,datasetname + ".csv"),"w+") as fin:

            fin.write("input,%d;\n"%dataset["size_input"])
            fin.write("output,%d;\n"%dataset["size_output"])
            
            for row in range(dataset["Input_training"].shape[0]):

                tmp_size = 0

                for col in range(dataset["Input_training"].shape[1]):
                    tmp_size += 1
                    fin.write("%.5e,"%dataset["Input_training"][row,col])
                for col in range(dataset["Expected_training"].shape[1]):
                    tmp_size += 1
                    fin.write("%.5e"%dataset["Expected_training"][row,col])
                    if col < dataset["Expected_training"].shape[1] - 1:
                        fin.write(",")
                    else:
                        fin.write(";\n")
                
if __name__ == "__main__":

    timings = {}

    do_computations = {
        "SamplingParametersLHS" : False,
        "GenerateGMSH"          : False,
        "HomogenizationProblem" : False,
        "GenerateDataset"       : True
    }

    # LHS sampling of the parameter space:
    start = time.time()
    param_sampling_files = SamplingParametersLHS(
        param_file = "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/ParameterSpaceSampling.json",
        out_dir    = "/home/romin/NewMethodsComputational/3DPrinting/OUTPUTS/ParameterSpaceSampling",
        PLOT       = False,
        do_computations = do_computations["SamplingParametersLHS"]
    )
    timings["SamplingParametersLHS"] = time.time()-start

    # Generate GMSH files
    start = time.time()
    gmsh_files = GenerateGMSH(
        general_params_file = "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/DomainProperties.json",
        files               = param_sampling_files,
        out_dir             = "/home/romin/NewMethodsComputational/3DPrinting/OUTPUTS/GMSH_FILES",
        do_computations = do_computations["GenerateGMSH"]
    )
    timings["GenerateGMSH"] = time.time()-start

    # Solve homogenization problem for all generated domains
    start = time.time()
    folders_with_homo = HomogenizationProblem(
        mat_props_file = "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/MaterialProperties.json",
        gmsh_files     = gmsh_files,
        do_computations = do_computations["HomogenizationProblem"]
    )
    timings["SolveHomogenization"] = time.time()-start

    # Generate dataset
    start = time.time()
    GenerateDataset(
        folders             = folders_with_homo,
        params_file         = "/home/romin/NewMethodsComputational/3DPrinting/INPUTS/DatasetParams.json",
        do_computations     = do_computations["GenerateDataset"],
        out_dir             = "/home/romin/NewMethodsComputational/3DPrinting/OUTPUTS/Datasets"
    )
    timings["GenerateDataset"] = time.time()-start


print(100*"#")
print("> Finished, with ",len(folders_with_homo)," samples.")
pprint.pprint(timings)