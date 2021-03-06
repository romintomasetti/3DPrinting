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
from CreateMaterialField_Ellipse import *
from SolveHomogenizationProblem import *
from GenerateEllipse import GenerateEllipse
import json,pprint,os,numpy,io,sys,time,multiprocessing

Do_3D = True

def SamplingParametersLHS(param_file,out_dir,PLOT,do_computations,do_only,typeField,params_domain=None):

    # Check directory
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Read parameters
    with open(param_file,"r") as fin:
        params = json.load(fin)

    # Print parameters
    pprint.pprint(params)

    file_names = {}

    timings = {}

    # Generate sampling points for each experiment in the parameters file
    for experiment in params:
        if do_only != -1:
            if do_only != experiment:
                continue
        print("\t> ",SamplingParametersLHS.__name__," : ",experiment)
        sys.stdout.flush()
        start = time.time()
        # LHS sampling
        if do_computations:
            if typeField != "Ellipse":
                points = LatinHyperCubicSampling(
                    params[experiment]["mins"],
                    params[experiment]["maxs"],
                    params[experiment]["num_pts"]
                )
            else:
                params_dom = json.load(open(params_domain,"r"))
                domain_limits = params_dom[experiment]["domain_limits"]
                length_x = domain_limits[0][1]-domain_limits[0][0]
                length_y = domain_limits[1][1]-domain_limits[1][0]
                if params_dom[experiment]["AngleType"] == "degree":
                    angle_max = 180.0
                else:
                    angle_max = numpy.pi
                points = LatinHyperCubicSampling(
                    [
                        0.05*length_x,0.05*length_y,
                        0.0,
                        domain_limits[0][0]/4.0,domain_limits[1][0]/4.0
                    ],
                    [
                        length_x/2.0,length_y/2.0,
                        angle_max,
                        domain_limits[0][1]/4.0,domain_limits[1][1]/4.0
                    ],
                    params[experiment]["num_pts"]
                )
                # Check ellipses lie inside domain and check aspect ratio
                aspect_max = params[experiment]["aspectRatio"]
                to_take = []
                for counter_point,point in enumerate(points):
                    ellipse = GenerateEllipse(
                        point[0],point[1],point[2],point[3],point[4]
                    )
                    if not ellipse.LiesInside(domain_limits[0],domain_limits[1])\
                        or ellipse.GetAspectRatio() > aspect_max\
                        or ellipse.GetAspectRatio() < 1.0/aspect_max:
                        pass
                    else:
                        to_take.append(counter_point)
                print("\t\t> Selecting ",len(to_take)," ellipses out of ",len(points)," !")
                sys.stdout.flush()
                points = points[to_take]
        # Save points
        file_names[experiment] = os.path.join(out_dir,experiment)
        if do_computations:
            numpy.save(file_names[experiment],points)
        file_names[experiment] += ".npy"

        #print("> Saved to : ",file_names[experiment])
        
        # If dimension equal to 3, plot scatter
        if "mins" in params[experiment] and\
            len(params[experiment]["mins"]) == 3 and PLOT and do_computations:
            import socket
            hostname = socket.gethostname()
            if hostname in ["lm3-m001"]:
                pass
            else:
                from mpl_toolkits.mplot3d import Axes3D
                import matplotlib.pyplot as plt
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.scatter(points[:,0], points[:,1], points[:,2], marker="o")
                ax.set_xlabel(params[experiment]["paramsName"][0])
                ax.set_ylabel(params[experiment]["paramsName"][1])
                ax.set_zlabel(params[experiment]["paramsName"][2])
                plt.show()

        timings[experiment] = time.time()-start

    return file_names,timings

def GenerateGMSH(general_params_file,files,out_dir,do_computations,typeField):

    with open(general_params_file,"r") as fin:
        parameters_domain_all = json.load(fin)
    
    gmsh_files = {}

    timings = {}

    # Loop over the files:
    for file in files:

        gmsh_files[os.path.join(out_dir,file)] = {}
        
        # Load file
        params = numpy.load(files[file])

        # Load domain parameters relative to the file
        domain_params = parameters_domain_all[file]

        # Loop over the samples and generate GMSH file
        print(100*"-")
        print("\t> ",GenerateGMSH.__name__," : ",file)
        print("\t> Number of sampling points : ",len(params))
        start = time.time()
        for counter,point in enumerate(params):
            print("\t\t" + 50 * "*-" + "*")
            print("\t\t> Doing domain ",counter," out of ",len(params))
            print("\t\t> eps_11 %.1e | eps_22 %.1e | alpha %.1e"%(point[0],point[1],point[2]))
            sys.stdout.flush()

            text_trap = io.StringIO()
            sys.stdout = text_trap

            if typeField != "Ellipse":
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
            else:
                MatFieldCreator = MaterialField_Ellipse(
                    "sample_" + str(counter),
                    os.path.join(out_dir,file),
                    domain_params["domain_limits"],
                    domain_params["nodes"],
                    point[0],
                    point[1],
                    point[2],
                    point[3],
                    point[4]
                )
                if do_computations:
                    MatFieldCreator.Create()
                gmsh_files[os.path.join(out_dir,file)]["_sample_" + str(counter)] = \
                    MatFieldCreator.ToGMSH(Do_3D = Do_3D,do_computations=do_computations)

            sys.stdout = sys.__stdout__
            sys.stdout.flush()
            print("\t\t> Mean generation time : ",(time.time()-start)/(counter+1))

        timings[file] = time.time()-start

    return gmsh_files,timings

def HomogenizationProblem(mat_props_file,gmsh_files,do_computations):

    # Contains all the folders in with cm3 solved the homogenization problem
    folders_with_homo = {}

    # Material properties information
    material_properties = json.load(open(mat_props_file,"r"))

    timings= {}

    # Loop over the experiments
    for experiment in gmsh_files:

        start = time.time()

        # Material properties specific to current experiment
        print("> ",HomogenizationProblem.__name__," : ",experiment)
        mat_props = material_properties[experiment.split("/")[-1]]

        # Initialization of the homogenization problem solver
        homogenization_solver = SolveHomogenization(
            "homogenization_solver",
            "output_homogenization"
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

        sys.stdout.flush()

        folders_with_homo[experiment],subprocs = homogenization_solver.Execute(do_computations)

        if len(subprocs) > 0:
            already_done = 0
            print("> Waiting for %d subprocesses..."%len(subprocs))
            while not all(subpro.poll() is not None for subpro in subprocs):
                counterdone = 0
                for proc in subprocs:
                    if proc.poll() is not None:
                        counterdone += 1
                if already_done < counterdone:
                    already_done = counterdone
                    print("> Waiting for %d subprocesses, already %d done !"%(len(subprocs),already_done))
                time.sleep(3)

        homogenization_solver.Analyse()

        sys.stdout.flush()

        timings[experiment.split("/")[-1]] = time.time()-start

    return folders_with_homo,timings

def GenerateDataset(folders,params_file,do_computations,out_dir,typeField):

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Read parameters of the datasets
    parameters = json.load(open(params_file,"r"))

    timings = {}

    # Loop over the datasets:
    for datasetname in parameters:

        start = time.time()

        folderdataname = parameters[datasetname]["datafolder"]

        print(50*"#")
        print("> Generating dataset ",datasetname)

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
            Warning("Experimental data " + folderdataname + " not found in list.")
            continue
        else:
            print("> Found " + folderdataname)

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
            #print(50*"-")
            #print("> Reading folder ",folder)
            tmp = folder.split("/")[-1]
            tmp = tmp.split("Realization_")[-1].split("binary")[0]

            # Parameter file
            param_file = os.path.join(
                folder,
                "Parameters_" + tmp + "binary.csv"
            )
            if os.path.exists(param_file) is False:
                raise Exception(param_file + " not found.")
            #print("> Reading parameter file ",param_file)
            params = numpy.zeros(dataset["size_input"])
            input_names  = ["" for x in params]
            output_names = []
            with open(param_file,"r") as fin:
                for line in fin:
                    if typeField != "Ellipse":
                        # We are not interested in the threshold
                        if "threshold" in line:
                            continue
                        elif "eps_11" in line:
                            params[0] = float(line.split(",")[1].split(";")[0])
                            input_names[0] = ("eps_11")
                        elif "eps_22" in line:
                            params[1] = float(line.split(",")[1].split(";")[0])
                            input_names[1] = ("eps_22")
                        elif "alpha" in line:
                            params[2] = float(line.split(",")[1].split(";")[0])
                            input_names[2] = ("alpha")
                        elif "lx" in line:
                            params[3] = float(line.split(",")[1].split(";")[0])
                            input_names[3] = ("lx")
                        elif "ly" in line:
                            params[4] = float(line.split(",")[1].split(";")[0])
                            input_names[4] = ("ly")
                        elif "nx" in line:
                            tmp       = float(line.split(",")[1].split(";")[0])
                            params[5] = params[3] / tmp
                            input_names[5] = ("nx")
                        elif "ny" in line:
                            tmp = float(line.split(",")[1].split(";")[0])
                            params[6] = params[4] / tmp
                            input_names[6] = ("ny")
                        elif "E_1" in line:
                            params[7] = float(line.split(",")[1].split(";")[0])
                            input_names[7] = ("E_1")
                        elif "E_2" in line:
                            params[8] = float(line.split(",")[1].split(";")[0])
                            input_names[8] = ("E_2")
                        elif "nu_1" in line:
                            params[9] = float(line.split(",")[1].split(";")[0])
                            input_names[9] = ("nu_1")
                        elif "nu_2" in line:
                            params[10] = float(line.split(",")[1].split(";")[0])
                            input_names[10] = ("nu_2")
                        elif "rho_1" in line:
                            params[11] = float(line.split(",")[1].split(";")[0])
                            input_names[11] = ("rho_1")
                        elif "rho_2" in line:
                            params[12] = float(line.split(",")[1].split(";")[0])
                            input_names[12] = ("rho_2")
                        else:
                            raise Exception("Cannot parse line " + line)
                    else:
                        par_value = float(line.split(",")[1].split(";")[0])
                        #print(line)
                        Found = False
                        for counter_tmp,name in enumerate(["a","b","t","x0","y0","lx_min","lx_max","ly_min","ly_max","nx","ny","E_1","E_2","nu_1","nu_2"]):
                            if name == line.split(",")[0]:
                                params[counter_tmp] = par_value
                                input_names[counter_tmp] = name
                                #print("> Found " + name)
                                #print(input_names)
                                #print(params)
                                Found = True
                                break
                        if not Found and "rho" not in line:
                            raise Exception("Cannot find : " + line)

            if typeField == "Ellipse":
                input_names[-1] = "Ltot"
                length_tot = params[input_names.index("lx_max")]-params[input_names.index("lx_min")]
                params[-1] = length_tot

            # Elastic tensor file
            TensorFile = os.path.join(
                folder,
                "E_0_GP_0_tangent.csv"
            )
            if os.path.exists(TensorFile) is False:
                raise Exception(TensorFile + " not found.")
            #print("> Reading tensor file ",TensorFile)
            stiffness4 = StiffnessTensor(TensorFile)

            # Assign input
            dataset["Input_training"][counter,:] = params

            # Assign output
            output = numpy.zeros(dataset["size_output"])
            if len(output) >= 1:
                output[0] = stiffness4.get(0,0,0,0)
                output_names.append("C_0000")
            if len(output) >= 2:
                output[1] = stiffness4.get(1,1,1,1)
                output_names.append("C_1111")
            if len(output) >= 3:
                output[2] = stiffness4.get(2,2,2,2)
                output_names.append("C_2222")
            if len(output) >= 4:
                output[3] = stiffness4.get(0,0,1,1)
                output_names.append("C_0011")
            if len(output) >= 5:
                output[4] = stiffness4.get(0,0,2,2)
                output_names.append("C_0022")
            if len(output) >= 6:
                output[5] = stiffness4.get(1,1,2,2)
                output_names.append("C_1122")
            if len(output) >= 7:
                output[6] = stiffness4.get(0,0,1,2)
                output_names.append("C_0012")
            if len(output) >= 8:
                output[7] = stiffness4.get(1,1,1,2)
                output_names.append("C_1112")
            if len(output) >= 9:
                output[8] = stiffness4.get(2,2,1,2)
                output_names.append("C_2212")
            if len(output) >= 10:
                output[9] = stiffness4.get(1,2,1,2)
                output_names.append("C_1212")
            if len(output) >= 11:
                output[10] = stiffness4.get(0,0,0,2)
                output_names.append("C_0002")
            if len(output) >= 12:
                output[11] = stiffness4.get(1,1,0,2)
                output_names.append("C_1102")
            if len(output) >= 13:
                output[12] = stiffness4.get(2,2,0,2)
                output_names.append("C_2202")
            if len(output) >= 14:
                output[13] = stiffness4.get(1,2,0,2)
                output_names.append("C_1202")
            if len(output) >= 15:
                output[14] = stiffness4.get(0,2,0,2)
                output_names.append("C_0202")
            if len(output) >= 16:
                output[15] = stiffness4.get(0,0,0,2)
                output_names.append("C_0002")
            if len(output) >= 17:
                output[16] = stiffness4.get(1,1,0,1)
                output_names.append("C_1101")
            if len(output) >= 18:
                output[17] = stiffness4.get(2,2,0,1)
                output_names.append("C_2201")
            if len(output) >= 19:
                output[18] = stiffness4.get(1,2,0,1)
                output_names.append("C_1201")
            if len(output) >= 20:
                output[19] = stiffness4.get(0,2,0,1)
                output_names.append("C_0201")
            if len(output) >= 21:
                output[20] = stiffness4.get(0,1,0,1)
                output_names.append("C_0101")
            dataset["Expected_training"][counter,:] = output
        
        with open(os.path.join(out_dir,datasetname + ".csv"),"w+") as fin:

            fin.write("input,%d;\n"%dataset["size_input"])
            fin.write("output,%d;\n"%dataset["size_output"])

            fin.write("ordering:" + ",".join(input_names) + "," + ",".join(output_names) + "\n")
            
            for row in range(dataset["Input_training"].shape[0]):

                for col in range(dataset["Input_training"].shape[1]):
                    fin.write("%.5e,"%dataset["Input_training"][row,col])
                for col in range(dataset["Expected_training"].shape[1]):
                    fin.write("%.5e"%dataset["Expected_training"][row,col])
                    if col < dataset["Expected_training"].shape[1] - 1:
                        fin.write(",")
                    else:
                        fin.write(";\n")

        timings[datasetname] = time.time()-start

    return timings
                
def GenerateData_workflow(
    do_computations,
    do_only,
    typeField
):

    timings = {}

    cpu_count = multiprocessing.cpu_count()

    print("> There are ",cpu_count," cpus.")

    # LHS sampling of the parameter space:
    sys.stdout.flush()
    start = time.time()
    param_sampling_files,timings_tmp = SamplingParametersLHS(
        param_file = do_computations["SamplingParametersLHS"][0],
        out_dir    = do_computations["SamplingParametersLHS"][1],
        PLOT       = do_computations["SamplingParametersLHS"][2],
        do_computations = do_computations["SamplingParametersLHS"][3],
        do_only = do_only,
        typeField = typeField,
        params_domain = do_computations["GenerateGMSH"][0]
    )
    timings["SamplingParametersLHS"] = timings_tmp
    timings["SamplingParametersLHS_all"] = time.time()-start

    if len(param_sampling_files) == 0:
        print("> Nothing was created. Exit.")
        sys.exit()

    # Generate GMSH files
    sys.stdout.flush()
    start = time.time()
    gmsh_files,timings_tmp = GenerateGMSH(
        general_params_file = do_computations["GenerateGMSH"][0],
        files               = param_sampling_files,
        out_dir             = do_computations["GenerateGMSH"][1],
        do_computations = do_computations["GenerateGMSH"][2],
        typeField = typeField
    )
    timings["GenerateGMSH"] = timings_tmp
    timings["GenerateGMSH_all"] = time.time()-start

    # Solve homogenization problem for all generated domains
    start = time.time()
    folders_with_homo,timings_tmp = HomogenizationProblem(
        mat_props_file = do_computations["HomogenizationProblem"][0],
        gmsh_files     = gmsh_files,
        do_computations = do_computations["HomogenizationProblem"][1],
    )
    timings["SolveHomogenization"] = timings_tmp
    timings["SolveHomogenization_all"] = time.time()-start
    print("*** INFO *** Homogenization took ",timings["SolveHomogenization_all"])

    # Generate dataset
    sys.stdout.flush()
    start = time.time()
    timings_tmp = GenerateDataset(
        folders             = folders_with_homo,
        params_file         = do_computations["GenerateDataset"][0],
        do_computations     = do_computations["GenerateDataset"][2],
        out_dir             = do_computations["GenerateDataset"][1],
        typeField = typeField
        )
    timings["GenerateDataset"] = timings_tmp
    timings["GenerateDataset_all"] = time.time()-start


    print(100*"#")
    number_of_samples = 0
    for folder in folders_with_homo:
        for f in folders_with_homo[folder]:
            number_of_samples += 1
    print("> Finished, with ",number_of_samples," samples.")
    pprint.pprint(timings)

if __name__ == "__main__":

    if len(sys.argv) == 2:
        do_only = sys.argv[1]
    else:
        do_only = -1

    typeField = "Ellipse"

    project_root = os.getcwd().split("3DPrinting")[0]

    print("> Project root is :",project_root)

    if typeField == "Ellipse":
        do_computations = {
            "SamplingParametersLHS" : [
                os.path.join(project_root,"3DPrinting/INPUTS/ParameterSpaceSampling_Ellipse.json"),
                os.path.join(project_root,"3DPrinting/OUTPUTS/ParameterSpaceSampling_Ellipse"),
                False,False],
            "GenerateGMSH"          : [
                os.path.join(project_root,"3DPrinting/INPUTS/DomainProperties_Ellipse.json"),
                os.path.join(project_root,"3DPrinting/OUTPUTS/GMSH_FILES_Ellipse"),
                False],
            "HomogenizationProblem" : [
                os.path.join(project_root,"3DPrinting/INPUTS/MaterialProperties_Ellipse.json"),
                False],
            "GenerateDataset"       : [
                os.path.join(project_root,"3DPrinting/INPUTS/DatasetParams_Ellipse.json"),
                os.path.join(project_root,"3DPrinting/OUTPUTS/Datasets_Ellipse"),
                True]
        }
    else:
        do_computations = {
            "SamplingParametersLHS" : [
                os.path.join(project_root,"3DPrinting/INPUTS/ParameterSpaceSampling.json"),
                os.path.join(project_root,"3DPrinting/OUTPUTS/ParameterSpaceSampling"),
                False,True],
            "GenerateGMSH"          : [
                os.path.join(project_root,"3DPrinting/INPUTS/DomainProperties.json"),
                os.path.join(project_root,"3DPrinting/OUTPUTS/GMSH_FILES"),
                False],
            "HomogenizationProblem" : [
                os.path.join(project_root,"3DPrinting/INPUTS/MaterialProperties.json"),
                True],
            "GenerateDataset"       : [
                os.path.join(project_root,"3DPrinting/INPUTS/DatasetParams.json"),
                os.path.join(project_root,"3DPrinting/OUTPUTS/Datasets"),
                True]
        }
    pprint.pprint(do_computations)
    GenerateData_workflow(
        do_computations = do_computations,
        do_only = do_only,
        typeField = typeField
    )
