import os,sys

from datetime import datetime

import time

import numpy

import shutil

from shutil import copyfile

from StiffnessTensor import *

import pathlib

def delete_folder(path) :
    pth = pathlib.Path(path)
    for sub in pth.iterdir() :
        if sub.is_dir() :
            delete_folder(sub)
        else :
            sub.unlink()

class SolveHomogenization:
    """
    Solve 2D bi-material homogenization problem.
    """
    def __init__(self,name,outname) -> None:
        """
        Initialization.
        Parameters
        ----------
        name : str
            Name of the solver.
        """
        self.name    = name
        self.outname = outname

    def Analyse(self) -> None:
        """
        Analyse the results of the execued processes.
        """
        for python_file in self.ToExecute:
            folder = os.path.dirname(python_file)
            folder = os.path.join(
                folder,
                os.path.splitext(
                    os.path.basename(
                        python_file
                    )
                )[0]
            ) + "_CM3"
            sys.stdout = open(os.path.join(
                folder,self.outname + "_analysis"
            ),"w+")
            print("\t> Analyzing ",python_file," stored in ",folder)
            try:
                with open(os.path.join(folder,"OrthotropicElasticProperties.csv"),"w+") as fin:
                    stiffness4 = StiffnessTensor(
                        os.path.join(folder,"E_0_GP_0_tangent.csv")
                    )
                    E, nu = stiffness4.get_isotropic_properties()
                    print("\t> E (isotropic): ",E)
                    print("\t> nu(isotropic): ",nu)
                    E_1, E_2, E_3, nu_12, nu_21, nu_13, nu_31, nu_23, nu_32, G_12, G_23, G_31 = \
                        stiffness4.get_orthotropic_properties()
                    print("\t> E_1 (orthotropic) : ",E_1)
                    fin.write("E_1,%.5e;\n"%E_1)
                    print("\t> E_2 (orthotropic) : ",E_2)
                    fin.write("E_2,%.5e;\n"%E_2)
                    print("\t> E_3 (orthotropic) : ",E_3)
                    fin.write("E_3,%.5e;\n"%E_3)
                    print("\t> nu12(orthotropic) : ",nu_12)
                    fin.write("nu_12,%.5e;\n"%nu_12)
                    print("\t> nu21(orthotropic) : ",nu_21)
                    fin.write("nu_21,%.5e\n"%nu_21)
                    print("\t> nu13(orthotropic) : ",nu_13)
                    fin.write("nu_13,%.5e;\n"%nu_13)
                    print("\t> nu31(orthotropic) : ",nu_31)
                    fin.write("nu_31,%.5e;\n"%nu_31)
                    print("\t> nu23(orthotropic) : ",nu_23)
                    fin.write("nu_23,%.5e;\n"%nu_23)
                    print("\t> nu32(orthotropic) : ",nu_32)
                    fin.write("nu_32,%.5e;\n"%nu_32)
                    print("\t> G_23(orthotropic) : ",G_23)
                    fin.write("G_23,%.5e;\n"%G_23)
                    print("\t> G_31(orthotropic) : ",G_31)
                    fin.write("G_31,%.5e;\n"%G_31)
                    print("\t> G_12(orthotropic) : ",G_12)
                    fin.write("G_12,%.5e;\n"%G_12)
            except Exception as e:
                print(e)
                pass
            sys.stdout = sys.__stdout__


    def Execute(self,do_computations,minutes_for_homo=30,MAX_SUBPROCESSES=70) -> list:
        """
        Execute Python 2 files stored in self.ToExecute list.
        Returns
        -------
        List of strings:
            List of the created folders in which results are stored.
        """
        folders = []
        subprocesses = []
        for python_file in self.ToExecute:
            print("\t> Executing ",python_file)
            # Timing the method
            start = time.time()
            # Store old working directory
            old_dir = os.getcwd()
            # Create a folder
            new_folder = \
                os.path.join(
                    os.path.dirname(
                        python_file
                    ),
                    os.path.splitext(os.path.basename(python_file))[0]
                )
            folders.append(
                new_folder + "_CM3"
            )
            proc = None
            if do_computations:
                if not os.path.exists(new_folder +"_CM3"):
                    os.makedirs(
                        new_folder + "_CM3"
                    )
                else:
                    #shutil.rmtree(new_folder + "_CM3")
                    #os.makedirs(
                    #    new_folder + "_CM3"
                    #)
                    # Depreciating shutil because it removes the folder as well, and sometimes ends in race condition with os makedirs ?
                    delete_folder(new_folder + "_CM3")
                # Redirect output
                """sys.stdout = open(os.path.join(
                    folders[-1],self.outname + "_executeCM3"
                ),"w+")"""
                # Copy the Python file in the new folder
                copyfile(
                    python_file,
                    os.path.join(new_folder + "_CM3",os.path.basename(python_file))
                )
                # Append material properties to parameter file
                tmp = os.path.splitext(os.path.basename(python_file))[0]
                tmp_1 = os.path.join(
                    os.path.dirname(python_file),
                    "Parameters_" + tmp.split("_",1)[-1] + ".csv"
                )
                tmp_2 = os.path.join(
                    new_folder + "_CM3",
                    "Parameters_" + tmp.split("_",1)[-1] + ".csv"
                )
                with open(tmp_1,"a") as fin:
                    fin.write("E_1,%.5e;\n"%self.E_1)
                    fin.write("E_2,%.5e;\n"%self.E_2)
                    fin.write("nu_1,%.5e;\n"%self.nu_1)
                    fin.write("nu_2,%.5e;\n"%self.nu_2)
                    fin.write("rho_1,%.5e;\n"%self.rho_1)
                    fin.write("rho_2,%.5e;\n"%self.rho_2)
                # Copy the parameter file
                copyfile(tmp_1,tmp_2)
                # Copy the GMSH file in the new folder
                tmp = os.path.splitext(
                        os.path.basename(python_file)
                    )[0] + ".geo"
                copyfile(
                    os.path.join(os.path.dirname(python_file),tmp),
                    os.path.join(new_folder + "_CM3",tmp)
                )
                # Change directory to this folder
                os.chdir(
                    new_folder + "_CM3"
                )
                # Run the Python file
                import socket
                if socket.gethostname() in ["romin-X550CL"]:
                    print("\t> Sequential execution, without subprocess and sbatch...")
                    os.system(
                        "python2 %s > %s.out 2>&1"%(
                            os.path.basename(python_file),
                            os.path.splitext(os.path.basename(python_file))[0]
                        )
                    )
                else:
                    with open("to_bash_execute.sh","w+") as fout:
                        #fout.write("srun python2 %s\n"%os.path.basename(python_file))
                        fout.write("#!/bin/bash\n")
                        fout.write("#\n")
                        fout.write("#SBATCH --job-name=test\n")
                        fout.write("#SBATCH --output=res.txt\n")
                        fout.write("#\n")
                        fout.write("#SBATCH --ntasks=1\n")
                        fout.write("#SBATCH --time=%d:00\n"%minutes_for_homo)
                        fout.write("#SBATCH --mem-per-cpu=8000\n")
                        fout.write("module load Python/2.7.14-foss-2017b\n")
                        fout.write("which python2\n")
                        fout.write("mpirun python2 %s\n"%os.path.basename(python_file))
                    #os.system("bash to_bash_execute.sh> %s.out 2>&1"%os.path.splitext(os.path.basename(python_file))[0])
                    import subprocess
                    subprocesses.append(subprocess.Popen(
                        ["sbatch","-W","to_bash_execute.sh"],stdout=open("SUBPROCESS_stdout","w+"),stderr=open("SUBPROCESS_stderr","w+")))
                    #proc.wait()
                    print("> Subprocess done !")
                """
                os.system(
                    "python2 %s"%os.path.basename(python_file) + 
                    " > %s.out 2>&1"%os.path.splitext(os.path.basename(python_file))[0]
                )
                """
                while len(subprocesses) > MAX_SUBPROCESSES:
                    to_pop_out = []
                    for subproc in range(len(subprocesses)):
                        if subprocesses[subproc].poll() is not None:
                            to_pop_out.append(subproc)
                    to_pop_out = numpy.flip(to_pop_out)
                    for pop_out in to_pop_out:
                        subprocesses.pop(pop_out)
                    time.sleep(3)
                print("\t> Done in ",time.time()-start," seconds.")
                # Roll back to old working directory
                os.chdir(old_dir)
                sys.stdout.flush()
                # Rool back to stdout
                #sys.stdout = sys.__stdout__
        return folders,subprocesses

    def Create(self,files,Do_3D = False) -> None:
        """
        Create the files to be executed by cm3/dG3Dpy library.
        It is ecessary to go throught the following steps:
            1) Create file with python 2 content
            2) Execute it with os.system()
        because dG3Dpy is not Python3-compatible.
        Parameters
        ----------
        files : list(str)
            List of GMSH files corresponding to each sample of each configuration.
        """
        self.ToExecute = []
        for file in files:
            filename = os.path.splitext(file)[0] + ".py"
            #file = os.path.basename(file)
            print("\t> Creating ",filename)
            self.ToExecute.append(filename)

            with open(filename,"w+") as fin:
                # Write some information
                now = datetime.now()
                date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
                fin.write("# Author : Romin Tomasetti\n")
                fin.write("# cm3/dG3Dpy homogenization for " + file + "\n")
                fin.write("# File created on " + date_time + "\n")
                # Write module import
                fin.write("from gmshpy import *\n")
                fin.write("from dG3Dpy import *\n")
                fin.write("import os\n")
                
                # Write options:
                fin.write("\n")
                if Do_3D:
                    fin.write("Do_3D = True\n")
                else:
                    fin.write("Do_3D = False\n")
                    
                fin.write("Do_planeStress = False\n")

                # Write material properties
                fin.write("\n\"\"\"\nMaterial properties\n\"\"\"\n")
                # Young modulus
                fin.write("E_1 = %.10f\n"%self.E_1)
                fin.write("E_2 = %.10f\n"%self.E_2)
                # Poisson ratio
                fin.write("nu_1 = %.10f\n"%self.nu_1)
                fin.write("nu_2 = %.10f\n"%self.nu_2)
                # Bulk modulus
                fin.write("K_1 = %.10f\n"%self.K_1)
                fin.write("K_2 = %.10f\n"%self.K_2)
                # Shear modulus
                fin.write("mu_1 = %.10f\n"%self.mu_1)
                fin.write("mu_2 = %.10f\n"%self.mu_2)
                # Bulk mass
                fin.write("rho_1 = %.10f\n"%self.rho_1)
                fin.write("rho_2 = %.10f\n"%self.rho_2)
                # Material laws
                fin.write("\n\"\"\"\nCreation of material law\n\"\"\"\n")
                # Material 1 is on Physical Surface 1
                fin.write(
                    "law_1 = dG3DLinearElasticMaterialLaw(1,rho_1,E_1,nu_1)\n"
                )
                # Material 2 is on Physical Surface 2
                fin.write(
                    "law_2 = dG3DLinearElasticMaterialLaw(2,rho_2,E_2,nu_2)\n"
                )
                fin.write("print(\"> Material laws created successfully.\")\n")
                # Geometry
                fin.write("\n\"\"\"\nGeometry\n\"\"\"\n")
                fin.write("mesh_file = os.path.splitext(\"%s\")[0]\n"%os.path.basename(file))
                
                fin.write("if Do_3D is False:\n")

                fin.write("\tif not os.path.exists(mesh_file + \".msh\"):\n"
                    "\t\tif os.path.exists(mesh_file + \".geo\"):\n"
                        "\t\t\tos.system(\"gmsh -2 \" + mesh_file + \".geo\")\n"
                    "\t\telse:\n"
                        "\t\t\traise Exception(mesh_file + \" doesn't exist.\")\n"
                    "\tmesh_file = mesh_file + \".msh\"\n"
                )
                
                fin.write("else:\n")
                fin.write("\tif not os.path.exists(mesh_file + \".msh\"):\n"
                    "\t\tif os.path.exists(mesh_file + \".geo\"):\n"
                        "\t\t\tos.system(\"gmsh -3 \" + mesh_file + \".geo\")\n"
                    "\t\telse:\n"
                        "\t\t\traise Exception(mesh_file + \" doesn't exist.\")\n"
                    "\tmesh_file = mesh_file + \".msh\"\n"
                )

                # Domain
                fin.write("\n\"\"\"\nCreation of domain\n\"\"\"\n")
                # Domain of material 1
                fin.write("if Do_3D is False:\n")
                fin.write("\tdomain_1 = dG3DDomain(3,1,0,1,0,2)\n")
                fin.write("\tif Do_planeStress is True and Do_3D is False:\n")
                fin.write("\t\tdomain_1.setPlaneStressState(True)\n")
                fin.write("else:\n")
                fin.write("\tdomain_1 = dG3DDomain(3,2221,0,1,0,3)\n")
                # Domain of material 2
                fin.write("if Do_3D is False:\n")
                fin.write("\tdomain_2 = dG3DDomain(3,2,0,2,0,2)\n")
                fin.write("\tif Do_planeStress is True and Do_3D is False:\n")
                fin.write("\t\tdomain_2.setPlaneStressState(True)\n")
                fin.write("else:\n")
                fin.write("\tdomain_2 = dG3DDomain(3,2222,0,2,0,3)\n")
                fin.write("print(\"> Domains created successfully.\")\n")
                # Solver parameters
                fin.write("\n\"\"\"\nSolver parameters\n\"\"\"\n")
                # Gmm=0 (default) Taucs=1 PETsc=2
                fin.write("sol = 2\n")
                # StaticLinear=0 (default) StaticNonLinear=1
                fin.write("soltype =1\n")
                # number of step (used only if soltype=1)
                fin.write("nstep = 1\n")
                # Final time (used only if soltype=1)
                fin.write("ftime =0.\n")
                # relative tolerance for NR scheme (used only if soltype=1) 
                fin.write("tol=1.e-6\n")
                # Number of step between 2 archiving (used only if soltype=1)
                fin.write("nstepArch=1\n")
                # Displacement elimination =0 Multiplier elimination = 1 Displacement+ multiplier = 2 
                fin.write("system = 1\n")
                # load control = 0 arc length control euler = 1
                fin.write("control = 0\n")
                # Solver
                fin.write("\n\"\"\"\nCreation of the solver\n\"\"\"\n")
                fin.write("mysolver = nonLinearMechSolver(3)\n")
                fin.write("mysolver.loadModel(mesh_file)\n")
                # Domain of material 1
                fin.write("mysolver.addDomain(domain_1)\n")
                # Domain of material 2
                fin.write("mysolver.addDomain(domain_2)\n")
                # Law of material 1
                fin.write("mysolver.addMaterialLaw(law_1)\n")
                # Law of material 2
                fin.write("mysolver.addMaterialLaw(law_2)\n")
                fin.write("mysolver.Scheme(soltype)\n")
                fin.write("mysolver.Solver(sol)\n")
                fin.write("mysolver.snlData(nstep,ftime,tol)\n")
                fin.write("mysolver.setSystemType(system)\n")
                fin.write("mysolver.setControlType(control)\n")
                fin.write("mysolver.stiffnessModification(bool(1))\n")
                fin.write("mysolver.iterativeProcedure(bool(1))\n")
                fin.write("mysolver.setMessageView(bool(1))\n")
                fin.write("print(\"> Solver created successfully.\")\n")
                # Boundary conditions
                fin.write("\n\"\"\"\nBoundary conditions\n\"\"\"\n")
                # nonLinearPeriodicBC(tag,dimension)
                fin.write("if Do_3D is False:\n")
                fin.write("\tmicroBC = nonLinearPeriodicBC(3,2)\n")
                fin.write("else:\n")
                fin.write("\tmicroBC = nonLinearPeriodicBC(3,3)\n")
                fin.write("microBC.setOrder(1)\n")
                # Periodic boundary conditions on 4 lines of tag 1, 2, 3, 4
                fin.write("if Do_3D is False:\n")
                fin.write("\tmicroBC.setBCPhysical(1,4,3,2)\n")
                fin.write("else:\n")
                fin.write("\tmicroBC.setBCPhysical(88880,88883,88884,88881,88882,88885)\n")

                fin.write("method =0	# Periodic mesh = 0, Langrange interpolation = 1, Cubic spline interpolation =2,  FE linear= 3, FE Quad = 4\n")
                fin.write("degree = 2	# Order used for polynomial interpolation\n")
                fin.write("addvertex = 0 # Polynomial interpolation by mesh vertex = 0, Polynomial interpolation by virtual vertex\n") 
                fin.write("microBC.setPeriodicBCOptions(method, degree,bool(addvertex))\n")
                # Deformation gradient
                fin.write("microBC.setDeformationGradient(1.0,0.02,0.02,0.0,0.97,0,0,0,1.02)\n")

                fin.write("mysolver.addMicroBC(microBC)\n")

                if Do_3D:
                    # If 3D, cannot clamp faces because it is redundant with Periodic Boundary Conditions
                    print()
                else:
                    #if Do_3D:
                        #fin.write("mysolver.displacementBC(\"Face\",1,0,0.)\n")
                        #fin.write("mysolver.displacementBC(\"Face\",1,1,0.)\n")
                    fin.write("mysolver.displacementBC(\"Face\",1,2,0.)\n")
                    
                    #if Do_3D:
                        #fin.write("mysolver.displacementBC(\"Face\",2,0,0.)\n")
                        #fin.write("mysolver.displacementBC(\"Face\",2,1,0.)\n")
                    fin.write("mysolver.displacementBC(\"Face\",2,2,0.)\n")

                fin.write("print(\"> Boundary conditions created successfully.\")\n")

                fin.write("\n")

                """
                NOELS
                """
                if False:
                    # Didn't work
                    fin.write(
                        "#stress averaging flag and averaging method 0- VOLUME, 1- SURFACE\n"
                        "mysolver.stressAveragingFlag(True) # set stress averaging ON- 0 , OFF-1\n"
                        "mysolver.setStressAveragingMethod(0) # 0 -volume 1- surface\n"
                        "#tangent averaging flag\n"
                        "mysolver.tangentAveragingFlag(True) # set tangent averaging ON -0, OFF -1\n"
                        "mysolver.setTangentAveragingMethod(2,1e-6) # 0- perturbation 1- condensation\n"
                    )
                elif False:
                    fin.write(
                        "#stress averaging flag and averaging method 0- VOLUME, 1- SURFACE\n"
                        "mysolver.stressAveragingFlag(bool(1)) # set stress averaging ON- 0 , OFF-1\n"
                        "mysolver.setStressAveragingMethod(0) # 0 -volume 1- surface\n"
                        "#tangent averaging flag\n"
                        "mysolver.tangentAveragingFlag(bool(1)) # set tangent averaging ON -0, OFF -1\n"
                        "mysolver.setTangentAveragingMethod(2,1e-6) # 0- perturbation 1- condensation\n"
                        "mysolver.setExtractElasticTangentOperator(True)\n"
                    )
                else:
                    #stress averaging flag and averaging method 0- VOLUME, 1- SURFACE
                    fin.write("mysolver.stressAveragingFlag(bool(1)) # set stress averaging ON- 0 , OFF-1\n")
                    fin.write("mysolver.setStressAveragingMethod(0) # 0 -volume 1- surface\n")
                    #tangent averaging flag
                    fin.write("mysolver.tangentAveragingFlag(bool(1)) # set tangent averaging ON -0, OFF -1\n")
                    fin.write("mysolver.setTangentAveragingMethod(2,1e-6) # 0- perturbation 1- condensation\n")

                #mysolver.setExtractPerturbationToFileFlag(0)	

                # build view
                fin.write("mysolver.internalPointBuildView(\"Green-Lagrange_xx\",IPField.STRAIN_XX, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"Green-Lagrange_yy\",IPField.STRAIN_YY, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"Green-Lagrange_zz\",IPField.STRAIN_ZZ, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"Green-Lagrange_xy\",IPField.STRAIN_XY, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"Green-Lagrange_yz\",IPField.STRAIN_YZ, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"Green-Lagrange_xz\",IPField.STRAIN_XZ, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"sig_xx\",IPField.SIG_XX, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"sig_yy\",IPField.SIG_YY, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"sig_zz\",IPField.SIG_ZZ, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"sig_xy\",IPField.SIG_XY, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"sig_yz\",IPField.SIG_YZ, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"sig_xz\",IPField.SIG_XZ, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"sig_VM\",IPField.SVM, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"Green-Lagrange equivalent strain\",IPField.GL_EQUIVALENT_STRAIN, 1, 1);\n")
                fin.write("mysolver.internalPointBuildView(\"Equivalent plastic strain\",IPField.PLASTICSTRAIN, 1, 1);\n")
                # solve
                fin.write("print(\"> Launching solver.\")\n")
                fin.write("mysolver.solve()\n")

                fin.write("\nif Do_3D:\n")
                fin.write("\tprint(\"> In 3D\")\n")
                fin.write("else:\n")
                fin.write("\tprint(\"> In 2D\")\n")


    def assign_material_properties(self,E_1,E_2,nu_1,nu_2,rho_1,rho_2):
        """
        Assign material properties of both materials.
        Parameters
        ----------
        E_1 : float
            Young modulus of the first material.
        E_2 : float
            Young modulus of the second material.
        nu_1 : float
            Poisson ratio of the first material.
        nu_2 : float
            Poisson ratio of the second material.
        rho_1 : float
            Density of the first material.
        rho_2 : float
            Density of the second material.
        """
        assert E_1 > 0.0 and E_2 > 0.0 and rho_1 > 0.0 and rho_2 > 0.0
        # Young modulus
        self.E_1 = E_1
        self.E_2 = E_2
        # Poisson ratio
        self.nu_1 = nu_1
        self.nu_2 = nu_2
        # Bulk mass
        self.rho_1 = rho_1
        self.rho_2 = rho_2
        # Bulk modulus
        self.K_1 = E_1/3.0 /(1.0-2.0*nu_1)
        self.K_2 = E_2/3.0 /(1.0-2.0*nu_2)
        # Shear modulus
        self.mu_1 = E_1/2.0 / (1.0+nu_1)
        self.mu_2 = E_2/2.0 / (1.0+nu_2)
