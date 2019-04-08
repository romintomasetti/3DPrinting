import os

from datetime import datetime

import time

import numpy

import shutil

from shutil import copyfile

class SolveHomogenization:
    """
    Solve 2D bi-material homogenization problem.
    """
    def __init__(self,name) -> None:
        """
        Initialization.
        Parameters
        ----------
        name : str
            Name of the solver.
        """
        self.name = name

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
            print("\t> Analyzing ",python_file," stored in ",folder)
            # Read the stiffness tensor
            with open(os.path.join(folder,"E_0_GP_0_tangent.csv"),"r") as fin:
                for line in fin:
                    if "Time" in line:
                        who_is_who = line.split(";")
                    if line[0] == "0":
                        tmp = line.split(";")
                        break
            StiffnessFourthOrderTensor = []
            for c,el in enumerate(tmp):
                try:
                    if c == 0:
                        continue
                    StiffnessFourthOrderTensor.append(float(el))
                except:
                    pass
            if len(StiffnessFourthOrderTensor) != 81:
                raise Exception("Wrong size.")
            
            """
            Isotropic case
            """
            # C_66 (Voigt) -> C_1212 -> C_0101
            index = who_is_who.index("Comp.0101")-1
            mu = StiffnessFourthOrderTensor[index]
            index = who_is_who.index("Comp.1111")-1
            C_11 = StiffnessFourthOrderTensor[index]
            K = C_11 - 4.0*mu/3.0
            nu = (0.5-mu/(3*K))/(1+mu/(3*K))
            E = 2*mu*(1+nu)
            print("\t> E (isotropic): ",E)
            print("\t> nu(isotropic): ",nu)

            """
            Orthotropic case
            """
            # Build the stiffness matrix of orthotropic materials
            StiffMatOrtho = numpy.zeros((6,6))

            # Index [0,0] : C_1111 -> C_11 (Voigt) -> Comp.0000
            index = who_is_who.index("Comp.0000")-1
            StiffMatOrtho[0,0] = StiffnessFourthOrderTensor[index]

            # Index [1,1] : C_2222 -> C_22 (Voigt) -> Comp.1111
            index = who_is_who.index("Comp.1111")-1
            StiffMatOrtho[1,1] = StiffnessFourthOrderTensor[index]

            # Index [2,2] : C_3333 -> C_33 (Voigt) -> Comp.2222
            index = who_is_who.index("Comp.2222")-1
            StiffMatOrtho[2,2] = StiffnessFourthOrderTensor[index]

            # Index [3,3] : C_2323 -> C_44 (Voigt) -> Comp.1212
            index = who_is_who.index("Comp.1212")-1
            StiffMatOrtho[3,3] = StiffnessFourthOrderTensor[index]

            # Index [4,4] : C_3131 -> C_55 (Voigt) -> Comp.2020
            index = who_is_who.index("Comp.2020")-1
            StiffMatOrtho[4,4] = StiffnessFourthOrderTensor[index]

            # Index [5,5] : C_1212 -> C_66 (Voigt) -> Comp.0101
            index = who_is_who.index("Comp.0101")-1
            StiffMatOrtho[5,5] = StiffnessFourthOrderTensor[index]

            # Index [0,1] and [1,0] : C_1122 -> C_12 (Voigt) -> Comp.0011
            index = who_is_who.index("Comp.0011")-1
            StiffMatOrtho[0,1] = StiffnessFourthOrderTensor[index]
            StiffMatOrtho[1,0] = StiffnessFourthOrderTensor[index]
            
            # Index [0,2] and [2,0] : C_1133 -> C_13 (Voigt) -> Comp.0022
            index = who_is_who.index("Comp.0022")-1
            StiffMatOrtho[2,0] = StiffnessFourthOrderTensor[index]
            StiffMatOrtho[0,2] = StiffnessFourthOrderTensor[index]
            
            # Index [1,2] and [2,1] : C_2233 -> C_23 (Voigt) -> Comp.1122
            index = who_is_who.index("Comp.1122")-1
            StiffMatOrtho[2,1] = StiffnessFourthOrderTensor[index]
            StiffMatOrtho[1,2] = StiffnessFourthOrderTensor[index]
            

            StiffMatOrtho_2D = numpy.zeros((3,3))
            StiffMatOrtho_2D[0,0] = StiffMatOrtho[0,0]
            StiffMatOrtho_2D[1,1] = StiffMatOrtho[1,1]
            StiffMatOrtho_2D[0,1] = StiffMatOrtho[0,1]
            StiffMatOrtho_2D[1,0] = StiffMatOrtho[1,0]
            StiffMatOrtho_2D[2,2] = StiffMatOrtho[5,5]

            # Print
            numpy.set_printoptions(formatter={'float': lambda x: format(x, '6.3e')})
            print(StiffMatOrtho)

            try:
                ComplMatOrtho_2D = numpy.linalg.inv(StiffMatOrtho_2D)
                print("> 2D compliance matrix:")
                print(ComplMatOrtho_2D)
                print("\t> E_1 (orthotropic) : ",1.0/ComplMatOrtho_2D[0,0])
                print("\t> E_2 (orthotropic) : ",1.0/ComplMatOrtho_2D[1,1])
                print("\t> nu12(orthotropic) : ",-ComplMatOrtho_2D[0,1]/ComplMatOrtho_2D[0,0])
                print("\t> nu21(orthotropic) : ",-ComplMatOrtho_2D[1,0]/ComplMatOrtho_2D[1,1])
                ComplMatOrtho = numpy.linalg.inv(StiffMatOrtho)
                print("> 3D compliance matrix:")
                print(ComplMatOrtho)
                print("\t> E_1 (orthotropic) : ",1.0/ComplMatOrtho[0,0])
                print("\t> E_2 (orthotropic) : ",1.0/ComplMatOrtho[1,1])
                print("\t> nu12(orthotropic) : ",-ComplMatOrtho[0,1]/ComplMatOrtho[0,0])
                print("\t> nu21(orthotropic) : ",-ComplMatOrtho[1,0]/ComplMatOrtho[1,1])
            except Exception as e:
                print(e)
                pass


    def Execute(self) -> None:
        """
        Execute Python 2 files stored in self.ToExecute list.
        """
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
            if not os.path.exists(new_folder +"_CM3"):
                os.makedirs(
                    new_folder + "_CM3"
                )
            else:
                shutil.rmtree(new_folder + "_CM3")
                os.makedirs(
                    new_folder + "_CM3"
                )
            # Copy the Python file in the new folder
            copyfile(
                python_file,
                os.path.join(new_folder + "_CM3",os.path.basename(python_file))
            )
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
            os.system(
                "python %s"%os.path.basename(python_file) + 
                " > %s.out"%os.path.splitext(os.path.basename(python_file))[0]
            )
            print("\t> Done in ",time.time()-start," seconds.")
            # Roll back to old working directory
            os.chdir(old_dir)

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
                if not Do_3D:
                    fin.write("if not os.path.exists(mesh_file + \".msh\"):\n"
                        "\tif os.path.exists(mesh_file + \".geo\"):\n"
                            "\t\tos.system(\"gmsh -2 \" + mesh_file + \".geo\")\n"
                        "\telse:\n"
                            "\t\traise Exception(mesh_file + \" doesn't exist.\")\n"
                        "mesh_file = mesh_file + \".msh\"\n"
                    )
                else:
                    fin.write("if not os.path.exists(mesh_file + \".msh\"):\n"
                        "\tif os.path.exists(mesh_file + \".geo\"):\n"
                            "\t\tos.system(\"gmsh -3 \" + mesh_file + \".geo\")\n"
                        "\telse:\n"
                            "\t\traise Exception(mesh_file + \" doesn't exist.\")\n"
                        "mesh_file = mesh_file + \".msh\"\n"
                    )
                # Domain
                fin.write("\n\"\"\"\nCreation of domain\n\"\"\"\n")
                # Domain of material 1
                if not Do_3D:
                    fin.write("domain_1 = dG3DDomain(3,1,0,1,0,2)\n")
                    fin.write("domain_1.setPlaneStressState(True)\n")
                else:
                    fin.write("domain_1 = dG3DDomain(3,2221,0,1,0,3)\n")
                # Domain of material 2
                if not Do_3D:
                    fin.write("domain_2 = dG3DDomain(3,2,0,2,0,2)\n")
                    fin.write("domain_2.setPlaneStressState(True)\n")
                else:
                    fin.write("domain_2 = dG3DDomain(3,2222,0,2,0,3)\n")
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
                fin.write("ftime =1.\n")
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
                if not Do_3D:
                    fin.write("microBC = nonLinearPeriodicBC(3,2)\n")
                else:
                    fin.write("microBC = nonLinearPeriodicBC(3,3)\n")
                fin.write("microBC.setOrder(1)\n")
                # Periodic boundary conditions on 4 lines of tag 1, 2, 3, 4
                if not Do_3D:
                    fin.write("microBC.setBCPhysical(1,4,3,2)\n")
                else:
                    fin.write("microBC.setBCPhysical(88880,88883,88884,88881,88882,88885)\n")

                fin.write("method =0	# Periodic mesh = 0, Langrange interpolation = 1, Cubic spline interpolation =2,  FE linear= 3, FE Quad = 4\n")
                fin.write("degree = 2	# Order used for polynomial interpolation\n")
                fin.write("addvertex = 0 # Polynomial interpolation by mesh vertex = 0, Polynomial interpolation by virtual vertex\n") 
                fin.write("microBC.setPeriodicBCOptions(method, degree,bool(addvertex))\n")
                # Deformation gradient
                fin.write("microBC.setDeformationGradient(1.0,0.02,0.02,0.0,0.97,0,0,0,1.02)\n")

                fin.write("mysolver.addMicroBC(microBC)\n")

                if Do_3D:
                    fin.write("mysolver.displacementBC(\"Face\",1,0,0.)\n")
                    fin.write("mysolver.displacementBC(\"Face\",1,1,0.)\n")
                fin.write("mysolver.displacementBC(\"Face\",1,2,0.)\n")
                
                if Do_3D:
                    fin.write("mysolver.displacementBC(\"Face\",2,0,0.)\n")
                    fin.write("mysolver.displacementBC(\"Face\",2,1,0.)\n")
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
                    fin.write("mysolver.setStressAveragingMethod(1) # 0 -volume 1- surface\n")
                    #tangent averaging flag
                    fin.write("mysolver.tangentAveragingFlag(bool(1)) # set tangent averaging ON -0, OFF -1\n")
                    fin.write("mysolver.setTangentAveragingMethod(1,1e-6) # 0- perturbation 1- condensation\n")

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