import numpy

import time

import scipy

import os

from matplotlib import pyplot as plt

from numba.decorators import njit
from numba import float64

class KarhunenLoeve:
    """
    Karhunen-Loeve
    """
    def __init__(self,X,eig_vectors,eig_values) -> None:
        """
        Initialization.
        Parameters
        ----------
        X : numpy.array
            Points of the domain.
        eig_vectors : numpy.array
            Eigen vectors in matrix form.
        eig_values : numpy.array
            Eigen values in vector form. 
        """
        assert isinstance(X,numpy.ndarray)
        assert X.ndim == 2
        self.points = X
        
        assert isinstance(eig_vectors,numpy.ndarray)
        assert eig_vectors.ndim == 2
        self.eig_vectors = eig_vectors
        
        assert isinstance(eig_values,numpy.ndarray)
        assert eig_values.ndim == 1
        self.eig_values = eig_values
        
        assert X.shape[0] == eig_vectors.shape[0]
        
        assert eig_vectors.shape[1] == eig_values.shape[0]

        assert numpy.all(eig_values >= 0.)
        
    def SampleKL(self,num_xis,filename,size=1) -> numpy.ndarray:
        """
        Create a sample of the Karhunen-Loeve field represented by the eigen values and eigen vectors.
        Parameters
        ----------
        num_xis : int
            Number of xis in the K.-L. decomposition.
        filename : str
            File in which xis are saved.
        """
        # Generate xis:
        xi = numpy.random.normal(0, 1, size=(size,num_xis))
        with open(filename,"wb+") as fin:
            xi.tofile(fin)
        # Generate the random field
        return numpy.dot(xi * numpy.sqrt(self.eig_values), self.eig_vectors.T)

class MaterialField:
    """
    Material field.
    """
    def __init__(
        self,name,folder,threshold,eps_11,eps_22,
        Alphas,Lengths,Nodes,Samples,
        consider_as_zero,AngleType,
        isOrthotropicTest = False):
        """
        Initialization.
        Parameters
        ----------
        name : str
            Name of the material field.
        folder : str
            Directory in which material fields will be stored.
        threshold : float
            Threshold between the materials.
        eps_11 : List(float)
            First characteristic length.
        eps_22 : List(float)
            Second characteristic length.
        Alphas : List(float)
            Tilting angle.
        Lengths : List(float)
            Lengths of the domain along each direction.
        Nodes : List(int)
            Number of nodes along each direction.
        Samples : List(int)
            Number of samples of the K.-L. to create for each configuration.
        consider_as_zero : float
            Eigen values smaller than consider_as_zero are considered as being zero.
        AngleType : str
            Type of angle (radian,degree).
        isOrthotropicTest : bool
            If true, a purely orthotropic domain is created.
        """
        
        self.name = name
        
        self.folder = folder

        folder = os.path.join(self.folder,self.name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        assert isinstance(threshold, float)
        self.threshold = threshold

        assert len(eps_11) == len(eps_22)
        assert len(eps_11) == len(Alphas)

        assert all(isinstance(x, float) for x in eps_11)
        self.eps_11 = eps_11

        assert all(isinstance(x, float) for x in eps_22)
        self.eps_22 = eps_22

        assert all(isinstance(x, float) for x in Alphas)
        self.Alphas = Alphas

        assert all(isinstance(x, float) for x in Lengths)
        assert len(Lengths) == 2
        self.Lengths = Lengths

        assert all(isinstance(x, int) for x in Nodes)
        assert len(Nodes) == 2
        self.Nodes = Nodes

        assert all(isinstance(x,int) for x in Samples)
        assert len(Samples) == len(eps_11)
        self.Samples = Samples

        self.consider_as_zero = consider_as_zero

        if AngleType == "degree":
            for a in range(len(self.Alphas)):
                self.Alphas[a] = self.Alphas[a] / 180. * numpy.pi

        self.isOrthotropicTest = isOrthotropicTest

    def ToGMSH(self,Do_3D) -> list:
        """
        Transform realizations of the random material field in GMSH file.
        """
        print("\t" + 50*"-")
        files_geo = []
        domain_size = (self.Nodes[0],self.Nodes[1])
        for config in range(len(self.eps_11)):
            print("\t> Configuration ",config," to GMSH.")
            for sample in range(self.Samples[config]):
                print("\t> Realization sample ",sample," to GMSH.")
                filename = os.path.join(
                    self.folder,
                    self.name,
                    "Realization_" + str(config) + "_" + str(sample) + "_binary"
                )
                mat_distrib = \
                    numpy.fromfile(
                        filename,
                        dtype=numpy.int64
                    ).reshape(domain_size)
                mat_values = numpy.unique(mat_distrib)
                filename_geo= os.path.join(
                    self.folder,
                    self.name,
                    "Realization_" + str(config) + "_" + str(sample) + "_binary.geo"
                )
                files_geo.append(filename_geo)
                start = time.time()
                # Write the .geo file
                with open(filename_geo,"w+") as geo:
                    # Use quads
                    geo.write("Mesh.Algorithm = 8;\n")
                    # Domain length
                    geo.write("_Lx = %.5f;\n"%self.Lengths[0])
                    geo.write("_Ly = %.5f;\n"%self.Lengths[1])
                    # Domain division
                    geo.write("_nx = %d;\n"%self.Nodes[0])
                    geo.write("_ny = %d;\n"%self.Nodes[1])
                    # Add all points
                    counter_points = 1
                    dx = self.Lengths[0]/self.Nodes[0]
                    dy = self.Lengths[1]/self.Nodes[1]
                    default_mesh_elm_size = dx
                    
                    for row in range(self.Nodes[1]):
                        for col in range(self.Nodes[0]):
                            # For each point, we create a square, whose center is the point
                            geo.write(
                                "Point(%d) = {%.5f,%.5f,%.2f,%.5f};\n"\
                                %(
                                    counter_points,
                                    col*dx,
                                    row*dy,
                                    0.0,
                                    default_mesh_elm_size
                                )
                            )
                            counter_points += 1
                        geo.write(
                                "Point(%d) = {%.5f,%.5f,%.2f,%.5f};\n"\
                                %(
                                    counter_points,
                                    self.Nodes[0]*dx,
                                    row*dy,
                                    0.0,
                                    default_mesh_elm_size
                                )
                            )
                        counter_points += 1
                    for col in range(self.Nodes[0]+1):
                        geo.write(
                                "Point(%d) = {%.5f,%.5f,%.2f,%.5f};\n"\
                                %(
                                    counter_points,
                                    col*dx,
                                    self.Nodes[1]*dy,
                                    0.0,
                                    default_mesh_elm_size
                                )
                            )
                        counter_points += 1
                    
                    # Add horizontal lines:       
                    counter_lines = 1
                    for row in range(self.Nodes[1]+1):
                        for col in range(self.Nodes[0]):
                            pt_1 = row*(self.Nodes[0]+1) + col + 1
                            pt_2 = row*(self.Nodes[0]+1) + col + 1 + 1
                            
                            geo.write("Line(%d) = {%d,%d};\n"\
                                %(
                                    counter_lines,
                                    pt_1,
                                    pt_2
                                )
                            )
                            counter_lines += 1
                        
                    # Add vertical lines:      
                    for row in range(self.Nodes[1]):
                        for col in range(self.Nodes[0]+1):
                        
                            pt_1 = row*(self.Nodes[0]+1) + col + 1
                            pt_2 = row*(self.Nodes[0]+1) + col + self.Nodes[0]+1 + 1

                            geo.write("Line(%d) = {%d,%d};\n"\
                                %(
                                    counter_lines,
                                    pt_1,
                                    pt_2
                                )
                            )
                            counter_lines += 1
                    
                    geo.write("Transfinite Line {%d"%(1))
                    for i in range(2,counter_lines-1):
                        geo.write(",%d"%i)
                    geo.write(",%d} = 2;\n"%(counter_lines-1))
                    
                    # Add curve loops, plane surfaces and transfinite surfaces
                    number_horizontal_lines = (self.Nodes[0])*(self.Nodes[1]+1)
                    curve_loop_counter      = 1
                    for row in range(self.Nodes[1]):
                        for col in range(self.Nodes[0]):
                            lines = \
                                [
                                    row   *(self.Nodes[0]) + col + 1,
                                    number_horizontal_lines + (col+1) + row*(self.Nodes[0]+1) + 1,
                                    (row+1)*(self.Nodes[0]) + col + 1,
                                    number_horizontal_lines + (col  ) + row*(self.Nodes[0]+1) + 1                   
                                ]
                            
                            geo.write("Curve Loop(%d) = {%d,%d,%d,%d};\n"\
                                %(
                                    curve_loop_counter,
                                    lines[0],
                                    lines[1],
                                    -lines[2],
                                    -lines[3]
                                )
                            )
                            geo.write("Plane Surface(%d) = {%d};\n"%(curve_loop_counter,curve_loop_counter))
                            geo.write("Transfinite Surface {%d};\n"%curve_loop_counter)
                            geo.write("Recombine Surface {%d};\n"%curve_loop_counter)
                            curve_loop_counter += 1

                    # Add physical curves (for boundary conditions)
                    lines = [
                        [i                       for i in range(1,self.Nodes[0]+1)  ], # y = 0
                        [number_horizontal_lines + i*(self.Nodes[0]+1) for i in range(1,self.Nodes[1]+1)], # x = Lx
                        [(self.Nodes[1])*(self.Nodes[0]) + i     for i in range(1,self.Nodes[0]+1)  ], # y = Ly
                        [(self.Nodes[1]+1)*(self.Nodes[0]) + i*(self.Nodes[0]+1) + 1 for i in range(0,self.Nodes[1])  ], # x = 0
                    ]

                    counter_physical_curve = 1
                    for line in lines:
                        geo.write("Physical Curve (%d) = {%d"%(counter_physical_curve,line[0]))
                        for i in range(1,len(line)):
                            geo.write(",%d"%line[i])
                        geo.write("};\n")
                        counter_physical_curve += 1
                    
                    # Add physical surfaces
                    surface_mat_1 = []
                    surface_mat_2 = []

                    counter_surfaces = 1
                    for row in reversed(range(self.Nodes[1])):
                        for col in range(self.Nodes[0]):
                            if mat_distrib[row,col] == mat_values[0]:
                                surface_mat_1.append(counter_surfaces)
                            else:
                                surface_mat_2.append(counter_surfaces)
                            counter_surfaces += 1

                    geo.write("Physical Surface (1) = {%d"%surface_mat_1[0])
                    for i in range(1,len(surface_mat_1)):
                        geo.write(",%d"%surface_mat_1[i])
                    geo.write("};\n")

                    geo.write("Physical Surface (2) = {%d"%surface_mat_2[0])
                    for i in range(1,len(surface_mat_2)):
                        geo.write(",%d"%surface_mat_2[i])
                    geo.write("};\n")

                    if Do_3D:
                        # Create array with all material 1 surfaces
                        geo.write("MaterialSurfaces_1 = {%d"%surface_mat_1[0])
                        for i in range(1,len(surface_mat_1)):
                            geo.write(",%d"%surface_mat_1[i])
                        geo.write("};\n")
                        # Create array with all material 2 surfaces
                        geo.write("MaterialSurfaces_2 = {%d"%surface_mat_2[0])
                        for i in range(1,len(surface_mat_2)):
                            geo.write(",%d"%surface_mat_2[i])
                        geo.write("};\n")

                        # Extrude thickness
                        ExtrudeThickness = 10

                        # Extrude all surfaces with material 1:
                        geo.write("VolumesMaterial_1[] = Extrude {0.,0.,%.5f}{\n"%ExtrudeThickness)
                        geo.write("\tSurface{MaterialSurfaces_1[{0:#MaterialSurfaces_1[]-1}]};\n")
                        geo.write("\tLayers{4};\n")
                        geo.write("\tRecombine;\n")
                        geo.write("};\n")
                        # Physical volume with volumes with material 1:
                        geo.write(
                            "Physical Volume(2221) = {VolumesMaterial_1[{0:#VolumesMaterial_1[]-1}]};\n"
                        )
                        # Extrude all surfaces with material 2:
                        geo.write("VolumesMaterial_2[] = Extrude {0.,0.,%.5f}{\n"%ExtrudeThickness)
                        geo.write("\tSurface{MaterialSurfaces_2[{0:#MaterialSurfaces_2[]-1}]};\n")
                        geo.write("\tLayers{1};\n")
                        geo.write("\tRecombine;\n")
                        geo.write("};\n")
                        # Physical volume with volumes with material 2:
                        geo.write(
                            "Physical Volume(2222) = {VolumesMaterial_2[{0:#VolumesMaterial_2[]-1}]};\n"
                        )

                        # Tolerance
                        EPS = 1.0e-8;

                        # Get surfaces with x = 0
                        geo.write(
                            "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                            (
                                -EPS,-EPS,-EPS,+EPS,self.Lengths[1]+EPS,ExtrudeThickness+EPS
                            )
                        )

                        geo.write(
                            "Physical Surface (88880) = {s[{0:#s[]-1}]};\n"
                        )

                        # Get surfaces with x = self.Lengths[0]
                        geo.write(
                            "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                            (
                                -EPS+self.Lengths[0],-EPS,-EPS,+EPS+self.Lengths[0],self.Lengths[1]+EPS,ExtrudeThickness+EPS
                            )
                        )

                        geo.write(
                            "Physical Surface (88881) = {s[{0:#s[]-1}]};\n"
                        )

                        # Get surfaces with y = 0
                        geo.write(
                            "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                            (
                                -EPS,-EPS,-EPS,+EPS+self.Lengths[0],+EPS,ExtrudeThickness+EPS
                            )
                        )

                        geo.write(
                            "Physical Surface (88882) = {s[{0:#s[]-1}]};\n"
                        )
                        
                        # Get surfaces with y = self.Lengths[1]
                        geo.write(
                            "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                            (
                                -EPS,-EPS+self.Lengths[1],-EPS,+EPS+self.Lengths[0],+EPS+self.Lengths[1],ExtrudeThickness+EPS
                            )
                        )

                        geo.write(
                            "Physical Surface (88883) = {s[{0:#s[]-1}]};\n"
                        )

                        # Get surfaces with z = 0
                        geo.write(
                            "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                            (
                                -EPS,
                                -EPS,
                                -EPS,
                                +EPS+self.Lengths[0],
                                +EPS+self.Lengths[1],
                                +EPS
                            )
                        )

                        geo.write(
                            "Physical Surface (88884) = {s[{0:#s[]-1}]};\n"
                        )

                        # Get surfaces with z = ExtrudeThickness
                        geo.write(
                            "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                            (
                                -EPS,
                                -EPS,
                                -EPS+ExtrudeThickness,
                                +EPS+self.Lengths[0],
                                +EPS+self.Lengths[1],
                                +EPS+ExtrudeThickness
                            )
                        )

                        geo.write(
                            "Physical Surface (88885) = {s[{0:#s[]-1}]};\n"
                        )

                print(
                    "\t> ",
                    filename_geo,
                    " successfully created (size ",
                    os.path.getsize(filename_geo)/1e6,
                    " MBytes) in ",
                    time.time()-start,
                    "seconds.")

        return files_geo

    def Create(self) -> None:
        """
        Create the material fields.
        """
        print("\t" + 50*"-")
        # The points where the random field is evaluated
        x = numpy.linspace(0, self.Lengths[0], self.Nodes[0])
        y = numpy.linspace(0, self.Lengths[1], self.Nodes[1])
        XX, YY = numpy.meshgrid(x, y)
        X = numpy.hstack([XX.flatten()[:, None], YY.flatten()[:, None]])

        for config in range(len(self.eps_11)):
            print("\t> Configuration ",config)
            # Timing the completion of the current configuration
            start = time.time()
            # Parameters of the current configuration
            eps_11 = self.eps_11[config]
            eps_22 = self.eps_22[config]
            alpha  = self.Alphas[config]
            print("\t> eps_11 : %f | eps_22 : %f | alpha : %f"%(eps_11,eps_22,alpha))
            # Create the covariance matrix kernel
            H = numpy.array([[eps_11,0.],[0.,eps_22]])
            # Rotation matrix 2D:
            R = numpy.array([
                [numpy.cos(alpha),-numpy.sin(alpha)],
                [numpy.sin(alpha), numpy.cos(alpha)]
                ])
            # Rotate the covariance matrix:
            H = numpy.matmul(numpy.matmul(R,H),numpy.transpose(R))
            # Inverse the H matrix:
            H = numpy.linalg.inv(H)
            # Check that H is semi-positive definite:
            try:
                numpy.linalg.cholesky(H)
            except Exception as e:
                print(e)
                raise Exception("H is not semi-def pos !")
            # Build the covariance matrix
            CovMat = self.ComputeCovarianceMatrix(X,H)
            print("\t> Covariance matrix computed in ",time.time()-start," seconds.")
            # Plot the covariance matrix and save it
            fig = plt.figure()
            plt.imshow(CovMat)
            plt.savefig(
                os.path.join(self.folder,self.name,"CovarianceMatrix_" + str(config) + ".eps"),
                format = 'eps'
            )
            plt.close(fig)
            # Compute the eigenvalues and eigenvectors of the covariance matrix
            start_eig = time.time()
            MAX_LARGEST_EIGVALS = int(0.33 * CovMat.shape[0])
            eig_values, eig_vectors = \
                scipy.linalg.eigh(
                    CovMat,
                    eigvals = (CovMat.shape[0]-MAX_LARGEST_EIGVALS,CovMat.shape[0]-1)
                )
            print("\t> Found ",len(eig_values)," eig. values in ",time.time()-start_eig," seconds.")
            # Sort the eigenvalues from largest to smallest
            idx = numpy.argsort(eig_values)[::-1]
            eig_values = eig_values[idx]
            eig_vectors = eig_vectors[:, idx]
            # Plot eigen values and save it
            fig = plt.figure()
            plt.plot(eig_values,"bo-")
            plt.savefig(
                os.path.join(
                    self.folder,self.name,
                    "EigenValues_" + str(config) + ".eps"
                ),
                format = "eps"
            )
            # Plot the first eigenvectors
            eig_vec_to_plot = 10
            print("\t> Saving first %d eigen vectors."%eig_vec_to_plot)
            for i in range(numpy.min(numpy.asarray([eig_vec_to_plot,len(eig_values)]))):
                fig = plt.figure()
                c = plt.contourf(
                    XX, YY,
                    eig_vectors[:, i].reshape((self.Nodes[0], self.Nodes[1])),
                    levels = 50
                )
                plt.colorbar(c)
                plt.xlabel('x', fontsize=16)
                plt.ylabel('y', fontsize=16)
                plt.title('Eigenvector %d' % (i), fontsize=16)
                plt.savefig(
                    os.path.join(
                        self.folder,self.name,
                        "EigenVector_" + str(config) + "_" + str(i) + ".eps"
                    ),
                    format = "eps"
                )
                plt.close(fig)
            # Remove eigen values smaller than a threshold (machine errors)
            eig_values = numpy.where(numpy.abs(eig_values) < self.consider_as_zero,0.,eig_values)
            if numpy.any(eig_values < 0.0):
                raise Exception("Negative eigen value.")
            # Build the Karhunen-Loeve object
            kl_obj = KarhunenLoeve(X, eig_vectors,eig_values)
            # Create the required number of samples
            for sample in range(self.Samples[config]):
                print("\t> Sampling K.-L. ",sample)
                # Create a sample of the K.-L. decomposition
                filename = os.path.join(
                    self.folder,
                    self.name,
                    "Realization_" + str(config) + "_" + str(sample) + "_xis"
                )
                realization = \
                    kl_obj.SampleKL(
                        MAX_LARGEST_EIGVALS,
                        filename
                    ).reshape(
                        (self.Nodes[0],self.Nodes[1])
                    )
                realization_binary = \
                    numpy.where(realization > self.threshold, -1, +1)
                # If we must perform an orthotropic test, ignore the usual procedure
                if self.isOrthotropicTest is True:
                    # Alternatively short the columns
                    realization = numpy.zeros((self.Nodes[0],self.Nodes[1]))
                    prop_1 = 1
                    prop_2 = 3
                    col = 0
                    while col < self.Nodes[0]:
                        for col_ in range(col,col+prop_1):
                            col = col_
                            if col < self.Nodes[0]:
                                realization[col,:] = +1
                        col += 1
                        for col_ in range(col,col+prop_2):
                            col = col_
                            if col < self.Nodes[0]:
                                realization[col,:] = -1

                    realization_binary = realization
                # Save realization
                filename = os.path.join(
                    self.folder,
                    self.name,
                    "Realization_" + str(config) + "_" + str(sample)
                )
                with open(filename,'wb+') as fin:
                    realization.tofile(fin)
                # Plot realization and save it
                fig = plt.figure()
                c = plt.imshow(
                    realization,
                    extent=[0, self.Lengths[0], 0, self.Lengths[1]]
                )
                plt.colorbar(c)
                plt.xlabel('x', fontsize=16)
                plt.ylabel('y', fontsize=16)
                plt.title('Sample %d' % sample, fontsize=16)
                plt.savefig(
                    filename + ".eps",
                    format = 'eps'
                )
                plt.close(fig)
                # Save binary realization
                filename = os.path.join(
                    self.folder,
                    self.name,
                    "Realization_" + str(config) + "_" + str(sample) + "_binary"
                )
                with open(filename,'wb+') as fin:
                    realization_binary.tofile(fin)
                # Plot binary realization and save it
                fig = plt.figure()
                c = plt.imshow(
                    realization_binary,
                    extent=[0, self.Lengths[0], 0, self.Lengths[1]]
                )
                plt.colorbar(c)
                plt.xlabel('x', fontsize=16)
                plt.ylabel('y', fontsize=16)
                plt.title('Sample %d' % sample, fontsize=16)
                plt.savefig(
                    filename + ".eps",
                    format = 'eps'
                )
                plt.close(fig)
            # Save the eigen values
            filename = os.path.join(
                self.folder,
                self.name,
                "EigenValues_" + str(config)
            )
            with open(filename,"wb+") as fin:
                eig_values.tofile(fin)
            # Save the eigen vectors
            filename = os.path.join(
                self.folder,
                self.name,
                "EigenVectors_" + str(config)
            )
            with open(filename,"wb+") as fin:
                eig_vectors.tofile(fin)
            # Save the covariance matrix
            filename = os.path.join(
                self.folder,
                self.name,
                "CovarianceMatrix_" + str(config)
            )
            with open(filename,"wb+") as fin:
                CovMat.tofile(fin)

            print("\t> Configuration ",config," finished in ",time.time()-start)
                

    def ComputeCovarianceMatrix(self,X,H):
        """
        Compute covariance matrix.
        Parameters
        ----------
        X : numpy.ndarray
            Points of the domain.
        H : numpy.ndarray
            Covariance matrix kernel.
        """
        @njit([float64[:,:](float64[:,:],float64[:,:])],parallel = True)
        def _ComputeCovarianceMatrix(X,H):
            assert X.ndim == 2
            C = numpy.zeros((X.shape[0], X.shape[0]))
            dist = numpy.zeros(X.ndim)
            for i in range(0,X.shape[0]):
                for j in range(0,X.shape[0]):
                    dist[0] = X[i,0] - X[j,0]
                    dist[1] = X[i,1] - X[j,1]
                    C[i,j] = numpy.exp(
                        - dist.dot(  H.dot( dist))
                    )
            return C

        return _ComputeCovarianceMatrix(X,H)

