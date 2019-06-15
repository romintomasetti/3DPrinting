import numpy

import time

import scipy

import os

from GmshObjects import *

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import skimage

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
        isOrthotropicTest = False,
        RatioHighestSmallestEig = 100.0
    ):
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
        RatioHighestSmallestEig : float
            Ratio between largest and smallest eigenvalues
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

        self.RatioHighestSmallestEig = RatioHighestSmallestEig

    def ToGMSH(self,Do_3D,num_3d_layers=1) -> list:
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
                    # Domain length
                    geo.write("_Lx = %.5f;\n"%self.Lengths[0])
                    geo.write("_Ly = %.5f;\n"%self.Lengths[1])
                    # Domain division
                    geo.write("_nx = %d;\n"%self.Nodes[0])
                    geo.write("_ny = %d;\n"%self.Nodes[1])

                    # Labeling the material distribution
                    all_labels = skimage.measure.label(mat_distrib,connectivity=1)
                    # Don't want zero labels:
                    if numpy.where(all_labels == 0):
                        all_labels += 1
                    # get unique labels
                    labels = numpy.unique(all_labels)
                    # Match the labels with the materials
                    label_to_mat_value = []
                    for label in labels:
                        itemindex = numpy.where(all_labels == label)
                        label_to_mat_value.append(
                            [label,mat_distrib[itemindex[0][0]][itemindex[1][0]]]
                        )

                    # Create 2D surfaces
                    list_of_surfs = []
                    for c,label in enumerate(labels):
                        list_of_surfs.append(Surface_2D(label,c+1))

                    # Create 2D plane
                    gmsh_plane_2D = Plane_2D(list_of_surfs)
                    

                    # Add all points
                    counter_points = 1
                    dx = self.Lengths[0]/self.Nodes[0]
                    dy = self.Lengths[1]/self.Nodes[1]
                    default_mesh_elm_size = dx
                    default_line_density = 1

                    pos_z = 0.0
                    extrude_z = 5.0

                    for row in range(self.Nodes[1]+1):
                        for col in range(self.Nodes[0]+1):
                            pos_y = dy * row
                            pos_x = dx * col
                            gmsh_plane_2D.add_point(
                                Point(counter_points,pos_x,pos_y,pos_z,default_mesh_elm_size)
                            )
                            counter_points += 1

                    counter_lines = 1

                    # Horizontal lines between label zones
                    for col in range(self.Nodes[0]):
                        current_label = all_labels[col][0]
                        for row in range(self.Nodes[1]):
                            if all_labels[col][row] != current_label:
                                tmp_1 = (col-0) * (self.Nodes[1]+1) + row + 1
                                tmp_2 = (col+1) * (self.Nodes[1]+1) + row + 1
                                gmsh_plane_2D.add_line(
                                    Line(
                                        counter_lines,
                                        gmsh_plane_2D.get_point(tmp_1),
                                        gmsh_plane_2D.get_point(tmp_2),
                                        current_label,
                                        all_labels[col][row],
                                        "H",
                                        default_line_density
                                    )
                                )
                                current_label = all_labels[col][row]
                                counter_lines += 1

                    # Vertical lines between label zones
                    for row in range(self.Nodes[1]):
                        current_label = all_labels[0][row]
                        for col in range(self.Nodes[0]):
                            if all_labels[col][row] != current_label:
                                tmp_1 = (col - 0 ) * (self.Nodes[1]+1) + row    + 1
                                tmp_2 = (col - 0 ) * (self.Nodes[1]+1) + row +1 + 1
                                gmsh_plane_2D.add_line(
                                    Line(
                                        counter_lines,
                                        gmsh_plane_2D.get_point(tmp_1),
                                        gmsh_plane_2D.get_point(tmp_2),
                                        all_labels[col][row],
                                        current_label,
                                        "V",
                                        default_line_density
                                    )
                                )
                                current_label = all_labels[col][row]
                                counter_lines += 1

                    # External contour lines, horizontal
                    for pt in range(self.Nodes[1]):
                        # One side
                        gmsh_plane_2D.add_line(
                            Line(
                                counter_lines,
                                gmsh_plane_2D.get_point(pt+1),
                                gmsh_plane_2D.get_point(pt+1+1),
                                all_labels[0,pt],
                                -1,
                                "V",
                                default_line_density
                            )
                        )
                        counter_lines += 1
                        # The other side
                        tmp = self.Nodes[0] * (self.Nodes[1]+1)
                        gmsh_plane_2D.add_line(
                            Line(
                                counter_lines,
                                gmsh_plane_2D.get_point(tmp + pt + 1),
                                gmsh_plane_2D.get_point(tmp + pt + 1 + 1),
                                -1,
                                all_labels[-1,pt],
                                "V",
                                default_line_density
                            )
                        )
                        counter_lines += 1

                    # External contour lines, vertical
                    for pt in range(self.Nodes[0]):
                        # On one side
                        tmp = self.Nodes[1] + 1
                        gmsh_plane_2D.add_line(
                            Line(
                                counter_lines,
                                gmsh_plane_2D.get_point(tmp * pt + 1),
                                gmsh_plane_2D.get_point(tmp * (pt + 1) + 1),
                                -1,
                                all_labels[pt,0],
                                "H",
                                default_line_density
                            )
                        )
                        counter_lines += 1
                        # On the other side
                        gmsh_plane_2D.add_line(
                            Line(
                                counter_lines,
                                gmsh_plane_2D.get_point(tmp * (pt+1) -1 + 1),
                                gmsh_plane_2D.get_point(tmp * (pt + 2) - 1 + 1),
                                all_labels[pt,-1],
                                -1,
                                "H",
                                default_line_density
                            )
                        )
                        counter_lines += 1


                    gmsh_plane_2D.ToGMSH(geo)

                    # Curve loops
                    curve_loop_counter = 1

                    for label in labels:
                        lines = gmsh_plane_2D.get_lines_label(label)
                        curve_loop_counter = gmsh_plane_2D.get_surfaces_label(label)[0].ID
                        
                        lines_oriented = gmsh_plane_2D.get_lines_oriented(label)

                        string = "Curve Loop(%d) = {"%curve_loop_counter
                        for i in lines_oriented:
                            string += str(i) + ","
                        string = string[:-1]
                        string += "};\n"
                        geo.write(string)

                        geo.write("Plane Surface(%d) = {%d};\n"%(curve_loop_counter,curve_loop_counter))
        
                        geo.write("Recombine Surface {%d};\n"%(curve_loop_counter))
                        
                        curve_loop_counter += 1
                    
                    # Physical curves around the 2D plane
                    eps = 1e-9
                    # For y = 0
                    geo.write(
                        "s[] = Curve In BoundingBox{%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%(\
                            -eps,-eps,-eps,
                            self.Lengths[0] + eps, eps, eps
                        )
                    )
                    geo.write(
                        "Physical Curve(1) = {s[{0:#s[]-1}]};\n"
                    )
                    # For x = Lx
                    geo.write(
                        "s[] = Curve In BoundingBox{%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%(\
                            -eps + self.Lengths[0],-eps                  ,-eps,
                            +eps + self.Lengths[0], eps + self.Lengths[1], eps
                        )
                    )
                    geo.write(
                        "Physical Curve(2) = {s[{0:#s[]-1}]};\n"
                    )
                    # For y = Ly
                    geo.write(
                        "s[] = Curve In BoundingBox{%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%(\
                            -eps                   ,-eps+self.Lengths[1],-eps,
                            self.Lengths[0] + eps  , eps+self.Lengths[1], eps
                        )
                    )
                    geo.write(
                        "Physical Curve(3) = {s[{0:#s[]-1}]};\n"
                    )
                    # For x = 0
                    geo.write(
                        "s[] = Curve In BoundingBox{%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%(\
                            -eps,-eps,-eps,
                            +eps, eps + self.Lengths[1], eps
                        )
                    )
                    geo.write(
                        "Physical Curve(4) = {s[{0:#s[]-1}]};\n"
                    )

                    
                    counter_phys_surf = 1

                    geo.write("LAYERS = %d;\n"%num_3d_layers)
                    
                    for material in mat_values:
                        surfs = []
                        for tmp in label_to_mat_value:
                            if tmp[1] == material:
                                #print("> Looking for surfaces with label ",tmp[0])
                                surfs += [x.ID for x in gmsh_plane_2D.get_surfaces_label(tmp[0])]
                        if not surfs:
                            raise Exception("No surface correcponding to material " + str(material))
                        string = "Physical Surface(%d) = {"%counter_phys_surf
                        for s in surfs:
                            string += str(s) + ","
                        string = string[:-1]
                        string += "};\n"
                        
                        geo.write(string)
                        
                        if not Do_3D:
                            geo.write("If (0)\n")
                        else:
                            geo.write("If (1)\n")
                        # Group surface for current material
                        geo.write("MaterialSurfaces_%d = {"%counter_phys_surf)
                        string = ""
                        for s in surfs:
                            string += str(s) + ","
                        string = string[:-1]
                        string += "};\n"
                        geo.write(string)
                        
                        # Physical volume
                        geo.write("VolumesMaterial_%d[] = Extrude {0.0,0.0,%.3f}{\n"%(counter_phys_surf,extrude_z))
                        geo.write("\tSurface{MaterialSurfaces_%d[{0:#MaterialSurfaces_%d[]-1}]};\n"%(\
                            counter_phys_surf,counter_phys_surf))
                        geo.write("\tLayers{LAYERS};\n\tRecombine;\n};\n")
                        
                        geo.write("Physical Volume(222%d) = {VolumesMaterial_%d[{0:#VolumesMaterial_%d[]-1}]};\n"%(\
                            counter_phys_surf,counter_phys_surf,counter_phys_surf))
                        
                        geo.write("EndIf\n")
                    
                        counter_phys_surf += 1

                    if not Do_3D:
                        geo.write("If (0)\n")
                    else:
                        geo.write("If (1)\n")
                    # Get surfaces with x = 0
                    geo.write(
                        "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                        (
                            -eps,-eps,-eps,
                            +eps,self.Lengths[1]+eps,extrude_z+eps
                        )
                    )

                    geo.write(
                        "Physical Surface (88880) = {s[{0:#s[]-1}]};\n"
                    )

                    # Get surfaces with x = self.Lengths[0]
                    geo.write(
                        "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                        (
                            -eps+self.Lengths[0],-eps,-eps,
                            +eps+self.Lengths[0],self.Lengths[1]+eps,extrude_z+eps
                        )
                    )

                    geo.write(
                        "Physical Surface (88881) = {s[{0:#s[]-1}]};\n"
                    )

                    # Get surfaces with y = 0
                    geo.write(
                        "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                        (
                            -eps,-eps,-eps,
                            +eps+self.Lengths[0],+eps,extrude_z+eps
                        )
                    )

                    geo.write(
                        "Physical Surface (88882) = {s[{0:#s[]-1}]};\n"
                    )
                    
                    # Get surfaces with y = self.Lengths[1]
                    geo.write(
                        "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                        (
                            -eps,-eps+self.Lengths[1],-eps,
                            +eps+self.Lengths[0],+eps+self.Lengths[1],extrude_z+eps
                        )
                    )

                    geo.write(
                        "Physical Surface (88883) = {s[{0:#s[]-1}]};\n"
                    )

                    # Get surfaces with z = 0
                    geo.write(
                        "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                        (
                            -eps,
                            -eps,
                            -eps,
                            +eps+self.Lengths[0],
                            +eps+self.Lengths[1],
                            +eps
                        )
                    )

                    geo.write(
                        "Physical Surface (88884) = {s[{0:#s[]-1}]};\n"
                    )

                    # Get surfaces with z = ExtrudeThickness
                    geo.write(
                        "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                        (
                            -eps,
                            -eps,
                            -eps+extrude_z,
                            +eps+self.Lengths[0],
                            +eps+self.Lengths[1],
                            +eps+extrude_z
                        )
                    )

                    geo.write(
                        "Physical Surface (88885) = {s[{0:#s[]-1}]};\n"
                    )

                    geo.write("EndIf\n")

                print(
                    "\t> ",
                    filename_geo,
                    " successfully created (size ",
                    os.path.getsize(filename_geo)/1e6,
                    " MBytes) in ",
                    time.time()-start,
                    "seconds.")

        return files_geo

    
    def ToGMSH_old(self,Do_3D) -> list:
        """
        Transform realizations of the random material field in GMSH file.
        This version works but is very slow when the number of points gets high.
        See ToGMSH for a better version
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

                        # Number of layers along z
                        nbr_layers = 2

                        # Extrude all surfaces with material 1:
                        geo.write("VolumesMaterial_1[] = Extrude {0.,0.,%.5f}{\n"%ExtrudeThickness)
                        geo.write("\tSurface{MaterialSurfaces_1[{0:#MaterialSurfaces_1[]-1}]};\n")
                        geo.write("\tLayers{%d};\n"%nbr_layers)
                        geo.write("\tRecombine;\n")
                        geo.write("};\n")
                        # Physical volume with volumes with material 1:
                        geo.write(
                            "Physical Volume(2221) = {VolumesMaterial_1[{0:#VolumesMaterial_1[]-1}]};\n"
                        )
                        # Extrude all surfaces with material 2:
                        geo.write("VolumesMaterial_2[] = Extrude {0.,0.,%.5f}{\n"%ExtrudeThickness)
                        geo.write("\tSurface{MaterialSurfaces_2[{0:#MaterialSurfaces_2[]-1}]};\n")
                        geo.write("\tLayers{%d};\n"%nbr_layers)
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
   
    def Create(self,eig_vec_to_plot = 0) -> None:
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
            # We want to compute only the largest. compute by default the 10% largest
            # and then verify how the largest and smallest scale.
            # Example: Largest = 10, smallest = 1 => ratio 10 not enough if 100 is specified
            start_eig = time.time()
            eigenvalues_computed = numpy.array([])
            eigenvectors_computed = None
            RatioHighestSmallestEig = 0.0
            # Compute by slices of 10%
            number_of_eigs_to_compute = int(0.1 * CovMat.shape[0])
            while RatioHighestSmallestEig < self.RatioHighestSmallestEig\
                and CovMat.shape[0]-number_of_eigs_to_compute-len(eigenvalues_computed) >= 0 :
                eig_values, eig_vectors = \
                    scipy.linalg.eigh(
                        CovMat,
                        eigvals = (
                            CovMat.shape[0]-number_of_eigs_to_compute-len(eigenvalues_computed),
                            CovMat.shape[0]-1-len(eigenvalues_computed)
                        )
                    )
                eigenvalues_computed = numpy.concatenate(
                    (eigenvalues_computed,
                    eig_values)
                )
                #eigenvectors_computed.append(eig_vectors)
                if eigenvectors_computed is not None:
                    eigenvectors_computed = numpy.concatenate(
                        (
                            eigenvectors_computed,
                            eig_vectors
                        )
                        ,axis=1
                    )
                else:
                    eigenvectors_computed = eig_vectors
                
                RatioHighestSmallestEig = numpy.max(eigenvalues_computed)/numpy.min(eig_values)
                #print(50*"*")
                #print(RatioHighestSmallestEig,numpy.max(eigenvalues_computed),numpy.min(eig_values))
                #print(eigenvalues_computed)
                #print(eigenvectors_computed.shape)
            eig_values  = eigenvalues_computed
            eig_vectors = eigenvectors_computed
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
            plt.close(fig)
            # Plot the first eigenvectors
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
                        len(eig_values),
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
                # Save the parameters
                filename = os.path.join(
                    self.folder,
                    self.name,
                    "Parameters_" + str(config) + "_" + str(sample) + "_binary.csv"
                )
                with open(filename,"w+") as fin:
                    fin.write("threshold,%.5e;\n"%self.threshold)
                    fin.write("eps_11,%.5e;\n"%eps_11)
                    fin.write("eps_22,%.5e;\n"%eps_22)
                    fin.write("alpha,%.5e;\n"%alpha)
                    fin.write("lx,%.5e;\n"%self.Lengths[0])
                    fin.write("ly,%.5e;\n"%self.Lengths[1])
                    fin.write("nx,%.5e;\n"%self.Nodes[0])
                    fin.write("ny,%.5e;\n"%self.Nodes[1])
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

