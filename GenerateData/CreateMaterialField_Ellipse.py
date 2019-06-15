import numpy

import time

import scipy

import os

from GmshObjects import *

from GenerateEllipse import GenerateEllipse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import skimage

class MaterialField_Ellipse:
    """
    Material field.
    """
    def __init__(
        self,
        name,
        folder,
        Lengths,
        Nodes,
        a,
        b,
        t,
        x0,
        y0,
        AngleType = "degree"
    ):
        
        self.name = name
        
        self.folder = folder

        folder = os.path.join(self.folder,self.name)
        if not os.path.exists(folder):
            os.makedirs(folder)

        self.Lengths = Lengths

        assert all(isinstance(x, int) for x in Nodes)
        assert len(Nodes) == 2
        self.Nodes = Nodes

        if AngleType == "degree":
            t = t / 180. * numpy.pi

        # Ellipse semi-axes:
        self.a = a
        self.b = b
        # Ellipse angle:
        self.t = t
        # Ellipse centers
        self.x0 = x0
        self.y0 = y0

    def ToGMSH(self,Do_3D,num_3d_layers=1) -> list:
        """
        Transform realizations of the random material field in GMSH file.
        """
        print("\t" + 50*"-")
        files_geo = []
        domain_size = (self.Nodes[0],self.Nodes[1])

        # Read the material distribution from file
        filename = os.path.join(
            self.folder,
            self.name,
            "Realization_binary"
        )

        mat_distrib = \
            numpy.fromfile(
                filename,
                dtype=numpy.int64
            ).reshape(domain_size)
        
        # Unique values in material field
        mat_values = numpy.unique(mat_distrib)

        # GMSH .geo file
        filename_geo= os.path.join(
            self.folder,
            self.name,
            "Realization_binary.geo"
        )
        files_geo.append(filename_geo)

        start = time.time()
        # Write the .geo file
        with open(filename_geo,"w+") as geo:
            # Domain length
            geo.write("_LxMIN = %.5f;\n"%self.Lengths[0][0])
            geo.write("_LxMAX = %.5f;\n"%self.Lengths[0][1])
            geo.write("_LyMIN = %.5f;\n"%self.Lengths[1][0])
            geo.write("_LyMAX = %.5f;\n"%self.Lengths[1][1])
            
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
            dx = (self.Lengths[0][1]-self.Lengths[0][0])/self.Nodes[0]
            dy = (self.Lengths[1][1]-self.Lengths[1][0])/self.Nodes[1]
            default_mesh_elm_size = dx
            default_line_density = 1

            extrude_z = default_mesh_elm_size
            pos_z = 0.0

            for row in range(self.Nodes[1]+1):
                for col in range(self.Nodes[0]+1):
                    pos_y = dy * row + self.Lengths[0][0]
                    pos_x = dx * col + self.Lengths[1][0]
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
            geo.write("// For y = ylim_min\n")
            geo.write(
                "s[] = Curve In BoundingBox{%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%(\
                    -eps + self.Lengths[0][0],-eps+self.Lengths[1][0],-eps,
                    +eps + self.Lengths[0][1], eps+self.Lengths[1][0], eps
                )
            )
            geo.write(
                "Physical Curve(1) = {s[{0:#s[]-1}]};\n"
            )
            # For x = Lx
            geo.write("// For x = xlim_max\n")
            geo.write(
                "s[] = Curve In BoundingBox{%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%(\
                    -eps + self.Lengths[0][1],-eps + self.Lengths[1][0], -eps,
                    +eps + self.Lengths[0][1], eps + self.Lengths[1][1], eps
                )
            )
            geo.write(
                "Physical Curve(2) = {s[{0:#s[]-1}]};\n"
            )
            # For y = Ly
            geo.write("// For y = ylim_max\n")
            geo.write(
                "s[] = Curve In BoundingBox{%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%(\
                    -eps + self.Lengths[0][0],-eps+self.Lengths[1][1],-eps,
                    +eps + self.Lengths[0][1], eps+self.Lengths[1][1], eps
                )
            )
            geo.write(
                "Physical Curve(3) = {s[{0:#s[]-1}]};\n"
            )
            # For x = 0
            geo.write("// For x = xlim_min\n")
            geo.write(
                "s[] = Curve In BoundingBox{%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%(\
                    -eps+self.Lengths[0][0],-eps + self.Lengths[1][0],-eps,
                    +eps+self.Lengths[0][0], eps + self.Lengths[1][1], eps
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
            geo.write("// For surface with x = xlim_min\n")
            geo.write(
                "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                (
                    -eps+self.Lengths[0][0],self.Lengths[1][0]-eps,-eps,
                    +eps+self.Lengths[0][0],self.Lengths[1][1]+eps,extrude_z+eps
                )
            )

            geo.write(
                "Physical Surface (88880) = {s[{0:#s[]-1}]};\n"
            )

            # Get surfaces with x = self.Lengths[0]
            geo.write("// For surface with x = xlim_max\n")
            geo.write(
                "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                (
                    -eps+self.Lengths[0][1],self.Lengths[1][0]-eps,-eps,
                    +eps+self.Lengths[0][1],self.Lengths[1][1]+eps,extrude_z+eps
                )
            )

            geo.write(
                "Physical Surface (88881) = {s[{0:#s[]-1}]};\n"
            )

            # Get surfaces with y = 0
            geo.write("// For surface with y = ylim_min\n")
            geo.write(
                "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                (
                    -eps+self.Lengths[0][0],-eps+self.Lengths[1][0],-eps,
                    +eps+self.Lengths[0][1],+eps+self.Lengths[1][0],extrude_z+eps
                )
            )

            geo.write(
                "Physical Surface (88882) = {s[{0:#s[]-1}]};\n"
            )
            
            # Get surfaces with y = self.Lengths[1]
            geo.write("// For surface with y = ylim_max\n")
            geo.write(
                "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                (
                    -eps+self.Lengths[0][0],-eps+self.Lengths[1][1],-eps,
                    +eps+self.Lengths[0][1],+eps+self.Lengths[1][1],extrude_z+eps
                )
            )

            geo.write(
                "Physical Surface (88883) = {s[{0:#s[]-1}]};\n"
            )

            # Get surfaces with z = 0
            geo.write(
                "s[] = Surface In BoundingBox {%.15f,%.15f,%.15f,%.15f,%.15f,%.15f};\n"%\
                (
                    -eps+self.Lengths[0][0],
                    -eps+self.Lengths[1][0],
                    -eps,
                    +eps+self.Lengths[0][1],
                    +eps+self.Lengths[1][1],
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
                    -eps+self.Lengths[0][0],
                    -eps+self.Lengths[1][0],
                    -eps+extrude_z,
                    +eps+self.Lengths[0][1],
                    +eps+self.Lengths[1][1],
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

   
    def Create(self) -> None:
        """
        Create the material field, as an ellipse.
        """
        print("\t" + 50*"-")
        # The points where the field is evaluated
        x = numpy.linspace(self.Lengths[0][0], self.Lengths[0][1], self.Nodes[0])
        y = numpy.linspace(self.Lengths[1][0], self.Lengths[1][1], self.Nodes[1])
        XX, YY = numpy.meshgrid(x, y)
        
        ellipse = GenerateEllipse(
            self.a,
            self.b,
            self.t,
            self.x0,
            self.y0
        )

        realization = ellipse.IsInside(XX,YY,[1,-1])

        # Save matrix
        filename = os.path.join(
            self.folder,
            self.name,
            "Realization_binary"
        )

        with open(filename,'wb+') as fin:
            realization.tofile(fin)
 
        # Plot realization and save it
        fig = plt.figure()
        c = plt.imshow(
            realization,
            extent=[self.Lengths[0][0], self.Lengths[0][1], self.Lengths[1][0], self.Lengths[1][1]]
        )
        plt.colorbar(c)
        plt.xlabel('x', fontsize=16)
        plt.ylabel('y', fontsize=16)
        plt.savefig(
            filename + ".eps",
            format = 'eps'
        )
        plt.close(fig)
                
        # Save the parameters
        filename = os.path.join(
            self.folder,
            self.name,
            "Parameters_binary.csv"
        )
        with open(filename,"w+") as fin:
            fin.write("a,%.5e;\n"%self.a)
            fin.write("b,%.5e;\n"%self.b)
            fin.write("t,%.5e;\n"%self.t)
            fin.write("x0,%.5e;\n"%self.x0)
            fin.write("y0,%.5e;\n"%self.y0)
            fin.write("lx_min,%.5e;\n"%self.Lengths[0][0])
            fin.write("lx_max,%.5e;\n"%self.Lengths[0][1])
            fin.write("ly_min,%.5e;\n"%self.Lengths[1][0])
            fin.write("ly_max,%.5e;\n"%self.Lengths[1][1])
            fin.write("nx,%.5e;\n"%self.Nodes[0])
            fin.write("ny,%.5e;\n"%self.Nodes[1])                

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    from LatinHyperCubicSampling import LatinHyperCubicSampling

    domain_limits = [
        [-2.5,2.5],
        [-2.5,2.5]
    ]

    num_nodes = [100,100]

    length_x = domain_limits[0][1]-domain_limits[0][0]
    length_y = domain_limits[1][1]-domain_limits[1][0]

    deltas = [length_x/num_nodes[0],length_y/num_nodes[1]]

    num_points = 200

    points = LatinHyperCubicSampling(
        [0.05*length_x,0.05*length_y,0.0,domain_limits[0][0]/4.0,domain_limits[1][0]/4.0],
        [length_x/2.0,length_y/2.0,numpy.pi,domain_limits[0][1]/4.0,domain_limits[1][1]/4.0],
        num_points
    )

    x = numpy.linspace(
        domain_limits[0][0],
        domain_limits[0][1],
        int((domain_limits[0][1]-domain_limits[0][0])/deltas[0])
    )

    y = numpy.linspace(
        domain_limits[1][0],
        domain_limits[1][1],
        int((domain_limits[1][1]-domain_limits[1][0])/deltas[1])
    )


    x,y = numpy.meshgrid(x,y)

    counter_notinside = 0

    to_take = []

    for cc,point in enumerate(points):

        ellipse = GenerateEllipse(
            point[0],point[1],point[2],point[3],point[4]
        )

        aspect_max = 10

        if not ellipse.LiesInside(domain_limits[0],domain_limits[1])\
            or ellipse.GetAspectRatio() > aspect_max\
            or ellipse.GetAspectRatio() < 1.0/aspect_max:
            title = "Not Inside"
            counter_notinside += 1
            continue
        else:
            title = "Inside"
            to_take.append(cc)

        print(cc,title)

        matfield = MaterialField_Ellipse(
            "test_" + str(cc),
            "./tests/",
            domain_limits,
            num_nodes,
            point[0],
            point[1],
            point[2],
            point[3],
            point[4],
            AngleType="radians"
        )

        matfield.Create()

        matfield.ToGMSH(Do_3D = True,num_3d_layers=1)

    print("> There were ",counter_notinside," ellipses not inside the domain out of ",num_points," !")