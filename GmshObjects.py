
class Point:
    """
    Representing a point in the 3D space.
    """
    def __init__(self,ID,x,y,z,dens):
        self.ID   = ID
        self.x    = x
        self.y    = y
        self.z    = z
        self.dens = dens

    def ToGMSH(self,file):
        file.write("Point(%d) = {%.3f,%.3f,%.3f,%.1f};\n"%(\
                self.ID,
                self.x,
                self.y,
                self.z,
                self.dens
            ))

class Line:
    """
    Representing a line from one point to another.
    For a vertical line, the orientation is assumed from bottom to top.
    For an horizontal line, the orientation is assumed from left to right.
    """
    def __init__(self,ID,pt_beg,pt_end,scope_label_right,scope_label_left,orientation,dens):
        # ID
        self.ID                = ID
        # Begin and end points
        assert isinstance(pt_beg,Point)
        assert isinstance(pt_end,Point)
        self.pt_beg            = pt_beg
        self.pt_end            = pt_end
        # Labels at left and right of the line
        self.scope_label_right = scope_label_right
        self.scope_label_left  = scope_label_left
        # Orientation
        if not orientation in ["H","V"]:
            raise Exception("Orientation must be H or V.")
        self.orientation = orientation
        # Density:
        self.dens = dens
    
    def ToGMSH(self,file):
        file.write("Line(%d) = {%d,%d};\n"%(\
            self.ID,
            self.pt_beg.ID,
            self.pt_end.ID
        ))
        file.write("Transfinite Line{%d} = %.2f;\n"%(\
            self.ID,
            self.dens))

    def print(self):
        print("> Line %d : Pt(%d) -> Pt(%d) : right(%d) : left(%d) : %s"%(\
            self.ID,self.pt_beg.ID,self.pt_end.ID,self.scope_label_right,self.scope_label_left,self.orientation))

class Surface_2D:
    """
    Representing a 2D surface
    """
    def __init__(self,scope_label,ID):
        self.ID          = ID
        self.scope_label = scope_label
        self.lines       = []
        
    def add_line(self,line):
        assert isinstance(line,Line)
        if line.scope_label_right != self.scope_label and line.scope_label_left != self.scope_label:
            return False
        self.lines.append(line)
        return True
    
    def get_lines_label(self,label):
        """
        Get lines whose label (left or right) is equal to label.
        """
        lines = []
        for line in self.lines:
            if line.scope_label_right == label or line.scope_label_left == label:
                lines.append(line)
        return lines
    
    def print(self):
        print("> Surface 2D :ID(%d) : label(%d)"%(self.ID,self.scope_label))

class Plane_2D:
    """
    Representing a 2D plane composed of points and 2D surfaces.
    """
    def __init__(self,surfaces_2D):
        assert all(isinstance(x,Surface_2D) for x in surfaces_2D)
        self.surfaces_2D = surfaces_2D
        self.points      = []
        self.lines       = []

    def get_point(self,ID):
        for point in self.points:
            if point.ID == ID:
                return point
        raise Exception("Cannot find point with ID " + str(ID))

    def ToGMSH(self,file):
        for point in self.points:
            point.ToGMSH(file)
        for line in self.lines:
            line.ToGMSH(file)
    
    def add_point(self,point):
        assert isinstance(point,Point)
        self.points.append(point)
    
    def add_line(self,line):
        assert isinstance(line,Line)
        self.lines.append(line)
        for surf in self.surfaces_2D:
            if surf.add_line(line) is True:
                return
        raise Exception("Cannot assign this line !")
        
    def get_lines_label(self,label):
        lines = []
        for surf in self.surfaces_2D:
            tmp = surf.get_lines_label(label)
            if tmp:
                lines += tmp
        return lines
    
    def get_surfaces_label(self,label):
        res = []
        for surf in self.surfaces_2D:
            if surf.scope_label == label:
                res.append(surf)
        return res
        
    def get_lines_oriented(self,label):
        lines = self.get_lines_label(label)
        
        IDs = []
        
        for I in range(len(lines)):
            # Horizontal lines
            if lines[I].orientation == "H":
                # Determine if the label if above (must then be oriented left to right)
                #                        or below (must then be oriented right to left)
                if lines[I].scope_label_right == label:
                    # label zone is below line -> put minus sign
                    IDs.append(-lines[I].ID)
                else:
                    IDs.append(lines[I].ID)
                
            # Vertical lines
            else:
                if lines[I].scope_label_right == label:
                    # Label zone is to the right so put minus sign
                    IDs.append(-lines[I].ID)
                else:
                    IDs.append(lines[I].ID)

        return IDs         
    
    def print(self):
        num_p = len(self.points)
        num_s = len(self.surfaces_2D)
        num_l = 0
        for surf in self.surfaces_2D:
            num_l += len(surf.lines)
        print("> Plane_2D with:\n\t> %d points\n\t> %d lines\n\t> %d surfaces 2D\n"%(\
            num_p,num_l,num_s))