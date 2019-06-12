import numpy

class StiffnessTensor:

    def __init__(self,file):
        """
        Initializtion of a 4th order stiffness tensor from a file generated with cm3.
        Parameters
        ----------
        file : str
            File name in which the 4th order tensor is stored.
        """
        with open(file,"r") as fin:
            for line in fin:
                if "Time" in line:
                    who_is_who = line.split(";")
                if line[0] == "0":
                    tmp = line.split(";")
                    break
            index = who_is_who.index("Time")
            who_is_who.pop(index)
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

            self.elements = {}

            for name,value in zip(who_is_who,StiffnessFourthOrderTensor):
                self.elements[name] = value

    def get(self,index_1,index_2,index_3,index_4):
        string = "Comp." + str(index_1) + str(index_2) + str(index_3) + str(index_4)
        return self.elements[string]

    def get_orthotropic_properties(self):
        # Build the stiffness matrix of orthotropic materials
        StiffMatOrtho = numpy.zeros((6,6))

        # Index [0,0] : C_1111 -> C_11 (Voigt) -> Comp.0000
        StiffMatOrtho[0,0] = self.get(0,0,0,0)

        # Index [1,1] : C_2222 -> C_22 (Voigt) -> Comp.1111
        StiffMatOrtho[1,1] = self.get(1,1,1,1)

        # Index [2,2] : C_3333 -> C_33 (Voigt) -> Comp.2222
        StiffMatOrtho[2,2] = self.get(2,2,2,2)

        # Index [3,3] : C_2323 -> C_44 (Voigt) -> Comp.1212
        StiffMatOrtho[3,3] = self.get(1,2,1,2)

        # Index [4,4] : C_3131 -> C_55 (Voigt) -> Comp.2020
        StiffMatOrtho[4,4] = self.get(2,0,2,0)

        # Index [5,5] : C_1212 -> C_66 (Voigt) -> Comp.0101
        StiffMatOrtho[5,5] = self.get(0,1,0,1)

        # Index [0,1] and [1,0] : C_1122 -> C_12 (Voigt) -> Comp.0011
        StiffMatOrtho[0,1] = self.get(0,0,1,1)
        StiffMatOrtho[1,0] = self.get(0,0,1,1)
        
        # Index [0,2] and [2,0] : C_1133 -> C_13 (Voigt) -> Comp.0022
        StiffMatOrtho[2,0] = self.get(0,0,2,2)
        StiffMatOrtho[0,2] = self.get(0,0,2,2)
        
        # Index [1,2] and [2,1] : C_2233 -> C_23 (Voigt) -> Comp.1122
        StiffMatOrtho[2,1] = self.get(1,1,2,2)
        StiffMatOrtho[1,2] = self.get(1,1,2,2)

        ComplMatOrtho = numpy.linalg.inv(StiffMatOrtho)
        E_1 = 1.0/ComplMatOrtho[0,0]
        E_2 = 1.0/ComplMatOrtho[1,1]
        E_3 = 1.0/ComplMatOrtho[2,2]
        nu_12 = -ComplMatOrtho[1,0]/ComplMatOrtho[0,0]
        nu_21 = -ComplMatOrtho[0,1]/ComplMatOrtho[1,1]
        nu_13 = -ComplMatOrtho[2,0]/ComplMatOrtho[0,0]
        nu_31 = -ComplMatOrtho[0,2]/ComplMatOrtho[2,2]
        nu_23 = -ComplMatOrtho[2,1]/ComplMatOrtho[1,1]
        nu_32 = -ComplMatOrtho[1,2]/ComplMatOrtho[2,2]
        G_23  = 1.0/ComplMatOrtho[3,3]
        G_31  = 1.0/ComplMatOrtho[4,4]
        G_12  = 1.0/ComplMatOrtho[5,5]

        return E_1, E_2, E_3, nu_12, nu_21, nu_13, nu_31, nu_23, nu_32, G_12, G_23, G_31

    def get_isotropic_properties(self):
        mu   = self.get(0,1,0,1)
        C_11 = self.get(1,1,1,1)
        K    = C_11 - 4.0*mu/3.0
        nu   = (0.5-mu/(3*K))/(1+mu/(3*K))
        E    = 2*mu*(1+nu)
        return E, nu

    def print(self) -> None:
        import pprint
        pprint.pprint(self.elements)