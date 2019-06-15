import numpy

from LatinHyperCubicSampling import LatinHyperCubicSampling

class GenerateEllipse:

    def __init__(self,a,b,t,x0,y0):
        # Ellipse semi-axes:
        self.a = a
        self.b = b
        # Ellipse angle:
        self.t = t
        # Ellipse centers
        self.x0 = x0
        self.y0 = y0

    def GetAspectRatio(self):
        return self.a/self.b

    def IsInside(self,x,y,values):
        assert len(values) == 2
        cos = numpy.cos
        sin = numpy.sin
        term1 = ( (x-self.x0) * cos(self.t) + (y-self.y0) * sin(self.t) )**2 / self.a**2
        term2 = ( (x-self.x0) * sin(self.t) - (y-self.y0) * cos(self.t) )**2 / self.b**2
        return numpy.where(term1 + term2 <= 1.0,values[0],values[1])

    def Draw(self):
        pass

    def LiesInside(self,x_lims,y_lims):
        BB_x,BB_y = self.GetBoundingBox()
        if BB_x[0] < x_lims[0]:
            return False
        if BB_x[1] > x_lims[1]:
            return False
        if BB_y[0] < y_lims[0]:
            return False
        if BB_y[1] > y_lims[1]:
            return False
        return True

    def GetBoundingBox(self):
        tmp1 = numpy.sqrt(self.a**2 * numpy.cos(self.t)**2 + self.b**2 * numpy.sin(self.t)**2)
        tmp2 = numpy.sqrt(self.a**2 * numpy.sin(self.t)**2 + self.b**2 * numpy.cos(self.t)**2)

        x = numpy.asarray([
            -tmp1 + self.x0,
            tmp1 + self.x0
        ])
        
        y = numpy.asarray([
            -tmp2 + self.y0,
            tmp2 + self.y0
        ])
        return x,y

if __name__ == "__main__":

    from matplotlib import pyplot as plt

    domain_limits = [
        [-2.5,2.5],
        [-2.5,2.5]
    ]

    num_nodes = [120,120]

    length_x = domain_limits[0][1]-domain_limits[0][0]
    length_y = domain_limits[1][1]-domain_limits[1][0]

    deltas = [length_x/num_nodes[0],length_y/num_nodes[1]]

    num_points = 1000

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

    for point in points:

        ellipse = GenerateEllipse(
            point[0],point[1],point[2],point[3],point[4]
        )

        if not ellipse.LiesInside(domain_limits[0],domain_limits[1]):
            title = "Not Inside"
            counter_notinside += 1
        else:
            title = "Inside"

        if True:
            continue

        BB_x,BB_y = ellipse.GetBoundingBox()

        res = ellipse.IsInside(x,y,[1,-1])

        plt.scatter(x,y,c=res)

        plt.hlines(BB_y,plt.xlim()[0],plt.xlim()[1])

        plt.vlines(BB_x,plt.ylim()[0],plt.ylim()[1])

        plt.title(title)

        plt.show()

    print("> There were ",counter_notinside," ellipses not inside the domain out of ",num_points," !")