import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axisartist import Axes
from matplotlib import rcParams
from mpl_toolkits.axisartist.grid_helper_curvelinear import GridHelperCurveLinear

rcParams['axes.linewidth'] = .2
rcParams['font.size'] = 5

class Transform():
    @staticmethod
    def rotation(angle):
        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle), np.cos(angle)]
                        ])
    
    @staticmethod
    def identity():
        return np.array([[1, 0],
                         [0, 1]
                        ])
    
class Vector(np.ndarray):
    def __new__(cls, input_array, info=None):
        obj = np.asarray(input_array).view(cls)
        obj.x = input_array[0]
        obj.y = input_array[1]
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        self.x = getattr(obj, 'x', None)
        self.y = getattr(obj, 'y', None)

        
class CoordinateSystem():
    def __init__(self, e1, e2, T=Transform.identity()):
        self.vectors = []
        self.e1 = Vector(e1)
        self.e2 = Vector(e2)
        self.x_origin = 0
        self.y_origin = 0
        self.T = T
        
    def transform(self, T):
        T_inv = np.linalg.inv(T)
        c = CoordinateSystem(T.dot(self.e1), T.dot(self.e2), T)
        c.set_vector(self.v)
        return c
    
    def set_vector(self, v):
        self.v = Vector(v)
        
    def plot(self, ax=None):
        
        if ax is None:
            fig, ax = plt.subplots(nrows=1,
                                   ncols=1)
            fig.set_size_inches(5, 5)
            fig.set_dpi(150)
            
        max_int = np.rint(np.abs(np.linalg.norm(self.v))) + 1
        
        major_ticks = np.arange(start=-max_int, stop=max_int)
        minor_ticks = np.arange(start=-max_int, stop=max_int)
        
        ax.set_xticks(major_ticks)
        #ax.set_xticklabels(major_ticks, fontsize=5)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_yticks(major_ticks)
        #ax.set_yticklabels(major_ticks, fontsize=5)
        ax.set_yticks(minor_ticks, minor=True)

        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)
        
        ax.set_xlim([-max_int, max_int])
        ax.set_ylim([-max_int, max_int])
        
        ax.set_aspect(1)
    
        # Plot basis vectors first
        ax.quiver(0, 0, self.e1.x, self.e1.y, units='xy', scale=1, alpha=.8, color='black', headwidth=8, linestyle='dashed')
        ax.quiver(0, 0, self.e2.x, self.e2.y, units='xy', scale=1, alpha=.8, color='black', headwidth=8, linestyle='dashed')
             
        if self.v is not None:
            
            x = self.v.dot(self.e1)
            y = self.v.dot(self.e2)
            
            # Plot vector
            ax.quiver(0,
                      0,
                      self.v.x,
                      self.v.y,
                      units='xy',
                      scale=1,
                      alpha=.8,
                      headwidth=8,
                      color='blue')
            
            ax.quiver(0,
                      0,
                      x,
                      0,
                      units='xy',
                      scale=1,
                      alpha=.8,
                      headwidth=8,
                      color='red')
            
            ax.quiver(0,
                      0,
                      0,
                      y,
                      units='xy',
                      scale=1,
                      alpha=.8,
                      headwidth=8,
                      color='red')
                        
            #x = self.e1.dot(self.v)
            #y = self.e2.dot(self.v)
            
            # Transform according to transformation law
            #v1 = self.T.dot(np.array([x, 0]))
            #v2 = self.T.dot(np.array([0, y]))

            #ax.quiver(self.x_origin, self.y_origin, v1[0], v1[1], units='xy', scale=1, alpha=.6, color='red', headwidth=4, linestyle='dashed')
            #ax.quiver(self.x_origin, self.y_origin, v2[0], v2[1], units='xy', scale=1, alpha=.6, color='red', headwidth=4, linestyle='dashed')
            
def compare(c1, c2):
    
    fig = plt.figure()
    fig.set_size_inches(5, 5)
    fig.set_dpi(150)
      
    def tr(x, y):
        return c2.T.dot([x, y])

    def inv_tr(x, y):
        v = c2.T.T.dot([x, y])
        return v[0], v[1]
    
    grid_helper = GridHelperCurveLinear((tr, inv_tr))

    # Instruct matplotlib to create object from AxisArtist instead
    ax1 = fig.add_subplot(1, 2, 1, axes_class=Axes)
    ax2 = fig.add_subplot(1, 2, 2, axes_class=Axes, grid_helper=grid_helper)
    
    # Add axis that (in general) have transformed
    ax2.axis["t1"] = ax2.new_floating_axis(nth_coord=0, value=0)
    ax2.axis["t2"] = ax2.new_floating_axis(nth_coord=1, value=0)
    
    ax2.axis['top'].set_visible(False)
    ax2.axis['bottom'].set_visible(False)
    ax2.axis['left'].set_visible(False)
    ax2.axis['right'].set_visible(False)
    
    c1.plot(ax1)
    c2.plot(ax2)
    print('          c1           c2')
    print('')
    print(f'   e1   {c1.e1}    {c2.e1}')
    print(f'   e2   {c1.e2}    {c2.e2}')
    print('')
    print(f'||e1||  {np.linalg.norm(c1.e1)}      {np.linalg.norm(c2.e1)}')
    print(f'||e2||  {np.linalg.norm(c1.e2)}      {np.linalg.norm(c2.e2)}')
    print('')
    print(f'||T||   {np.linalg.norm(c1.v)}        {np.linalg.norm(c2.v)}')
    print('')
    print(f'   Tx   {c1.v.x}        {c2.v.x}')
    print(f'   Ty   {c1.v.y}        {c1.v.y}')
    
    
c1 = CoordinateSystem([1, 0],
                      [0, 1])
c1.set_vector([3, 3])
#T = Transform.rotation(angle=np.pi / 5)
T = Transform.rotation(angle=np.pi / 4)
c2 = c1.transform(T)
compare(c1, c2)