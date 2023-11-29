# https://stackoverflow.com/a/39402483

from math import atan2,degrees,log10
import numpy as np
from matplotlib.transforms import Affine2D

#Label line with line2D label data
def label_line(line,x,label=None,align=True,**kwargs):

    ax = line.axes
    xdata = line.get_xdata()
    ydata = line.get_ydata()

    if (x < xdata[0]) or (x > xdata[-1]):
        print('x label location is outside data range!')
        return

    #Find corresponding y co-ordinate and angle of the line
    ip = 1
    for i in range(len(xdata)):
        if x < xdata[i]:
            ip = i
            break

    y = ydata[ip-1] + (ydata[ip]-ydata[ip-1])*(x-xdata[ip-1])/(xdata[ip]-xdata[ip-1])

    if not label:
        label = line.get_label()

    if align:
        #Compute the slope
        if ax.xaxis.get_scale() == "log":
            dx = log10(xdata[ip]) - log10(xdata[ip-1])
        else:
            dx = xdata[ip] - xdata[ip-1]

        if ax.yaxis.get_scale() == "log":
            dy = log10(ydata[ip]) - log10(ydata[ip-1])
        else:
            dy = ydata[ip] - ydata[ip-1]

        ang = degrees(atan2(dy,dx))

        #Transform to screen co-ordinates
        # pt = np.array([x,y]).reshape((1,2))
        # trans_angle = ax.transData.transform_angles(np.array((ang,)),pt)[0]
        # print(f"dx: {dx}    dy: {dy}   angle: {ang}")

    else:
        ang = 0
        # trans_angle = 0

    #Set a bunch of keyword arguments
    if 'color' not in kwargs:
        kwargs['color'] = line.get_color()

    if ('horizontalalignment' not in kwargs) and ('ha' not in kwargs):
        kwargs['ha'] = 'center'

    if ('verticalalignment' not in kwargs) and ('va' not in kwargs):
        kwargs['va'] = 'center'

    if 'backgroundcolor' not in kwargs:
        kwargs['backgroundcolor'] = ax.get_facecolor()

    if 'clip_on' not in kwargs:
        kwargs['clip_on'] = True

    if 'zorder' not in kwargs:
        kwargs['zorder'] = 2.5

    # ax.text(x,y,label,transform=Affine2D().translate(tx=0, ty=0.0),rotation=ang,**kwargs)
    t = ax.text(x*2.2,y * 1.1,label,rotation=ang,**kwargs)
    t.set_bbox(dict(alpha=0.0, color = kwargs['backgroundcolor']))

def label_lines(lines,align=True,xvals=None,**kwargs):

    ax = lines[0].axes
    lab_lines = []
    labels = []

    #Take only the lines which have labels other than the default ones
    for line in lines:
        label = line.get_label()
        if "_line" not in label:
            lab_lines.append(line)
            labels.append(label)

    if xvals is None:
        xmin,xmax = ax.get_xlim()
        if ax.xaxis.get_scale() == "log":
            xvals = np.logspace(log10(xmin),log10(xmax),len(lab_lines)+2)[1:-1]
        else:
            xvals = np.linspace(xmin,xmax,len(lab_lines)+4)[2:-2]

    print(xvals)
    for line,x,label in zip(lab_lines,xvals,labels):
        label_line(line,x,label,align,**kwargs)
