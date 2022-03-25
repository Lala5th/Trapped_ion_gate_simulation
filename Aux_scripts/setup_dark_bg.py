import matplotlib.pyplot as plt
from cycler import cycler
from PIL import ImageColor

if plt.rcParams['axes.facecolor'] != "#303030":
    a = plt.rcParams['axes.prop_cycle']
    plt.style.use("dark_background")
    plt.rcParams['axes.prop_cycle'] = a
    half = lambda x : int(255 - (255 - x)/1.5)
    n = []
    for c in a:
        x = (ImageColor.getcolor(c['color'],"RGB"))
        n.append('#%02x%02x%02x' % (half(x[0]),half(x[1]),half(x[2])))
    plt.rcParams['axes.prop_cycle'] = cycler(color=n)
    plt.rcParams['axes.facecolor'] = "#303030"
    plt.rcParams['figure.facecolor'] = "#303030"

