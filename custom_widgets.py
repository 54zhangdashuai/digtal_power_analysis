# custom_widgets.py
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

# ----- Matplotlib 中文设置 -----
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class MplCanvas(FigureCanvas):
    """一个用于嵌入Matplotlib图形的自定义Qt控件"""
    def __init__(self, parent=None, width=7.5, height=5.5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()
        self.fig.patch.set_facecolor('#FFFFFF')