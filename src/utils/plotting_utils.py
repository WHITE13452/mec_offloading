import matplotlib.pyplot as plt
import matplotlib as mpl
import platform
import warnings

def init_plotting_style():
    """
    初始化绘图样式，包括中文字体配置
    在所有绘图代码前调用此函数
    """
    # 根据操作系统选择合适的中文字体
    system = platform.system()
    
    if system == 'Windows':
        fonts = ['Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi']
    elif system == 'Darwin':  # macOS
        fonts = ['Heiti TC', 'PingFang SC', 'Arial Unicode MS', 'STHeiti']
    else:  # Linux
        fonts = ['WenQuanYi Micro Hei', 'Droid Sans Fallback', 'DejaVu Sans']
    
    # 尝试设置字体
    for font in fonts:
        try:
            plt.rcParams['font.sans-serif'] = [font]
            plt.rcParams['axes.unicode_minus'] = False
            # 测试是否能正常显示中文
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, '测试中文', ha='center', va='center')
            plt.close(fig)
            print(f"成功设置中文字体: {font}")
            break
        except:
            continue
    else:
        warnings.warn("无法自动设置中文字体，请手动配置")
    
    # 设置其他绘图参数
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3
    
    # 设置颜色主题
    plt.rcParams['axes.prop_cycle'] = plt.cycler('color', 
        ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
         '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'])