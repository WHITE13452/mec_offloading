# 直接运行修复后的绘图代码
from fix_plotting_issues import load_and_plot_results

# 指定您的结果文件路径
result_file = 'results/aoi/experiment_results_20250712_015516.json'

# 重新绘制图表
load_and_plot_results(result_file)