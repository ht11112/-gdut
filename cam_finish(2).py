import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from matplotlib import rcParams
import time
from threading import Thread
import matplotlib.pyplot as plt


# 解决中文显示乱码问题
rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体以支持中文显示
rcParams['axes.unicode_minus'] = False  # 确保在使用Unicode字符时不出现负号显示问题

# 定义抛物线方程 y = ax^2 + bx + c
def parabola(x, a, b, c):
    # x: 自变量
    # a, b, c: 抛物线参数
    return a * x**2 + b * x + c

# 推程运动方程
def push_motion(h, delta, delta_0):
    # h: 升程
    # delta: 当前角度
    # delta_0: 推程总角度
    return h * (1 - np.cos(np.pi * delta / delta_0)) / 2

# 回程运动方程
def return_motion(h, delta, delta_0):
    # h: 升程
    # delta: 当前角度
    # delta_0: 回程总角度
    return h * (1 + np.cos(np.pi * delta / delta_0)) / 2

# 数值导数函数
def numerical_derivative(func, x, dx=1e-6):
    # func: 目标函数
    # x: 自变量值
    # dx: 用于计算导数的小增量
    return (func(x + dx) - func(x - dx)) / (2 * dx)

# 等弦长逼近函数
def approximate_curve_equal_chord_length(func, params, start, end, tolerance):
    # func: 目标函数
    # params: 目标函数的参数
    # start: 起始点
    # end: 结束点
    # tolerance: 误差容限
    
    def f(x):
        return func(x, *params)
    
    def f_prime(x):
        return numerical_derivative(f, x)
    
    def f_double_prime(x):
        return numerical_derivative(f_prime, x)
    
    def curvature_radius(x):
        # 计算曲率半径
        return (1 + f_prime(x)**2)**(3/2) / abs(f_double_prime(x))
    
    x_vals = np.linspace(start, end, 400)
    curvatures = [curvature_radius(x_val) for x_val in x_vals]
    R_min = min(curvatures)  # 最小曲率半径
    l = 2 * np.sqrt(2 * R_min * tolerance)  # 等弦长步长
    
    points = [(start, f(start))]
    x_current = start

    while x_current < end:
        step_size = l / np.sqrt(1 + f_prime(x_current)**2)
        x_next = x_current + step_size
        if x_next > end:
            x_next = end
        y_next = f(x_next)
        points.append((x_next, y_next))
        x_current = x_next
    
    return points

# 用等间距的方法逼近曲线
def approximate_curve_equal_distance(func, params, start, end, tolerance):
    # func: 目标函数
    # params: 目标函数的参数
    # start: 起始点
    # end: 结束点
    # tolerance: 误差容限
    
    points = [(start, func(start, *params))]
    x_current = start
    step_size = (end - start) / tolerance  # 初始步长
    
    while x_current < end:
        x_next = x_current + step_size
        if x_next > end:
            x_next = end
        y_current = func(x_current, *params)
        y_next = func(x_next, *params)
        
        mid_x = (x_current + x_next) / 2
        mid_y = func(mid_x, *params)
        line_mid_y = (y_current + y_next) / 2
        
        if abs(mid_y - line_mid_y) > tolerance:
            step_size /= 2  # 减小步长以满足误差容限
            continue
        
        points.append((x_next, y_next))
        x_current = x_next
    
    return points

# 生成凸轮轮廓曲线，支持等弦长和等间距逼近方法
def cam_profile(r, h, push_angle, dwell_angle_far, return_angle, dwell_angle_near, tolerance, method):
    # r: 基圆半径
    # h: 升程
    # push_angle: 推程角度
    # dwell_angle_far: 远休止角度
    # return_angle: 回程角度
    # dwell_angle_near: 近休止角度
    # tolerance: 误差容限
    # method: 逼近方法（等弦长或等距离）

    total_angle = 360  # 总角度
    angles = np.arange(0, total_angle, 1)  # 角度数组
    
    push_steps = int(push_angle / total_angle * len(angles))  # 推程步数
    dwell_far_steps = int(dwell_angle_far / total_angle * len(angles))  # 远休止步数
    return_steps = int(return_angle / total_angle * len(angles))  # 回程步数
    dwell_near_steps = int(dwell_angle_near / total_angle * len(angles))  # 近休止步数

    # 推程阶段
    push_curve = [r + push_motion(h, step, push_steps) for step in range(push_steps)]
    
    # 远休止阶段
    dwell_far_curve = [r + h] * dwell_far_steps
    
    # 回程阶段
    return_curve = [r + return_motion(h, step, return_steps) for step in range(return_steps)]
    
    # 近休止阶段
    dwell_near_curve = [r] * dwell_near_steps
    
    # 组合所有阶段
    radii = push_curve + dwell_far_curve + return_curve + dwell_near_curve
    angles = angles[:len(radii)]
    
    if method == "等弦长":
        points = approximate_curve_equal_chord_length(lambda x: np.interp(x, angles, radii), (), 0, 360, tolerance)
    elif method == "等距离":
        points = approximate_curve_equal_distance(lambda x: np.interp(x, angles, radii), (), 0, 360, tolerance)
    
    angles, radii = zip(*points)
    return angles, radii

# 逐点比较法插补
def point_by_point_interpolation(points, pulse_equivalent):
    # points: 曲线上的点集
    # pulse_equivalent: 脉冲当量

    interpolated_points = []
    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]
        dx = x2 - x1
        dy = y2 - y1
        distance = np.sqrt(dx**2 + dy**2)
        num_steps = max(1, int(distance / pulse_equivalent))  # 计算步数
        x_steps = np.linspace(x1, x2, num_steps)
        y_steps = np.linspace(y1, y2, num_steps)
        for j in range(num_steps):
            interpolated_points.append((x_steps[j], y_steps[j]))
    interpolated_points.append(points[-1])
    return interpolated_points

#在画布生成曲线
def generate_curve():
    global interpolated_points  # 确保在全局范围内定义
    module = module_var.get()  # 选择的模块（抛物线或盘形凸轮）
    method = interpolation_method_var.get()  # 选择的逼近方法（等弦长或等距离）
    tolerance = float(tolerance_entry.get())  # 误差容限
    tool_radius = float(tool_radius_entry.get()) if tool_radius_entry.get() else 0  # 刀具半径
    compensation_type = compensation_type_var.get()  # 刀具补偿类型
    pulse_equivalent = float(pulse_equivalent_entry.get())  # 脉冲当量

    overcut_warning = False  # 初始化过切警告
    overcut_message = ""  # 初始化过切警告信息

    if module == "盘形凸轮":
        r = float(r_entry.get())  # 基圆半径
        h = float(h_entry.get())  # 升程
        push_angle = float(push_angle_entry.get())  # 推程角度
        dwell_angle_far = float(dwell_angle_far_entry.get())  # 远休止角度
        return_angle = float(return_angle_entry.get())  # 回程角度
        dwell_angle_near = float(dwell_angle_near_entry.get())  # 近休止角度

        # 检查刀具半径是否大于基圆半径
        if tool_radius > r:
            overcut_warning = True
            overcut_message = "刀具半径大于基圆半径，可能存在过切情况，请调整参数。"

        # 生成凸轮轮廓曲线
        angles, radii = cam_profile(r, h, push_angle, dwell_angle_far, return_angle, dwell_angle_near, tolerance, method)
        x_points = np.array(radii) * np.cos(np.radians(angles))  # 将极坐标转换为笛卡尔坐标
        y_points = np.array(radii) * np.sin(np.radians(angles))
        points = list(zip(x_points, y_points))  # 组合为点集
        number_of_segments = len(points)
    else:
        # 抛物线模块
        params = [float(a_entry.get()), float(b_entry.get()), float(c_entry.get())]
        if method == "等弦长":
            points = approximate_curve_equal_chord_length(parabola, params, -10, 10, tolerance)
        elif method == "等距离":
            points = approximate_curve_equal_distance(parabola, params, -10, 10, tolerance)
        x_points, y_points = zip(*points)
        number_of_segments = len(points)

    # 根据刀具补偿类型进行补偿
    if compensation_type in ["左刀补", "右刀补"]:
        compensated_points = apply_tool_compensation(points, tool_radius, compensation_type)
    else:
        compensated_points = points

    # 插补点生成
    interpolated_points = point_by_point_interpolation(compensated_points, pulse_equivalent)

    # 检测过切（仅针对抛物线）
    if module == "抛物线" and is_overcut(compensated_points, compensation_type):
        overcut_warning = True
        overcut_message = "在左刀补时可能存在过切情况，请调整参数。"

    # 绘制曲线
    ax.clear()
    if module == "盘形凸轮":
        ax.plot(x_points, y_points, label='原始曲线')
    else:
        x = np.linspace(-10, 10, 400)
        y = parabola(x, *params)
        ax.plot(x, y, label='原始曲线')

    px, py = zip(*compensated_points) if compensated_points else ([], [])
    ix, iy = zip(*interpolated_points) if interpolated_points else ([], [])

    # 绘制逼近点和插补点
    ax.scatter(px, py, color='red', label='逼近点')
    ax.plot(ix, iy, color='green', label='插补线')
    ax.legend()
    canvas.draw()

    steps_label.config(text=f"段数: {number_of_segments}")

    # 显示过切警告
    if overcut_warning:
        messagebox.showwarning("过切警告", overcut_message)

# 判断是否发生过切，遍历所有点，检查下一个点的 x 值是否大于前一个点的 x 值
def is_overcut(points, compensation_type):
    if compensation_type != "左刀补":
        return False

    for i in range(len(points) - 1):
        if points[i+1][0] <= points[i][0]:
            return True
    return False

# 仿真绘制刀具路径
def run_simulation():
    global simulation_running
    try:
        pulse_equivalent = float(pulse_equivalent_entry.get())
        feed_rate = float(feed_rate_entry.get())
        spindle_speed = float(spindle_speed_entry.get())
        tool_radius = float(tool_radius_entry.get()) if tool_radius_entry.get() else 0

        # 确定刷新间隔时间
        refresh_rate = 10 / feed_rate  # 10秒内输出feed_rate个点

        points = interpolated_points  # 使用插补点进行仿真
        simulation_running = True  # 启动仿真

        def draw_toolpath():
            ax.clear()
            ax.plot([p[0] for p in points], [p[1] for p in points], 'r.', label='逼近点')

            def update_point(i):
                if i < len(points) and simulation_running:
                    if i > 0:
                        prev_point = points[i - 1]
                        ax.plot([prev_point[0], points[i][0]], [prev_point[1], points[i][1]], color='blue', linestyle='dotted')
                    # 画刀具半径
                    circle = plt.Circle((points[i][0], points[i][1]), tool_radius, color='green', fill=False)
                    ax.add_artist(circle)
                    canvas.draw()
                    app.after(int(refresh_rate * 1000), update_point, i + 1)
                else:
                    # 最后一个点更新完成后清除圆圈
                    ax.patches = []

            update_point(0)

        draw_toolpath()

    except ValueError as e:
        messagebox.showerror("输入错误", f"请确保所有输入都是有效的数字：{str(e)}")
    except Exception as e:
        messagebox.showerror("错误", f"发生了一个错误：{str(e)}")


# 清除画布
def clear_canvas():
    global simulation_running
    simulation_running = False  # 停止仿真
    ax.clear()
    canvas.draw()
    steps_label.config(text="段数: ")


# 保存NC代码
def save_nc_code():
    module = module_var.get()
    method = interpolation_method_var.get()
    tolerance = float(tolerance_entry.get())
    tool_radius = float(tool_radius_entry.get()) if tool_radius_entry.get() else 0
    compensation_type = compensation_type_var.get()
    pulse_equivalent = float(pulse_equivalent_entry.get())
    tool_x = float(tool_x_entry.get())
    tool_y = float(tool_y_entry.get())
    tool_z = float(tool_z_entry.get())
    workpiece_thickness = float(workpiece_thickness_entry.get())
    
    if module == "盘形凸轮":
        r = float(r_entry.get())
        h = float(h_entry.get())
        push_angle = float(push_angle_entry.get())
        dwell_angle_far = float(dwell_angle_far_entry.get())
        return_angle = float(return_angle_entry.get())
        dwell_angle_near = float(dwell_angle_near_entry.get())
        
        angles, radii = cam_profile(r, h, push_angle, dwell_angle_far, return_angle, dwell_angle_near, tolerance, method)
        x_points = np.array(radii) * np.cos(np.radians(angles))
        y_points = np.array(radii) * np.sin(np.radians(angles))
        points = list(zip(x_points, y_points))
    else:
        params = [float(a_entry.get()), float(b_entry.get()), float(c_entry.get())]
        if method == "等弦长":
            points = approximate_curve_equal_chord_length(parabola, params, -10, 10, tolerance)
        elif method == "等距离":
            points = approximate_curve_equal_distance(parabola, params, -10, 10, tolerance)

    if compensation_type in ["左刀补", "右刀补"]:
        compensated_points = apply_tool_compensation(points, tool_radius, compensation_type)
        points = compensated_points
    
    interpolated_points = point_by_point_interpolation(points, pulse_equivalent)
    
    feed_rate = feed_rate_entry.get()
    spindle_speed = spindle_speed_entry.get()
    is_absolute = coordinate_type_var.get() == "绝对"
    nc_lines = generate_nc_code(points, is_absolute, feed_rate, spindle_speed, compensation_type, tool_x, tool_y, tool_z, workpiece_thickness)
    
    file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files", "*.txt")])
    if file_path:
        with open(file_path, 'w') as file:
            file.write('\n'.join(nc_lines))
        messagebox.showinfo("成功", "NC代码已保存成功！")

    code_text.delete(1.0, tk.END)
    code_text.insert(tk.END, '\n'.join(nc_lines))

# 应用刀具补偿
def apply_tool_compensation(points, tool_radius, compensation_type):
    compensated_points = []
    for i in range(len(points)):
        if i == 0 or i == len(points) - 1:
            compensated_points.append(points[i])
        else:
            x1, y1 = points[i-1]
            x2, y2 = points[i]
            x3, y3 = points[i+1]
            dx1, dy1 = x2 - x1, y2 - y1
            dx2, dy2 = x3 - x2, y3 - y2
            norm1 = np.sqrt(dx1**2 + dy1**2)
            norm2 = np.sqrt(dx2**2 + dy2**2)
            nx1, ny1 = dy1 / norm1, -dx1 / norm1
            nx2, ny2 = dy2 / norm2, -dx2 / norm2
            nx, ny = (nx1 + nx2) / 2, (ny1 + ny2) / 2
            norm = np.sqrt(nx**2 + ny**2)
            nx, ny = nx / norm, ny / norm
            if compensation_type == "左刀补":
                compensated_points.append((x2 - tool_radius * nx, y2 - tool_radius * ny))
            elif compensation_type == "右刀补":
                compensated_points.append((x2 + tool_radius * nx, y2 + tool_radius * ny))
    return compensated_points

# 生成NC代码
def generate_nc_code(points, is_absolute, feed_rate, spindle_speed, compensation_type, tool_x, tool_y, tool_z, workpiece_thickness):
    nc_lines = []

    offset_x = -tool_x
    offset_y = -tool_y
    offset_z = -(tool_z + workpiece_thickness)
    
    if is_absolute:
        if compensation_type == "左刀补":
            nc_lines.append(f"N01 G90 G00 X0 Y0 Z0")
            nc_lines.append(f"N02 G41 G00 X{points[0][0]:.4f} Y{points[0][1]:.4f} F{feed_rate} S{spindle_speed}")
        elif compensation_type == "右刀补":
            nc_lines.append(f"N01 G90 G00 X0 Y0 Z0")
            nc_lines.append(f"N02 G42 G00 X{points[0][0]:.4f} Y{points[0][1]:.4f} F{feed_rate} S{spindle_speed}")
        else:
            nc_lines.append(f"N01 G90 G00 X0 Y0 Z0")
            nc_lines.append(f"N02 G00 X{points[0][0]:.4f} Y{points[0][1]:.4f} F{feed_rate} S{spindle_speed}")
        for i, (px, py) in enumerate(points[1:], start=3):
            nc_lines.append(f"N{i:02d} G01 X{px:.4f} Y{py:.4f}")
        nc_lines.append(f"N{len(points)+2:02d} G00 Z{-offset_z:.4f}")
        nc_lines.append(f"N{len(points)+3:02d} G00 X0 Y0")
    else:
        if compensation_type == "左刀补":
            nc_lines.append(f"N01 G91 G00 X{offset_x:.4f} Y{offset_y:.4f} Z{offset_z:.4f}")
            nc_lines.append(f"N02 G41 G00 X{points[0][0]:.4f} Y{points[0][1]:.4f} F{feed_rate} S{spindle_speed}")
        elif compensation_type == "右刀补":
            nc_lines.append(f"N01 G91 G00 X{offset_x:.4f} Y{offset_y:.4f} Z{offset_z:.4f}")
            nc_lines.append(f"N02 G42 G00 X{points[0][0]:.4f} Y{points[0][1]:.4f} F{feed_rate} S{spindle_speed}")
        else:
            nc_lines.append(f"N01 G91 G00 X{offset_x:.4f} Y{offset_y:.4f} Z{offset_z:.4f}")
            nc_lines.append(f"N02 G00 X{points[0][0]:.4f} Y{points[0][1]:.4f} F{feed_rate} S{spindle_speed}")
        last_x, last_y = points[0]
        for i, (px, py) in enumerate(points[1:], start=3):
            dx = px - last_x
            dy = py - last_y
            nc_lines.append(f"N{i:02d} G01 X{dx:.4f} Y{dy:.4f}")
            last_x, last_y = px, py
        nc_lines.append(f"N{len(points)+2:02d} G00 Z{-offset_z:.4f}")
        nc_lines.append(f"N{len(points)+3:02d} G00 X{-px:.4f} Y{-py:.4f}")
    nc_lines.append(f"N{len(points)+4:02d} G40 M02")  # 取消刀具半径补偿
    
    return nc_lines

# 切换模块
def switch_module():
    module = module_var.get()
    show_parameters()

# 显示参数
def show_parameters():
    module = module_var.get()
    for widget in curve_params_frame.winfo_children():
        widget.grid_remove()
    
    if module == "抛物线":
        a_label.grid(row=0, column=0, padx=5, pady=5)
        a_entry.grid(row=0, column=1, padx=5, pady=5)
        b_label.grid(row=1, column=0, padx=5, pady=5)
        b_entry.grid(row=1, column=1, padx=5, pady=5)
        c_label.grid(row=2, column=0, padx=5, pady=5)
        c_entry.grid(row=2, column=1, padx=5, pady=5)
    elif module == "盘形凸轮":
        r_label.grid(row=0, column=0, padx=5, pady=5)
        r_entry.grid(row=0, column=1, padx=5, pady=5)
        h_label.grid(row=1, column=0, padx=5, pady=5)
        h_entry.grid(row=1, column=1, padx=5, pady=5)
        push_angle_label.grid(row=2, column=0, padx=5, pady=5)
        push_angle_entry.grid(row=2, column=1, padx=5, pady=5)
        dwell_angle_far_label.grid(row=3, column=0, padx=5, pady=5)
        dwell_angle_far_entry.grid(row=3, column=1, padx=5, pady=5)
        return_angle_label.grid(row=4, column=0, padx=5, pady=5)
        return_angle_entry.grid(row=4, column=1, padx=5, pady=5)
        dwell_angle_near_label.grid(row=5, column=0, padx=5, pady=5)
        dwell_angle_near_entry.grid(row=5, column=1, padx=5, pady=5)

# 显示关于信息
def show_about():
    messagebox.showinfo("关于", "小组成员：胡   涛(3121000038)\n                  李梓霖(3121000040)\n                  黄苇苈(3121000039)\n指导老师:杨雪荣")

# 调整画布大小
def resize_canvas(event):
    scale = canvas_size_slider.get()
    ax.set_xlim(-scale, scale)
    ax.set_ylim(-scale, scale)
    canvas.draw()




# 创建主应用窗口
app = tk.Tk()
app.title("曲线插补工具")

# 创建和布局Frame
frame_left = ttk.Frame(app, padding="10")
frame_left.grid(row=0, column=0, sticky='ns')

frame_center = ttk.Frame(app, padding="10")
frame_center.grid(row=0, column=1, sticky='nsew')

frame_right = ttk.Frame(app, padding="10")
frame_right.grid(row=0, column=2, sticky='ns')

frame_bottom = ttk.Frame(app, padding="10")
frame_bottom.grid(row=1, column=4, columnspan=3, sticky='nsew')

# 创建和布局Matplotlib画布
fig = Figure(figsize=(5, 4), dpi=100)
ax = fig.add_subplot(111)
canvas = FigureCanvasTkAgg(fig, master=frame_center)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(expand=1, fill='both')

canvas_size_slider = ttk.Scale(frame_center, from_=1, to=20, orient='horizontal', command=resize_canvas)
canvas_size_slider.set(10)
canvas_size_slider.pack(fill='x')


# 代码显示区
code_label = ttk.Label(frame_right, text="生成NC代码")
code_label.pack()
code_text = tk.Text(frame_right, wrap='word', height=15, width=50)
code_text.pack(expand=1, fill='both')

# 模块选择框
module_var = tk.StringVar(value="抛物线")
module_parabola = ttk.Radiobutton(frame_left, text="抛物线", variable=module_var, value="抛物线", command=switch_module)
module_cam = ttk.Radiobutton(frame_left, text="盘形凸轮", variable=module_var, value="盘形凸轮", command=switch_module)
module_parabola.grid(row=0, column=0, padx=5, pady=5)
module_cam.grid(row=0, column=1, padx=5, pady=5)

# 曲线参数框
curve_params_frame = ttk.LabelFrame(frame_left, text="参数", padding="10")
curve_params_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

a_entry = ttk.Entry(curve_params_frame)
a_label = ttk.Label(curve_params_frame, text="参数 a:")

b_entry = ttk.Entry(curve_params_frame)
b_label = ttk.Label(curve_params_frame, text="参数 b:")

c_entry = ttk.Entry(curve_params_frame)
c_label = ttk.Label(curve_params_frame, text="参数 c:")

r_entry = ttk.Entry(curve_params_frame)
r_label = ttk.Label(curve_params_frame, text="基圆半径 (r):")

h_entry = ttk.Entry(curve_params_frame)
h_label = ttk.Label(curve_params_frame, text="升程 (h):")

push_angle_entry = ttk.Entry(curve_params_frame)
push_angle_label = ttk.Label(curve_params_frame, text="推程角 (度):")

dwell_angle_far_entry = ttk.Entry(curve_params_frame)
dwell_angle_far_label = ttk.Label(curve_params_frame, text="远休止角 (度):")

return_angle_entry = ttk.Entry(curve_params_frame)
return_angle_label = ttk.Label(curve_params_frame, text="回程角 (度):")

dwell_angle_near_entry = ttk.Entry(curve_params_frame)
dwell_angle_near_label = ttk.Label(curve_params_frame, text="近休止角 (度):")

# 机床参数框
machine_params_frame = ttk.LabelFrame(frame_left, text="机床参数", padding="10")
machine_params_frame.grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

feed_rate_entry = ttk.Entry(machine_params_frame)
feed_rate_label = ttk.Label(machine_params_frame, text="进给速度 (F):")
feed_rate_entry.grid(row=0, column=1, padx=5, pady=5)
feed_rate_label.grid(row=0, column=0, padx=5, pady=5)

spindle_speed_entry = ttk.Entry(machine_params_frame)
spindle_speed_label = ttk.Label(machine_params_frame, text="主轴转速 (S):")
spindle_speed_entry.grid(row=1, column=1, padx=5, pady=5)
spindle_speed_label.grid(row=1, column=0, padx=5, pady=5)

coordinate_type_var = tk.StringVar(value="绝对")
coordinate_type_combobox = ttk.Combobox(machine_params_frame, textvariable=coordinate_type_var, values=["绝对", "相对"])
coordinate_type_label = ttk.Label(machine_params_frame, text="坐标类型:")
coordinate_type_combobox.grid(row=2, column=1, padx=5, pady=5)
coordinate_type_label.grid(row=2, column=0, padx=5, pady=5)

# 新增的对刀点X、对刀点Y、对刀点Z、工件厚度参数框
tool_x_entry = ttk.Entry(machine_params_frame)
tool_x_label = ttk.Label(machine_params_frame, text="刀位点X:")
tool_x_entry.grid(row=3, column=1, padx=5, pady=5)
tool_x_label.grid(row=3, column=0, padx=5, pady=5)

tool_y_entry = ttk.Entry(machine_params_frame)
tool_y_label = ttk.Label(machine_params_frame, text="刀位点Y:")
tool_y_entry.grid(row=4, column=1, padx=5, pady=5)
tool_y_label.grid(row=4, column=0, padx=5, pady=5)

tool_z_entry = ttk.Entry(machine_params_frame)
tool_z_label = ttk.Label(machine_params_frame, text="刀位点Z:")
tool_z_entry.grid(row=5, column=1, padx=5, pady=5)
tool_z_label.grid(row=5, column=0, padx=5, pady=5)

workpiece_thickness_entry = ttk.Entry(machine_params_frame)
workpiece_thickness_label = ttk.Label(machine_params_frame, text="工件厚度:")
workpiece_thickness_entry.grid(row=6, column=1, padx=5, pady=5)
workpiece_thickness_label.grid(row=6, column=0, padx=5, pady=5)

# 插补参数框
interpolation_params_frame = ttk.LabelFrame(frame_left, text="插补参数", padding="10")
interpolation_params_frame.grid(row=3, column=0, columnspan=2, sticky="ew", padx=5, pady=5)

tool_radius_entry = ttk.Entry(interpolation_params_frame)
tool_radius_label = ttk.Label(interpolation_params_frame, text="刀具半径:")
tool_radius_entry.grid(row=0, column=1, padx=5, pady=5)
tool_radius_label.grid(row=0, column=0, padx=5, pady=5)

compensation_type_var = tk.StringVar(value="无")
compensation_type_combobox = ttk.Combobox(interpolation_params_frame, textvariable=compensation_type_var, values=["无", "左刀补", "右刀补"])
compensation_type_label = ttk.Label(interpolation_params_frame, text="刀补类型:")
compensation_type_combobox.grid(row=1, column=1, padx=5, pady=5)
compensation_type_label.grid(row=1, column=0, padx=5, pady=5)

interpolation_method_var = tk.StringVar(value="等距离")
interpolation_method_combobox = ttk.Combobox(interpolation_params_frame, textvariable=interpolation_method_var, values=["等距离", "等弦长"])
interpolation_method_label = ttk.Label(interpolation_params_frame, text="插补方法:")
interpolation_method_combobox.grid(row=2, column=1, padx=5, pady=5)
interpolation_method_label.grid(row=2, column=0, padx=5, pady=5)

tolerance_entry = ttk.Entry(interpolation_params_frame)
tolerance_label = ttk.Label(interpolation_params_frame, text="允许误差:")
tolerance_entry.grid(row=3, column=1, padx=5, pady=5)
tolerance_label.grid(row=3, column=0, padx=5, pady=5)

pulse_equivalent_entry = ttk.Entry(interpolation_params_frame)
pulse_equivalent_label = ttk.Label(interpolation_params_frame, text="脉冲当量:")
pulse_equivalent_entry.grid(row=4, column=1, padx=5, pady=5)
pulse_equivalent_label.grid(row=4, column=0, padx=5, pady=5)

# 创建和布局按钮
generate_button = ttk.Button(frame_left, text="生成曲线", command=generate_curve)
generate_button.grid(row=5, column=0, columnspan=2, pady=5, sticky='ew')

run_simulation_button = ttk.Button(frame_left, text="加工仿真", command=run_simulation)
run_simulation_button.grid(row=6, column=0, columnspan=2, pady=5, sticky='ew')

clear_button = ttk.Button(frame_left, text="清除画布", command=clear_canvas)
clear_button.grid(row=7, column=0, columnspan=2, pady=5, sticky='ew')

nc_button = ttk.Button(frame_left, text="生成NC代码", command=save_nc_code)
nc_button.grid(row=8, column=0, columnspan=2, pady=5, sticky='ew')

about_button = ttk.Button(frame_left, text="关于", command=show_about)
about_button.grid(row=9, column=0, columnspan=2, pady=5, sticky='ew')

steps_label = ttk.Label(frame_center, text="段数: ")
steps_label.pack(fill='x')

switch_module()
app.mainloop()
