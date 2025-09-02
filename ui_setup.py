# ui_setup.py

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox, 
    QListWidget, QSplitter, QGroupBox, QTextEdit, QTableWidget, 
    QTableWidgetItem, QHeaderView, QCheckBox, QSpinBox, QDoubleSpinBox, 
    QMenu, QTabWidget, QStackedWidget, QFormLayout, QStyle, QGridLayout, QLineEdit,
    QFrame
)
from PyQt5.QtGui import QFont, QIcon, QIntValidator
from PyQt5.QtCore import Qt, QSize

class UiSetupMixin:
    """一个包含所有UI创建和布局方法的Mixin类 (v6.8 默认值与样式优化)"""

    def initUI(self):
        self.setWindowTitle('数字电源IDE - 理论与实践 v6.8 (默认值与样式优化)    作者：张辰')
        self.setGeometry(80, 60, 1600, 940)
        self.setWindowIcon(QIcon("favicon.ico")) 
        self._apply_styles()
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        
        main_splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(main_splitter)
        
        left_panel = self._create_left_panel()
        right_panel = self._create_right_panel()
        
        left_panel.setMinimumWidth(450)
        left_panel.setMaximumWidth(700)
        
        main_splitter.addWidget(left_panel)
        main_splitter.addWidget(right_panel)
        main_splitter.setSizes([500, 1100])
        
        self.statusBar().showMessage("准备就绪")

    def _create_left_panel(self):
        left_panel_container = QWidget()
        content_layout = QVBoxLayout(left_panel_container)
        content_layout.setContentsMargins(0, 0, 0, 0)
        
        left_splitter = QSplitter(Qt.Vertical)
        left_splitter.setChildrenCollapsible(False)

        self.main_tabs = QTabWidget()
        data_analysis_tab = self._create_data_analysis_tab()
        design_tab = self._create_design_tab()
        self.main_tabs.addTab(data_analysis_tab, "数据分析与处理")
        self.main_tabs.addTab(design_tab, "理论模型设计")
        
        result_group = QGroupBox("信息输出")
        result_layout = QVBoxLayout(result_group)
        self.result_stack = QStackedWidget()
        placeholder_widget = QLabel("结果将在此处显示")
        placeholder_widget.setAlignment(Qt.AlignCenter)
        self.result_text = QTextEdit(); self.result_text.setReadOnly(True)
        self.result_table = QTableWidget()
        self.result_stack.addWidget(placeholder_widget)
        self.result_stack.addWidget(self.result_text)
        self.result_stack.addWidget(self.result_table)
        result_layout.addWidget(self.result_stack)
        
        left_splitter.addWidget(self.main_tabs)
        left_splitter.addWidget(result_group)
        
        left_splitter.setStretchFactor(0, 1)
        left_splitter.setStretchFactor(1, 1)
        
        content_layout.addWidget(left_splitter)
        
        return left_panel_container
        
    def _create_data_analysis_tab(self):
        tab_widget = QWidget()
        main_layout = QVBoxLayout(tab_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)

        file_group = self._create_file_group()
        column_select_group = self._create_column_select_group()
        analysis_group = self._create_analysis_group()
        
        main_layout.addWidget(file_group)
        main_layout.addWidget(column_select_group)
        main_layout.addWidget(analysis_group)
        main_layout.addStretch(1)
        
        return tab_widget
    
    def _create_right_panel(self):
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        from custom_widgets import MplCanvas

        right_splitter = QSplitter(Qt.Vertical)
        canvas_container = QWidget()
        canvas_layout = QVBoxLayout(canvas_container)
        canvas_layout.setContentsMargins(0,0,0,0)
        
        self.canvas_analysis = MplCanvas(self)
        self.toolbar_analysis = NavigationToolbar(self.canvas_analysis, self)
        
        toolbar_ext_layout = QHBoxLayout()
        toolbar_ext_layout.addWidget(self.toolbar_analysis)
        toolbar_ext_layout.addStretch(1)

        self.bode_tracker_check = QCheckBox("Bode图跟踪"); self.bode_tracker_check.setEnabled(False)
        toolbar_ext_layout.addWidget(self.bode_tracker_check)
        
        self.cursor_check = QCheckBox("启用光标")
        self.x_cursor_check = QCheckBox("X轴光标"); self.x_cursor_check.setChecked(True); self.x_cursor_check.setEnabled(False)
        self.y_cursor_check = QCheckBox("Y轴光标"); self.y_cursor_check.setChecked(False); self.y_cursor_check.setEnabled(False)
        toolbar_ext_layout.addWidget(self.cursor_check)
        toolbar_ext_layout.addWidget(self.x_cursor_check)
        toolbar_ext_layout.addWidget(self.y_cursor_check)
        
        canvas_layout.addLayout(toolbar_ext_layout)
        
        controls_container = QWidget()
        controls_layout = QHBoxLayout(controls_container)
        controls_layout.setContentsMargins(5, 0, 5, 5)

        self.axis_control_group = self._create_axis_control_group()
        controls_layout.addWidget(self.axis_control_group, 2)

        roi_group = self._create_roi_group()
        controls_layout.addWidget(roi_group, 2)
        
        action_buttons_group = self._create_action_buttons_group()
        controls_layout.addWidget(action_buttons_group, 1)
        
        canvas_layout.addWidget(controls_container)
        canvas_layout.addWidget(self.canvas_analysis, 1)
        self.cursor_group = self._create_cursor_group()
        
        right_splitter.addWidget(canvas_container)
        right_splitter.addWidget(self.cursor_group)
        right_splitter.setSizes([700, 240])
        return right_splitter
    
    def _create_file_group(self):
        group = QGroupBox("文件操作")
        layout = QVBoxLayout(group)
        layout.setSpacing(6)
        
        button_layout = QGridLayout()
        self.btn_open_files = QPushButton(" 打开文件")
        self.btn_open_files.setIcon(self.style().standardIcon(QStyle.SP_FileIcon))
        self.btn_open_folder = QPushButton(" 打开文件夹")
        self.btn_open_folder.setIcon(self.style().standardIcon(QStyle.SP_DirIcon))
        button_layout.addWidget(self.btn_open_files, 0, 0)
        button_layout.addWidget(self.btn_open_folder, 0, 1)

        self.no_header_check = QCheckBox("文件无标题行")
        self.no_header_check.setToolTip("勾选后将按列位置(0,1,2...)读取数据，并自动生成数字列名")

        layout.addLayout(button_layout)
        layout.addWidget(self.no_header_check)
        layout.addWidget(QLabel("已加载的文件 (右键可关闭):"))
        self.file_list = QListWidget()
        self.file_list.setContextMenuPolicy(Qt.CustomContextMenu)
        layout.addWidget(self.file_list)
        return group

    def _create_column_select_group(self):
        group = QGroupBox("数据列选择")
        layout = QVBoxLayout(group)
        form_layout = QFormLayout()
        form_layout.setSpacing(8)
        
        self.x_axis_combo = QComboBox()
        self.y_axis_combo = QComboBox()
        form_layout.addRow("X轴/时间轴数据列:", self.x_axis_combo)
        form_layout.addRow("Y轴数据列 (单列):", self.y_axis_combo)
        layout.addLayout(form_layout)
        
        self.multi_column_check = QCheckBox("启用多列分析")
        layout.addWidget(self.multi_column_check)
        self.multi_column_list = QListWidget()
        self.multi_column_list.setSelectionMode(QListWidget.MultiSelection)
        layout.addWidget(self.multi_column_list)
        
        btns_layout = QGridLayout()
        self.btn_select_all = QPushButton("全选")
        self.btn_clear_sel = QPushButton("清空")
        btns_layout.addWidget(self.btn_select_all, 0, 0)
        btns_layout.addWidget(self.btn_clear_sel, 0, 1)
        layout.addLayout(btns_layout)
        
        return group

    def _create_roi_group(self):
        group = QGroupBox("数据范围选择 (ROI)")
        layout = QGridLayout(group)
        
        self.roi_enable_check = QCheckBox("启用范围选择进行分析")
        self.roi_enable_check.setChecked(False)
        
        self.roi_xmin_spin = QDoubleSpinBox()
        self.roi_xmax_spin = QDoubleSpinBox()
        
        for spin in [self.roi_xmin_spin, self.roi_xmax_spin]:
            spin.setDecimals(6); spin.setRange(-1e12, 1e12); spin.setEnabled(False)
        
        self.btn_select_roi = QPushButton("鼠标框选范围"); self.btn_select_roi.setEnabled(False)
        self.btn_clear_roi = QPushButton("清除范围"); self.btn_clear_roi.setEnabled(False)

        layout.addWidget(self.roi_enable_check, 0, 0, 1, 4)
        layout.addWidget(QLabel("X-Min:"), 1, 0); layout.addWidget(self.roi_xmin_spin, 1, 1)
        layout.addWidget(QLabel("X-Max:"), 1, 2); layout.addWidget(self.roi_xmax_spin, 1, 3)
        layout.addWidget(self.btn_select_roi, 2, 0, 1, 2)
        layout.addWidget(self.btn_clear_roi, 2, 2, 1, 2)
        
        return group

    def _create_analysis_group(self):
        group = QGroupBox("分析与参数设置")
        layout = QVBoxLayout(group)
        layout.setSpacing(8)
        
        type_layout = QHBoxLayout()
        type_layout.addWidget(QLabel("分析类型:"))
        self.analysis_type = QComboBox()
        self.analysis_type.addItems([
            "负载瞬态响应分析", "环路稳定性分析 (Bode Plot)", "传递函数分析 (Bode Plot)",
            "FFT / 频域分析", "PLL性能分析", "ADC校准分析", "高级数据拟合 (Advanced Fitting)"
        ])
        type_layout.addWidget(self.analysis_type, 1)
        layout.addLayout(type_layout)

        self.param_stack = QStackedWidget()
        self.param_stack.addWidget(self._create_transient_param_widget())
        self.param_stack.addWidget(self._create_bode_param_widget()) 
        self.param_stack.addWidget(self._create_tf_bode_param_widget())
        self.param_stack.addWidget(self._create_fft_param_widget())
        self.param_stack.addWidget(self._create_pll_param_widget())
        self.param_stack.addWidget(self._create_adc_param_widget())
        self.param_stack.addWidget(self._create_fitting_param_widget())
        layout.addWidget(self.param_stack)
        
        ### MODIFICATION START: Set default UI state for analysis type ###
        default_analysis_index = 1 # "环路稳定性分析" is the 2nd item
        self.analysis_type.setCurrentIndex(default_analysis_index)
        self.param_stack.setCurrentIndex(default_analysis_index) # Also set the param stack
        ### MODIFICATION END ###

        self.analysis_type.currentIndexChanged.connect(self.param_stack.setCurrentIndex)
        
        gen_param_layout = QFormLayout()
        self.sample_spin = QSpinBox(); self.sample_spin.setRange(1, 1000); self.sample_spin.setValue(1)
        self.sample_spin.setToolTip("增大此值可以加快大数据量波形的绘制速度")
        gen_param_layout.addRow("绘图抽样步长:", self.sample_spin)
        layout.addLayout(gen_param_layout)

        return group
    
    def _create_design_tab(self):
        # This function and its sub-functions remain unchanged
        # ... (code is identical to previous versions)
        tab = QWidget()
        layout = QVBoxLayout(tab)
        self.design_tabs = QTabWidget()
        buck_tab = self._create_buck_designer_tab()
        pfc_tab = self._create_pfc_designer_tab()
        llc_tab = self._create_llc_designer_tab()
        compensator_tab = self._create_compensator_designer_tab()
        self.design_tabs.addTab(buck_tab, "Buck 设计器")
        self.design_tabs.addTab(pfc_tab, "Boost PFC 设计器")
        self.design_tabs.addTab(llc_tab, "LLC 设计器")
        self.design_tabs.addTab(compensator_tab, "数字补偿器设计器")
        layout.addWidget(self.design_tabs)
        return tab

    def _create_buck_designer_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        params_group = QGroupBox("设计参数")
        params_layout = QFormLayout(params_group)
        self.buck_vin = QLineEdit("12"); self.buck_vout = QLineEdit("5")
        self.buck_iout = QLineEdit("3"); self.buck_fs = QLineEdit("200000")
        self.buck_delta_il = QLineEdit("30"); self.buck_delta_vout = QLineEdit("1")
        params_layout.addRow("输入电压 Vin (V):", self.buck_vin)
        params_layout.addRow("输出电压 Vout (V):", self.buck_vout)
        params_layout.addRow("输出电流 Iout (A):", self.buck_iout)
        params_layout.addRow("开关频率 fs (Hz):", self.buck_fs)
        params_layout.addRow("电感电流纹波 (%):", self.buck_delta_il)
        params_layout.addRow("输出电压纹波 (%):", self.buck_delta_vout)
        self.btn_calc_buck = QPushButton("计算设计参数")
        self.btn_overlay_buck_bode = QPushButton("叠加理论Bode图")
        self.btn_send_buck_tf = QPushButton("发送TF到补偿器设计器")
        self.btn_send_buck_tf.setStyleSheet("background-color: #DAF7A6;")
        main_layout.addWidget(params_group)
        main_layout.addWidget(self.btn_calc_buck)
        main_layout.addWidget(self.btn_overlay_buck_bode)
        main_layout.addWidget(self.btn_send_buck_tf)
        main_layout.addStretch()
        return widget
        
    def _create_pfc_designer_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        params_group = QGroupBox("设计参数")
        params_layout = QFormLayout(params_group)
        self.pfc_vac_min = QLineEdit("85"); self.pfc_vac_max = QLineEdit("265")
        self.pfc_f_line = QLineEdit("50"); self.pfc_vout = QLineEdit("400")
        self.pfc_pout = QLineEdit("300"); self.pfc_fs = QLineEdit("65000")
        self.pfc_eff = QLineEdit("95"); self.pfc_delta_il = QLineEdit("30")
        params_layout.addRow("最小输入电压 Vac_min (Vrms):", self.pfc_vac_min)
        params_layout.addRow("最大输入电压 Vac_max (Vrms):", self.pfc_vac_max)
        params_layout.addRow("电网频率 f_line (Hz):", self.pfc_f_line)
        params_layout.addRow("输出直流电压 Vout (V):", self.pfc_vout)
        params_layout.addRow("额定输出功率 Pout (W):", self.pfc_pout)
        params_layout.addRow("开关频率 fs (Hz):", self.pfc_fs)
        params_layout.addRow("预估效率 eff (%):", self.pfc_eff)
        params_layout.addRow("电感电流纹波 (% @ low-line peak):", self.pfc_delta_il)
        self.btn_calc_pfc = QPushButton("计算设计参数")
        self.btn_overlay_pfc_bode = QPushButton("叠加理论Bode图")
        self.btn_send_pfc_tf = QPushButton("发送TF到补偿器设计器")
        self.btn_send_pfc_tf.setStyleSheet("background-color: #DAF7A6;")
        main_layout.addWidget(params_group)
        main_layout.addWidget(self.btn_calc_pfc)
        main_layout.addWidget(self.btn_overlay_pfc_bode)
        main_layout.addWidget(self.btn_send_pfc_tf)
        main_layout.addStretch()
        return widget

    def _create_llc_designer_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        params_group = QGroupBox("设计参数 (基于FHA)")
        params_layout = QFormLayout(params_group)
        self.llc_vin = QLineEdit("400"); self.llc_vout = QLineEdit("12")
        self.llc_pout = QLineEdit("300"); self.llc_fr = QLineEdit("100000")
        self.llc_n = QLineEdit("16"); self.llc_q = QLineEdit("0.4")
        self.llc_k = QLineEdit("5")
        params_layout.addRow("输入电压 Vin (V):", self.llc_vin)
        params_layout.addRow("输出电压 Vout (V):", self.llc_vout)
        params_layout.addRow("输出功率 Pout (W):", self.llc_pout)
        params_layout.addRow("谐振频率 fr (Hz):", self.llc_fr)
        params_layout.addRow("变压器匝比 n (Np/Ns):", self.llc_n)
        params_layout.addRow("品质因数 Q:", self.llc_q)
        params_layout.addRow("电感比 k (Lm/Lr):", self.llc_k)
        self.btn_calc_llc = QPushButton("计算谐振参数")
        self.btn_plot_llc_gain = QPushButton("绘制理论增益曲线")
        main_layout.addWidget(params_group)
        main_layout.addWidget(self.btn_calc_llc)
        main_layout.addWidget(self.btn_plot_llc_gain)
        main_layout.addStretch()
        return widget

    def _create_compensator_designer_tab(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        params_group = QGroupBox("设计目标")
        params_layout = QFormLayout(params_group)
        self.comp_plant_gain = QLineEdit("-5.5")
        self.comp_plant_phase = QLineEdit("-120")
        self.comp_fc = QLineEdit("20000")
        self.comp_pm = QLineEdit("60")
        params_layout.addRow("被控对象增益 @fc (dB):", self.comp_plant_gain)
        params_layout.addRow("被控对象相位 @fc (deg):", self.comp_plant_phase)
        params_layout.addRow("期望穿越频率 fc (Hz):", self.comp_fc)
        params_layout.addRow("期望相位裕度 PM (deg):", self.comp_pm)
        self.btn_design_comp = QPushButton("设计补偿器")
        results_group = QGroupBox("设计结果")
        results_layout = QVBoxLayout(results_group)
        self.comp_c_code_selector = QComboBox()
        self.comp_c_code_selector.addItems(["浮点 (Float)", "定点 (Q15)"])
        self.comp_c_code_output = QTextEdit()
        self.comp_c_code_output.setReadOnly(True)
        self.comp_c_code_output.setFont(QFont("Courier New", 10))
        results_layout.addWidget(QLabel("C代码生成:"))
        results_layout.addWidget(self.comp_c_code_selector)
        results_layout.addWidget(self.comp_c_code_output)
        self.btn_plot_comp_bode = QPushButton("绘制补偿器Bode图")
        main_layout.addWidget(params_group)
        main_layout.addWidget(self.btn_design_comp)
        main_layout.addWidget(results_group)
        main_layout.addWidget(self.btn_plot_comp_bode)
        main_layout.addStretch()
        return widget
    
    # ... (All other _create... methods and _apply_styles are unchanged from the last version) ...
    # ... (为了简洁省略，请只替换上面修改的部分) ...
    def on_analysis_type_changed(self, index):
        self.param_stack.setCurrentIndex(index)
        self._update_action_buttons_state()
    def _create_fitting_param_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        self.fitting_method_combo = QComboBox()
        self.fitting_method_combo.addItems(["线性拟合 (Linear)", "多项式拟合 (Polynomial)", "趋势振荡信号拟合 (Trending Oscillation)", "包络线分析 (Envelope Analysis)"])
        layout.addRow("拟合模型:", self.fitting_method_combo)
        self.fitting_params_stack = QStackedWidget()
        poly_widget = QWidget(); poly_layout = QFormLayout(poly_widget)
        self.fitting_poly_degree = QSpinBox(); self.fitting_poly_degree.setRange(2, 10); self.fitting_poly_degree.setValue(2)
        poly_layout.addRow("多项式阶数:", self.fitting_poly_degree)
        env_widget = QWidget(); env_layout = QFormLayout(env_widget)
        self.fitting_env_method = QComboBox(); self.fitting_env_method.addItems(['linear', 'polynomial'])
        self.fitting_env_poly_degree = QSpinBox(); self.fitting_env_poly_degree.setRange(1, 10)
        env_layout.addRow("包络拟合方法:", self.fitting_env_method)
        env_layout.addRow("多项式阶数:", self.fitting_env_poly_degree)
        self.fitting_params_stack.addWidget(QWidget())
        self.fitting_params_stack.addWidget(poly_widget)
        self.fitting_params_stack.addWidget(QWidget())
        self.fitting_params_stack.addWidget(env_widget)
        self.fitting_method_combo.currentIndexChanged.connect(self.fitting_params_stack.setCurrentIndex)
        layout.addRow(self.fitting_params_stack)
        return widget
    def _create_adc_param_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        self.adc_fit_method_combo = QComboBox()
        self.adc_fit_method_combo.addItems(["线性拟合 (1阶)", "二次多项式拟合 (2阶)", "三次多项式拟合 (3阶)"])
        layout.addRow("拟合方法:", self.adc_fit_method_combo)
        return widget
    def _create_transient_param_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        self.transient_method_combo = QComboBox(); self.transient_method_combo.addItems(["带宽法", "RMSE", "标准差"])
        layout.addRow("恢复时间判定方法:", self.transient_method_combo)
        self.transient_band_spin = QDoubleSpinBox(); self.transient_band_spin.setRange(0.1, 50.0); self.transient_band_spin.setDecimals(1); self.transient_band_spin.setValue(1.0)
        self.transient_band_label = QLabel("稳态容限带宽(±%):")
        layout.addRow(self.transient_band_label, self.transient_band_spin)
        self.transient_window_combo = QComboBox(); self.transient_window_combo.addItems(["10", "20", "50", "100", "200", "500", "1000"]); self.transient_window_combo.setCurrentText("100")
        self.transient_window_combo.setEditable(True); self.transient_window_combo.setValidator(QIntValidator(2, 100000, self))
        self.transient_win_label = QLabel("滑动窗口大小:")
        layout.addRow(self.transient_win_label, self.transient_window_combo)
        self.transient_thresh_spin = QDoubleSpinBox(); self.transient_thresh_spin.setRange(0.000001, 1.0); self.transient_thresh_spin.setDecimals(6); self.transient_thresh_spin.setValue(0.0002)
        self.transient_thresh_label = QLabel("稳态判定阈值 <")
        layout.addRow(self.transient_thresh_label, self.transient_thresh_spin)
        return widget
    def _create_bode_param_widget(self):
        widget = QWidget()
        main_layout = QVBoxLayout(widget)
        main_layout.setContentsMargins(0, 5, 0, 5)
        selection_group = QGroupBox("数据列选择")
        layout = QFormLayout(selection_group)
        layout.setContentsMargins(5, 10, 5, 10)
        self.bode_col_select_mode = QComboBox()
        self.bode_col_select_mode.addItems(["按名称", "按位置"])
        self.bode_col_select_mode.setCurrentIndex(1)
        layout.addRow("列选择模式:", self.bode_col_select_mode)
        self.bode_freq_combo = QComboBox()
        self.bode_gain_combo = QComboBox()
        self.bode_phase_combo = QComboBox()
        layout.addRow("频率列 (Hz):", self.bode_freq_combo)
        layout.addRow("增益列 (dB):", self.bode_gain_combo)
        layout.addRow("相位列 (deg):", self.bode_phase_combo)
        main_layout.addWidget(selection_group)
        self.fitting_group = QGroupBox("传递函数拟合 (系统辨识)")
        self.fitting_group.setCheckable(True)
        self.fitting_group.setChecked(False)
        fit_layout = QFormLayout(self.fitting_group)
        fit_layout.setContentsMargins(5, 10, 5, 10)
        self.fit_zeros_spin = QSpinBox(); self.fit_zeros_spin.setRange(0, 10)
        self.fit_poles_spin = QSpinBox(); self.fit_poles_spin.setRange(1, 10); self.fit_poles_spin.setValue(2)
        fit_layout.addRow("零点数量:", self.fit_zeros_spin)
        fit_layout.addRow("极点数量:", self.fit_poles_spin)
        main_layout.addWidget(self.fitting_group)
        main_layout.addStretch()
        return widget
    def _create_tf_bode_param_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        self.tf_bode_num = QLineEdit("1.0")
        self.tf_bode_den = QLineEdit("1.0, 1.0")
        layout.addRow("分子系数 (高阶->低阶):", self.tf_bode_num)
        layout.addRow("分母系数 (高阶->低阶):", self.tf_bode_den)
        return widget
    def _create_fft_param_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        self.fft_auto_fs_check = QCheckBox("根据时间列自动检测采样率"); self.fft_auto_fs_check.setChecked(True)
        layout.addRow(self.fft_auto_fs_check)
        self.fft_fs_spin = QDoubleSpinBox(); self.fft_fs_spin.setRange(1, 1e12); self.fft_fs_spin.setDecimals(2); self.fft_fs_spin.setValue(1000000.00)
        self.fft_fs_label = QLabel("手动输入采样率 (Fs):")
        layout.addRow(self.fft_fs_label, self.fft_fs_spin)
        return widget
    def _create_pll_param_widget(self):
        widget = QWidget()
        layout = QFormLayout(widget)
        self.pll_method_combo = QComboBox(); self.pll_method_combo.addItems(["带宽法", "RMSE", "标准差"])
        layout.addRow("锁定判定方法:", self.pll_method_combo)
        self.pll_band_spin = QDoubleSpinBox(); self.pll_band_spin.setRange(0.1, 50.0); self.pll_band_spin.setDecimals(1); self.pll_band_spin.setValue(1.0)
        self.pll_band_label = QLabel("稳态容限带宽(±%):")
        layout.addRow(self.pll_band_label, self.pll_band_spin)
        self.pll_window_combo = QComboBox(); self.pll_window_combo.addItems(["10", "20", "50", "100", "200", "500", "1000"]); self.pll_window_combo.setCurrentText("100")
        self.pll_window_combo.setEditable(True); self.pll_window_combo.setValidator(QIntValidator(2, 100000, self))
        self.pll_win_label = QLabel("滑动窗口大小:")
        layout.addRow(self.pll_win_label, self.pll_window_combo)
        self.pll_thresh_spin = QDoubleSpinBox(); self.pll_thresh_spin.setRange(0.000001, 1.0); self.pll_thresh_spin.setDecimals(6); self.pll_thresh_spin.setValue(0.0002)
        self.pll_thresh_label = QLabel("稳态判定阈值 <")
        layout.addRow(self.pll_thresh_label, self.pll_thresh_spin)
        return widget
    def _create_action_buttons_group(self):
        group = QGroupBox("执行与导出");
        layout = QGridLayout(group)
        font = QFont(); font.setPointSize(12); font.setBold(True)
        self.btn_analyze=QPushButton(" 执行分析/绘图"); self.btn_analyze.setFont(font); self.btn_analyze.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay)); self.btn_analyze.setIconSize(QSize(20, 20));
        self.btn_show_osc=QPushButton(" 查看原始数据"); self.btn_show_osc.setIcon(self.style().standardIcon(QStyle.SP_FileDialogDetailedView));
        self.btn_clear_plot = QPushButton("清空所有曲线"); self.btn_clear_plot.setIcon(self.style().standardIcon(QStyle.SP_DialogResetButton));
        self.btn_export=QPushButton(" 导出Excel"); self.btn_export.setIcon(self.style().standardIcon(QStyle.SP_DialogSaveButton));
        self.btn_save_fig=QPushButton(" 保存图像"); self.btn_save_fig.setIcon(self.style().standardIcon(QStyle.SP_DriveHDIcon));
        layout.addWidget(self.btn_analyze, 0, 0, 1, 2)
        layout.addWidget(self.btn_show_osc, 1, 0)
        layout.addWidget(self.btn_clear_plot, 1, 1)
        layout.addWidget(self.btn_export, 2, 0)
        layout.addWidget(self.btn_save_fig, 2, 1)
        return group
    def _create_axis_control_group(self):
        group = QGroupBox("坐标轴控制")
        layout = QGridLayout(group)
        self.x_min_spin=QDoubleSpinBox(); layout.addWidget(QLabel("X-Min:"),0,0); layout.addWidget(self.x_min_spin,0,1)
        self.x_max_spin=QDoubleSpinBox(); layout.addWidget(QLabel("X-Max:"),0,2); layout.addWidget(self.x_max_spin,0,3)
        self.y_min_spin=QDoubleSpinBox(); layout.addWidget(QLabel("Y-Min:"),1,0); layout.addWidget(self.y_min_spin,1,1)
        self.y_max_spin=QDoubleSpinBox(); layout.addWidget(QLabel("Y-Max:"),1,2); layout.addWidget(self.y_max_spin,1,3)
        for spin in [self.x_min_spin, self.x_max_spin, self.y_min_spin, self.y_max_spin]:
            spin.setDecimals(6); spin.setRange(-1e12, 1e12)
        self.btn_apply_axis_limits=QPushButton("应用")
        self.btn_reset_axis_limits=QPushButton("重置")
        layout.addWidget(self.btn_apply_axis_limits, 0, 4)
        layout.addWidget(self.btn_reset_axis_limits, 1, 4)
        group.setEnabled(False)
        return group
    def _create_cursor_group(self):
        group = QGroupBox("光标分析")
        group.setVisible(False)
        main_layout = QVBoxLayout(group)
        main_layout.setContentsMargins(10, 15, 10, 10)
        cursor_splitter = QSplitter(Qt.Vertical)
        self.cursor_info_table = QTableWidget()
        self.cursor_info_table.setEditTriggers(QTableWidget.NoEditTriggers); self.cursor_info_table.setRowCount(2); self.cursor_info_table.setColumnCount(8)
        self.cursor_info_table.horizontalHeader().setVisible(False); self.cursor_info_table.setVerticalHeaderLabels(["参数", "值"])
        self.cursor_info_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch); self.cursor_info_table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.cursor_stats_table = QTableWidget(); self.cursor_stats_table.setEditTriggers(QTableWidget.NoEditTriggers)
        cursor_splitter.addWidget(self.cursor_info_table)
        cursor_splitter.addWidget(self.cursor_stats_table)
        cursor_splitter.setSizes([90, 150])
        self.measure_group = QGroupBox("自动测量 (选中区域)")
        measure_layout = QHBoxLayout(self.measure_group)
        self.btn_measure_rise_time = QPushButton("上升时间")
        self.btn_measure_fall_time = QPushButton("下降时间")
        self.btn_measure_frequency = QPushButton("频率/周期")
        measure_layout.addWidget(self.btn_measure_rise_time)
        measure_layout.addWidget(self.btn_measure_fall_time)
        measure_layout.addWidget(self.btn_measure_frequency)
        main_layout.addWidget(cursor_splitter)
        main_layout.addWidget(self.measure_group)
        return group

    def _apply_styles(self):
        from PyQt5.QtWidgets import QApplication
        self.setFont(QFont("Microsoft YaHei",10));QApplication.instance().setStyleSheet("""
            QWidget{background-color:#F0F0F0;color:#212121;font-family:"Microsoft YaHei";}
            QMainWindow,QDialog{background-color:#F0F0F0;}
            
            /*** MODIFICATION START: Fix clipped GroupBox titles ***/
            QGroupBox{
                border:1px solid #CCCCCC;
                border-radius:5px;
                margin-top:1ex;
                padding-top: 0.8ex; /* Add padding to push content down */
                font-weight:bold;
            }
            /*** MODIFICATION END ***/

            QGroupBox::title{
                subcontrol-origin:margin;
                subcontrol-position:top left;
                padding:0 5px;
                color:#005A9C;
            }
            QGroupBox:checkable::title{padding-left: 20px;}
            QGroupBox::indicator{width: 16px; height: 16px; subcontrol-position: left; left: 5px;}
            QLineEdit,QSpinBox,QDoubleSpinBox,QTextEdit,QComboBox,QListWidget,QTableWidget{background-color:#FFFFFF;border:1px solid #BDBDBD;border-radius:3px;padding:4px;color:#212121;}
            QLineEdit:focus,QSpinBox:focus,QDoubleSpinBox:focus,QTextEdit:focus,QComboBox:focus,QListWidget:focus{border:1px solid #007ACC;}
            QPushButton{background-color:#E0E0E0;border:1px solid #BDBDBD;padding:5px 10px;border-radius:3px;}
            QPushButton:hover{background-color:#EAEAEA;border-color:#007ACC;}
            QPushButton:pressed{background-color:#D5D5D5;}
            QPushButton:disabled{background-color:#E8E8E8;color:#AAAAAA;border-color:#D5D5D5;}
            QListWidget::item:selected,QTableWidget::item:selected{background-color:#007ACC;color:#FFFFFF;}
            QHeaderView::section{background-color:#EAEAEA;border:1px solid #CCCCCC;padding:4px;font-weight:bold;}
            QTabWidget::pane{border-top:1px solid #CCCCCC;}
            QTabBar::tab{background:#E0E0E0;border:1px solid #CCCCCC;padding:6px 12px;border-bottom:none;border-top-left-radius:4px;border-top-right-radius:4px;}
            QTabBar::tab:hover{background:#EAEAEA;}
            QTabBar::tab:selected{background:#F0F0F0;color:#005A9C;font-weight:bold;border-bottom:1px solid #F0F0F0;}
        """)