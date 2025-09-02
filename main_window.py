# main_window.py

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.signal as signal
from matplotlib.widgets import RectangleSelector

from PyQt5.QtWidgets import (
    QMainWindow, QFileDialog, QMessageBox, QTableWidgetItem, QApplication, QHeaderView, QMenu
)
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QFont

# 导入本地模块
from ui_setup import UiSetupMixin
from analysis_logic import AnalysisMixin
# custom_widgets.py 包含 MplCanvas，它在 ui_setup.py 中被间接导入和使用

# 导入逻辑计算模块
from logic import buck_logic, pfc_boost_logic, llc_resonant_logic, control_logic

class DataAnalysisApp(QMainWindow, UiSetupMixin, AnalysisMixin):
    def __init__(self):
        super().__init__()
        # 建立对核心Qt应用和matplotlib的引用，以便Mixin类可以访问
        self.QApplication = QApplication.instance()
        self.plt = plt 

        # --- 应用程序核心状态变量 ---
        self.data = None
        self.current_file = None
        self.loaded_files = {}
        self.last_table_df = None
        self.plotted_traces = [] # 用于管理叠加绘图
        
        # --- 理论设计模块的状态变量 ---
        self.compensator_results = {}
        self.current_compensator_tf = None
        self.current_plant_tf = None
        self.llc_gain_curve_data = None
        
        # --- 光标和绘图状态变量 ---
        self.v_cursor1, self.v_cursor2, self.h_cursor1, self.h_cursor2 = None, None, None, None
        self.active_cursor = None
        self.cursor_events_connected = False
        self.cursor_plot_data = {}
        self.full_x_limits, self.full_y_limits = None, None
        self.axis_limits_cid = None
        
        self.RESULT_PLACEHOLDER_IDX = 0
        self.RESULT_TEXT_IDX = 1
        self.RESULT_TABLE_IDX = 2

        self.bode_tracker_annotations = []
        self.bode_tracker_connection_id = None

        # 构建UI (来自UiSetupMixin)
        self.initUI()
        
        # 连接所有信号与槽
        self._connect_signals()
        
        # 设置按钮的初始状态
        self.on_multi_toggle()
        self._update_action_buttons_state()
        
        # 初始化结果显示区
        self.result_stack.setCurrentIndex(self.RESULT_PLACEHOLDER_IDX)

        self.analysis_roi = {'xmin': None, 'xmax': None}
        self.is_roi_selection_mode = False
        self.roi_selector = None
        self.roi_span = None

    def plot_fit_results(self, x_data, y_data, y_fit, title):
        """通用拟合结果绘图函数，支持残差图"""
        self.clear_plot(show_message=False)
        self.canvas_analysis.fig.suptitle(title, fontsize=14)

        # 创建主图和残差图的栅格
        gs = self.canvas_analysis.fig.add_gridspec(3, 1)
        ax1 = self.canvas_analysis.fig.add_subplot(gs[0:2, 0]) # 主图占2/3
        ax2 = self.canvas_analysis.fig.add_subplot(gs[2, 0], sharex=ax1) # 残差图占1/3

        # 绘制主图
        ax1.plot(x_data, y_data, 'o', label='原始数据', markersize=3, alpha=0.6)
        line, = ax1.plot(x_data, y_fit, '-', label='拟合曲线', lw=2, color='r')
        ax1.set_ylabel("Y值")
        ax1.legend()
        ax1.grid(True)
        self.plt.setp(ax1.get_xticklabels(), visible=False) # 隐藏主图x轴刻度

        # 绘制残差图
        residuals = y_data - y_fit
        ax2.plot(x_data, residuals, 'o', markersize=3, color='g')
        ax2.axhline(0, color='black', linestyle='--')
        ax2.set_xlabel("X值")
        ax2.set_ylabel("残差")
        ax2.grid(True)
        
        self.canvas_analysis.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.canvas_analysis.draw()

        # 记录绘图状态
        self.plotted_traces.append({'name': 'Fit', 'lines': [line], 'type': 'single'})
        self.update_bode_tracker_visibility()

    def _connect_signals(self):
        """集中连接所有UI控件的信号到对应的处理函数(槽)"""
        # --- 主选项卡切换 ---
        self.main_tabs.currentChanged.connect(self._update_action_buttons_state)

        # --- 数据分析页 ---
        self.btn_open_files.clicked.connect(self.open_files)
        self.btn_open_folder.clicked.connect(self.open_folder)
        self.file_list.currentRowChanged.connect(self.switch_file)
        self.file_list.customContextMenuRequested.connect(self.show_file_list_menu)
        self.x_axis_combo.currentIndexChanged.connect(self._update_action_buttons_state)
        self.y_axis_combo.currentIndexChanged.connect(self._update_action_buttons_state)
        self.multi_column_check.stateChanged.connect(self.on_multi_toggle)
        self.multi_column_list.itemSelectionChanged.connect(self._update_action_buttons_state)
        self.btn_select_all.clicked.connect(self.select_all_multi)
        self.btn_clear_sel.clicked.connect(self.clear_multi_selection)
        self.analysis_type.currentIndexChanged.connect(self.on_analysis_type_changed)
        self.transient_method_combo.currentIndexChanged.connect(self.update_transient_params_visibility)
        self.pll_method_combo.currentIndexChanged.connect(self.update_pll_params_visibility)
        self.fft_auto_fs_check.stateChanged.connect(lambda state: self.fft_fs_spin.setEnabled(not state))
        self.btn_analyze.clicked.connect(self.analyze_data)
        self.btn_show_osc.clicked.connect(self.show_oscilloscope_view)
        
        # --- 理论设计页 ---
        self.btn_calc_buck.clicked.connect(self.on_calculate_buck)
        self.btn_overlay_buck_bode.clicked.connect(self.on_overlay_buck_bode)
        self.btn_send_buck_tf.clicked.connect(self.on_send_tf_to_compensator) # NEW
        self.btn_calc_pfc.clicked.connect(self.on_calculate_pfc)
        self.btn_overlay_pfc_bode.clicked.connect(self.on_overlay_pfc_bode)
        self.btn_send_pfc_tf.clicked.connect(self.on_send_tf_to_compensator) # NEW
        self.btn_calc_llc.clicked.connect(self.on_calculate_llc)
        self.btn_plot_llc_gain.clicked.connect(self.on_plot_llc_gain_curve)
        self.btn_design_comp.clicked.connect(self.on_design_compensator)
        self.comp_c_code_selector.currentTextChanged.connect(self.on_c_code_format_change)
        self.btn_plot_comp_bode.clicked.connect(self.on_plot_compensator_bode)

        # --- 通用功能与绘图区 ---
        self.btn_clear_plot.clicked.connect(self.clear_plot)
        self.btn_export.clicked.connect(self.export_excel)
        self.btn_save_fig.clicked.connect(self.save_figure)
        self.cursor_check.stateChanged.connect(self.toggle_cursors)
        self.x_cursor_check.stateChanged.connect(self.update_cursor_visibility)
        self.y_cursor_check.stateChanged.connect(self.update_cursor_visibility)
        
        self.btn_apply_axis_limits.clicked.connect(self.apply_manual_axis_limits)
        self.btn_reset_axis_limits.clicked.connect(self.reset_axis_limits_view)

        self.bode_tracker_check.stateChanged.connect(self.toggle_bode_tracker)
        self.btn_measure_rise_time.clicked.connect(self.on_measure_rise_time)
        self.btn_measure_fall_time.clicked.connect(self.on_measure_fall_time)
        self.btn_measure_frequency.clicked.connect(self.on_measure_frequency)

        # --- ROI信号连接 ---
        self.roi_enable_check.stateChanged.connect(self.on_roi_enable_changed)
        self.btn_select_roi.clicked.connect(self.on_start_roi_selection)
        self.btn_clear_roi.clicked.connect(self.on_clear_roi)
        self.roi_xmin_spin.valueChanged.connect(self.on_roi_spinbox_changed)
        self.roi_xmax_spin.valueChanged.connect(self.on_roi_spinbox_changed)

    # --- ROI处理方法 ---

    def on_roi_enable_changed(self, state):
        """启用/禁用ROI功能的总开关"""
        is_enabled = (state == Qt.Checked)
        self.roi_xmin_spin.setEnabled(is_enabled)
        self.roi_xmax_spin.setEnabled(is_enabled)
        self.btn_select_roi.setEnabled(is_enabled)
        self.btn_clear_roi.setEnabled(is_enabled)
        if not is_enabled:
            self.on_clear_roi()

    def on_clear_roi(self):
        """清除ROI选择"""
        self.analysis_roi = {'xmin': None, 'xmax': None}
        self.roi_xmin_spin.blockSignals(True); self.roi_xmin_spin.clear(); self.roi_xmin_spin.blockSignals(False)
        self.roi_xmax_spin.blockSignals(True); self.roi_xmax_spin.clear(); self.roi_xmax_spin.blockSignals(False)
        self.update_roi_visuals()
        self.statusBar().showMessage("数据分析范围已清除。")

    def on_roi_spinbox_changed(self):
        """当手动输入范围时更新状态"""
        self.analysis_roi['xmin'] = self.roi_xmin_spin.value()
        self.analysis_roi['xmax'] = self.roi_xmax_spin.value()
        self.update_roi_visuals()

    def on_start_roi_selection(self):
        """启动鼠标框选模式"""
        if not self.plotted_traces or len(self.canvas_analysis.fig.axes) != 1 or self.plotted_traces[0].get('type') != 'single':
            QMessageBox.warning(self, "操作无效", "请先绘制一个时域波形图，才能使用鼠标框选范围。")
            return

        self.is_roi_selection_mode = True
        QApplication.setOverrideCursor(Qt.CrossCursor)
        self.statusBar().showMessage("请在图上拖动鼠标左键选择范围，按'ESC'键取消。")
        
        ax = self.canvas_analysis.fig.axes[0]
        self.roi_selector = RectangleSelector(
            ax, self.on_roi_selected,
            useblit=True,
            button=[1],  # 左键
            minspanx=5, minspany=5,
            spancoords='data',
            interactive=True
        )
        self.roi_selector_cid = self.canvas_analysis.fig.canvas.mpl_connect('key_press_event', self.on_roi_cancel)

    def on_roi_selected(self, eclick, erelease):
        """鼠标框选完成后的回调函数"""
        xmin, xmax = sorted([eclick.xdata, erelease.xdata])
        
        self.roi_xmin_spin.setValue(xmin)
        self.roi_xmax_spin.setValue(xmax)
        
        self.on_roi_spinbox_changed() # 更新状态和视觉效果
        self.deactivate_roi_selector()

    def on_roi_cancel(self, event):
        """处理ESC键取消框选"""
        if event.key == 'escape':
            self.deactivate_roi_selector()
    
    def deactivate_roi_selector(self):
        """停用框选功能并恢复状态"""
        if self.roi_selector:
            self.roi_selector.set_active(False)
            self.canvas_analysis.fig.canvas.mpl_disconnect(self.roi_selector_cid)
            self.roi_selector = None
        
        QApplication.restoreOverrideCursor()
        self.is_roi_selection_mode = False
        self.statusBar().showMessage("范围选择完成。")

    def update_roi_visuals(self):
        """在图上绘制或更新/移除ROI的黄色区域"""
        if not self.canvas_analysis.fig.axes or len(self.canvas_analysis.fig.axes) != 1: return

        ax = self.canvas_analysis.fig.axes[0]
        
        if self.roi_span and self.roi_span in ax.patches:
            self.roi_span.remove()
            self.roi_span = None
            
        xmin = self.analysis_roi.get('xmin')
        xmax = self.analysis_roi.get('xmax')
        
        if xmin is not None and xmax is not None and xmax > xmin:
            self.roi_span = ax.axvspan(xmin, xmax, color='yellow', alpha=0.3)
        
        self.canvas_analysis.draw_idle()
            

    # --- 槽函数 for "理论模型设计" Tab ---

    def on_calculate_buck(self):
        try:
            params = {
                'vin': float(self.buck_vin.text()), 'vout': float(self.buck_vout.text()),
                'iout': float(self.buck_iout.text()), 'fs': float(self.buck_fs.text()),
                'delta_il_pct': float(self.buck_delta_il.text()),
                'delta_vout_pct': float(self.buck_delta_vout.text()),
            }
            results = buck_logic.calculate_buck(params)
            if results.get('error'):
                self.show_text(f"计算错误: {results['error']}")
                self.current_plant_tf = None
                return

            self.current_plant_tf = signal.TransferFunction(results['plant_tf_num'], results['plant_tf_den'])
            
            text = (f"--- Buck 设计结果 ---\n"
                    f"占空比 D: {results['duty_cycle']:.3f}\n"
                    f"推荐电感 L: {results['inductor_uH']:.2f} uH\n"
                    f"推荐电容 C: {results['capacitor_uF']:.2f} uF\n"
                    f"峰值电流 IL_peak: {results['i_peak_A']:.2f} A\n"
                    f"开关应力 V_sw: {results['v_stress_sw_V']:.2f} V\n"
                    f"\n--- 理论传递函数 ---\n"
                    f"分子系数: {results['plant_tf_num']}\n"
                    f"分母系数: {results['plant_tf_den']}\n")
            self.show_text(text)
            self.statusBar().showMessage("Buck参数计算完成。")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "所有设计参数都必须是有效的数字。")
            self.current_plant_tf = None

    def on_overlay_buck_bode(self):
        ### MODIFICATION START: Improved overlay logic ###
        if self.current_plant_tf is None:
            QMessageBox.warning(self, "无模型", "请先点击'计算设计参数'生成理论模型。")
            return
        
        is_bode_plot = len(self.canvas_analysis.fig.axes) == 2 and any(p.get('type') == 'bode' for p in self.plotted_traces)
        if not is_bode_plot:
            reply = QMessageBox.question(self, '操作确认', '当前绘图区不是Bode图。是否要清空绘图区并绘制新的理论Bode图？',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.analyze_tf_bode_plot(tf=self.current_plant_tf, name="理论模型 (Buck)", clear_plot=not is_bode_plot)
        self.main_tabs.setCurrentIndex(0)
        ### MODIFICATION END ###

    def on_calculate_pfc(self):
        try:
            params = {
                'vac_min': float(self.pfc_vac_min.text()), 'vac_max': float(self.pfc_vac_max.text()),
                'f_line': float(self.pfc_f_line.text()), 'vout': float(self.pfc_vout.text()),
                'pout': float(self.pfc_pout.text()), 'fs': float(self.pfc_fs.text()),
                'efficiency': float(self.pfc_eff.text()), 'delta_il_pct': float(self.pfc_delta_il.text()),
            }
            results = pfc_boost_logic.calculate_pfc_boost(params)
            if results.get('error'):
                self.show_text(f"计算错误: {results['error']}")
                self.current_plant_tf = None
                return

            self.current_plant_tf = signal.TransferFunction(results['plant_tf_num'], results['plant_tf_den'])
            
            text = (f"--- Boost PFC 设计结果 ---\n"
                    f"最大输入峰值电流: {results['input_peak_current_A']:.2f} A\n"
                    f"最大占空比: {results['max_duty_cycle']:.3f}\n"
                    f"推荐电感 L: {results['inductor_uH']:.2f} uH\n"
                    f"推荐输出电容 C: {results['output_capacitor_uF']:.2f} uF\n"
                    f"开关/二极管电压应力: {results['switch_stress_V']:.2f} V\n"
                    f"开关峰值电流应力: {results['switch_peak_current_A']:.2f} A\n")
            self.show_text(text)
            self.statusBar().showMessage("PFC参数计算完成。")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "所有设计参数都必须是有效的数字。")
            self.current_plant_tf = None

    def on_overlay_pfc_bode(self):
        ### MODIFICATION START: Improved overlay logic ###
        if self.current_plant_tf is None:
            QMessageBox.warning(self, "无模型", "请先点击'计算设计参数'生成理论模型。")
            return

        is_bode_plot = len(self.canvas_analysis.fig.axes) == 2 and any(p.get('type') == 'bode' for p in self.plotted_traces)
        if not is_bode_plot:
            reply = QMessageBox.question(self, '操作确认', '当前绘图区不是Bode图。是否要清空绘图区并绘制新的理论Bode图？',
                                       QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if reply == QMessageBox.No:
                return

        self.analyze_tf_bode_plot(tf=self.current_plant_tf, name="理论模型 (PFC)", clear_plot=not is_bode_plot)
        self.main_tabs.setCurrentIndex(0)
        ### MODIFICATION END ###

    def on_calculate_llc(self):
        try:
            params = {
                'vin': float(self.llc_vin.text()), 'vout': float(self.llc_vout.text()),
                'pout': float(self.llc_pout.text()), 'fr': float(self.llc_fr.text()),
                'n': float(self.llc_n.text()), 'q': float(self.llc_q.text()),
                'k': float(self.llc_k.text()),
            }
            results = llc_resonant_logic.calculate_llc(params)
            if results.get('error'):
                self.show_text(f"计算错误: {results['error']}")
                self.llc_gain_curve_data = None
                return

            self.llc_gain_curve_data = results
            
            text = (f"--- LLC 谐振腔设计结果 ---\n"
                    f"等效负载电阻 Rac: {results['equiv_resistance_rac_ohm']:.2f} Ω\n"
                    f"谐振电感 Lr: {results['resonant_inductor_lr_uH']:.3f} uH\n"
                    f"谐振电容 Cr: {results['resonant_capacitor_cr_nF']:.3f} nF\n"
                    f"激磁电感 Lm: {results['magnetizing_inductor_lm_uH']:.2f} uH\n")
            self.show_text(text)
            self.statusBar().showMessage("LLC参数计算完成。")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "所有设计参数都必须是有效的数字。")
            self.llc_gain_curve_data = None

    def on_plot_llc_gain_curve(self):
        if self.llc_gain_curve_data is None:
            QMessageBox.warning(self, "无数据", "请先点击'计算谐振参数'生成增益曲线数据。")
            return
        fn = self.llc_gain_curve_data['gain_curve_fn']
        gain = self.llc_gain_curve_data['gain_curve_M']
        self.plot_llc_gain_curve(fn, gain, name="理论增益曲线 (LLC)")
        self.main_tabs.setCurrentIndex(0)

    def on_design_compensator(self):
        try:
            params = {
                'plant_gain_db': float(self.comp_plant_gain.text()),
                'plant_phase': float(self.comp_plant_phase.text()),
                'fc': float(self.comp_fc.text()),
                'pm_req': float(self.comp_pm.text()),
            }
            results = control_logic.design_compensator(params)
            if results.get('error'):
                self.show_text(f"设计错误: {results['error']}")
                self.compensator_results = {}
                self.current_compensator_tf = None
                return
            
            self.compensator_results = results
            self.current_compensator_tf = results['compensator_tf']
            self.on_c_code_format_change()
            self.show_text("补偿器设计完成，已生成C代码。")
            self.statusBar().showMessage("补偿器设计完成。")
        except ValueError:
            QMessageBox.warning(self, "输入错误", "所有设计参数都必须是有效的数字。")
            self.compensator_results = {}
            self.current_compensator_tf = None
    
    ### MODIFICATION START: New function to link designers ###
    def on_send_tf_to_compensator(self):
        """将当前计算出的被控对象TF信息发送到补偿器设计器"""
        if self.current_plant_tf is None:
            QMessageBox.warning(self, "无模型", "请先计算Buck或PFC的设计参数以生成传递函数模型。")
            return
        
        try:
            fc = float(self.comp_fc.text())
        except ValueError:
            QMessageBox.warning(self, "输入错误", "请在补偿器设计器中输入有效的期望穿越频率fc。")
            return

        wc = 2 * np.pi * fc
        
        # 计算在该频率下的增益和相位
        w, mag, phase = signal.bode(self.current_plant_tf, w=[wc])
        
        plant_gain_db = mag[0]
        plant_phase = phase[0]

        self.comp_plant_gain.setText(f"{plant_gain_db:.2f}")
        self.comp_plant_phase.setText(f"{plant_phase:.2f}")

        # 切换到补偿器设计器选项卡
        self.design_tabs.setCurrentIndex(3) # 补偿器是第4个tab
        QMessageBox.information(self, "操作成功", f"已在频率 {fc} Hz 处计算被控对象特性，并自动填入补偿器设计器。")
    ### MODIFICATION END ###

    def on_c_code_format_change(self):
        if not self.compensator_results: 
            self.comp_c_code_output.clear()
            return
            
        format_choice = self.comp_c_code_selector.currentText()
        if "Q15" in format_choice:
            self.comp_c_code_output.setText(self.compensator_results.get('c_code_q15', ''))
        else:
            self.comp_c_code_output.setText(self.compensator_results.get('c_code_float', ''))

    def on_plot_compensator_bode(self):
        if self.current_compensator_tf is None:
            QMessageBox.warning(self, "无模型", "请先点击'设计补偿器'生成模型。")
            return
        self.analyze_tf_bode_plot(tf=self.current_compensator_tf, name="理论补偿器", clear_plot=True)
        self.main_tabs.setCurrentIndex(0)

    # --- 绘图与通用方法 ---
    
    def plot_bode(self, freq, mag, phase, name="Trace", clear_plot=True):
        self.axis_control_group.setEnabled(False)
        if clear_plot:
            self.clear_plot(show_message=False)
        
        if not self.plotted_traces or len(self.canvas_analysis.fig.axes) < 2 or any(p.get('type') != 'bode' for p in self.plotted_traces):
            self._clear_figure() 
            self.plotted_traces = []
            ax1 = self.canvas_analysis.fig.add_subplot(2, 1, 1)
            ax2 = self.canvas_analysis.fig.add_subplot(2, 1, 2, sharex=ax1)
            ax1.set_ylabel('增益 (dB)'); ax1.grid(True, which="both", ls="--")
            ax2.set_ylabel('相位 (度)'); ax2.set_xlabel('频率 (Hz)'); ax2.grid(True, which="both", ls="--")
        else:
            ax1, ax2 = self.canvas_analysis.fig.axes

        line1, = ax1.semilogx(freq, mag, label=name)
        line2, = ax2.semilogx(freq, phase, label=name)
        
        trace_data = {
            'name': name, 'lines': [line1, line2], 'type': 'bode',
            'data': {'freq': freq, 'mag': mag, 'phase': phase}
        }
        self.plotted_traces.append(trace_data)
        
        ax1.legend(); ax2.legend()
        self.canvas_analysis.draw()
        self.update_bode_tracker_visibility()

    def plot_llc_gain_curve(self, fn, gain, name="LLC Gain"):
        self.axis_control_group.setEnabled(False)
        self.clear_plot(show_message=False)
        ax = self.canvas_analysis.fig.add_subplot(1, 1, 1)
        
        line, = ax.plot(fn, gain, label=name)
        ax.axhline(1.0, color='r', linestyle='--', label="增益 M=1")
        ax.set_title("LLC 谐振腔增益曲线")
        ax.set_xlabel("归一化频率 fn (fs/fr)")
        ax.set_ylabel("直流增益 M")
        ax.grid(True, which="both")
        ax.legend()
        
        self.plotted_traces.append({'name': name, 'lines': [line], 'type': 'single'})
        self.canvas_analysis.draw()

    def update_bode_tracker_visibility(self):
        """根据当前绘图类型，决定Bode跟踪器复选框是否可用"""
        is_bode_plot = len(self.canvas_analysis.fig.axes) == 2 and any(p.get('type') == 'bode' for p in self.plotted_traces)
        self.bode_tracker_check.setEnabled(is_bode_plot)
        if not is_bode_plot:
            self.bode_tracker_check.setChecked(False)

    def toggle_bode_tracker(self, state):
        """连接或断开鼠标移动事件来控制跟踪器的启停"""
        if state == Qt.Checked and self.bode_tracker_connection_id is None:
            self.bode_tracker_connection_id = self.canvas_analysis.mpl_connect(
                'motion_notify_event', self.on_bode_mouse_move)
        elif state == Qt.Unchecked and self.bode_tracker_connection_id is not None:
            self.canvas_analysis.mpl_disconnect(self.bode_tracker_connection_id)
            self.bode_tracker_connection_id = None
            self._clear_bode_tracker_annotations()
            self.canvas_analysis.draw()
    
    def _clear_bode_tracker_annotations(self):
        """清除画布上所有跟踪器留下的标注和线条"""
        for item in self.bode_tracker_annotations:
            try:
                item.remove()
            except Exception:
                pass
        self.bode_tracker_annotations = []

    def on_bode_mouse_move(self, event):
        """鼠标在Bode图上移动时的核心处理函数"""
        if event.inaxes is None or not self.canvas_analysis.fig.axes: return
        
        self._clear_bode_tracker_annotations()
        
        ax1, ax2 = self.canvas_analysis.fig.axes
        
        freq_mouse = event.xdata
        if freq_mouse is None: return
        
        vline1 = ax1.axvline(freq_mouse, color='grey', linestyle=':', lw=1)
        vline2 = ax2.axvline(freq_mouse, color='grey', linestyle=':', lw=1)
        self.bode_tracker_annotations.extend([vline1, vline2])

        for trace in self.plotted_traces:
            if trace.get('type') != 'bode': continue
            
            freq_data = trace['data']['freq']
            idx = np.searchsorted(freq_data, freq_mouse)
            if idx >= len(freq_data): idx = len(freq_data) - 1
            
            f = freq_data[idx]
            m = trace['data']['mag'][idx]
            p = trace['data']['phase'][idx]
            
            text = f"{trace['name']}\nFreq: {f:.2f} Hz\nGain: {m:.2f} dB\nPhase: {p:.2f}°"
            
            current_ax = event.inaxes 

            if current_ax == ax1:
                target_ax, anchor_xy = ax1, (f, m)
            elif current_ax == ax2:
                target_ax, anchor_xy = ax2, (f, p)
            else:
                target_ax, anchor_xy = ax1, (f, m)

            ann = target_ax.annotate(text, xy=anchor_xy, xytext=(15, 15), textcoords='offset points',
                                     bbox=dict(boxstyle="round,pad=0.4", fc=trace['lines'][0].get_color(), alpha=0.3),
                                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.1"),
                                     fontsize=15)

            hline1 = ax1.axhline(m, color=trace['lines'][0].get_color(), linestyle=':', lw=1, alpha=0.7)
            hline2 = ax2.axhline(p, color=trace['lines'][1].get_color(), linestyle=':', lw=1, alpha=0.7)
            self.bode_tracker_annotations.extend([ann, hline1, hline2])

        self.canvas_analysis.draw_idle()

    def _get_cursor_region_data(self):
        """获取当前光标选区内的数据，并进行有效性检查"""
        if not self.cursor_check.isChecked() or not self.x_cursor_check.isChecked():
            QMessageBox.warning(self, "操作无效", "请先启用光标并使用X轴光标选择一个区域。")
            return None, None, None
        
        y_cols = self.cursor_plot_data.get('y_cols', {})
        if len(y_cols) > 1:
            QMessageBox.warning(self, "操作无效", "自动测量功能一次仅支持分析一条曲线。")
            return None, None, None
        if not y_cols:
            return None, None, None
            
        x_full = self.cursor_plot_data['x']
        y_name, y_full = list(y_cols.items())[0]
        
        x1, x2 = sorted([self.v_cursor1.get_xdata()[0], self.v_cursor2.get_xdata()[0]])
        indices = np.where((x_full >= x1) & (x_full <= x2))
        
        x_region = x_full[indices]
        y_region = y_full[indices]
        
        if len(x_region) < 5:
            QMessageBox.warning(self, "数据不足", "光标选区内的数据点过少，无法进行精确测量。")
            return None, None, None
            
        return x_region, y_region, y_name

    def on_measure_rise_time(self):
        x, y, name = self._get_cursor_region_data()
        if x is None: return

        v_min, v_max = np.min(y), np.max(y)
        v_range = v_max - v_min
        if v_range < 1e-9:
             self.show_text(f"--- 上升时间测量 ---\n曲线: {name}\n测量失败：信号无变化。"); return

        v10 = v_min + 0.1 * v_range
        v90 = v_min + 0.9 * v_range
        
        try:
            t10 = np.interp(v10, y, x)
            t90 = np.interp(v90, y, x)
            rise_time = abs(t90 - t10)
            self.show_text(f"--- 上升时间测量 ---\n曲线: {name}\n10% 电平 (@{v10:.3f}) 时间: {t10:.4g} s\n90% 电平 (@{v90:.3f}) 时间: {t90:.4g} s\n\n上升时间 (10%-90%): {rise_time:.4g} s")
        except Exception as e:
            self.show_text(f"上升时间测量失败: {e}")

    def on_measure_fall_time(self):
        x, y, name = self._get_cursor_region_data()
        if x is None: return
        
        x_rev, y_rev = x[::-1], y[::-1]
        v_min, v_max = np.min(y), np.max(y)
        v_range = v_max - v_min
        if v_range < 1e-9:
             self.show_text(f"--- 下降时间测量 ---\n曲线: {name}\n测量失败：信号无变化。"); return

        v10 = v_min + 0.1 * v_range
        v90 = v_min + 0.9 * v_range

        try:
            t90 = np.interp(v90, y_rev, x_rev)
            t10 = np.interp(v10, y_rev, x_rev)
            fall_time = abs(t10 - t90)
            self.show_text(f"--- 下降时间测量 ---\n曲线: {name}\n90% 电平 (@{v90:.3f}) 时间: {t90:.4g} s\n10% 电平 (@{v10:.3f}) 时间: {t10:.4g} s\n\n下降时间 (90%-10%): {fall_time:.4g} s")
        except Exception as e:
            self.show_text(f"下降时间测量失败: {e}")

    def on_measure_frequency(self):
        x, y, name = self._get_cursor_region_data()
        if x is None: return
        
        mean_val = np.mean(y)
        indices = np.where(np.diff(np.sign(y - mean_val)))[0]
        
        if len(indices) < 2:
            self.show_text("--- 频率测量 ---\n测量失败：在选区内未找到足够的周期。"); return
            
        periods = np.diff(x[indices]) * 2
        avg_period = np.mean(periods)
        frequency = 1 / avg_period
        
        self.show_text(f"--- 频率/周期测量 ---\n曲线: {name}\n在选区内找到 {len(periods)} 个周期\n\n平均周期: {avg_period:.4g} s\n计算频率: {frequency:.4g} Hz")

    def clear_plot(self, show_message=True):
        self._clear_figure()
        self.plotted_traces = []
        self.canvas_analysis.draw()
        self.update_bode_tracker_visibility()
        if show_message:
            self.statusBar().showMessage("绘图区已清空")

    def show_oscilloscope_view(self):
        if not self.btn_show_osc.isEnabled(): return
        self.clear_plot(show_message=False)
        x_col, multi = self.x_axis_combo.currentText(), self.multi_column_check.isChecked()
        y_cols = [i.text() for i in self.multi_column_list.selectedItems()] if multi else [self.y_axis_combo.currentText()]
        
        df = self.align_numeric_df([x_col] + y_cols)
        if df.shape[0] < 2: 
            QMessageBox.warning(self, "警告", "对齐后有效数据不足，无法绘图。"); return

        y_cols_unique = [c for c in df.columns if c != x_col]
        if not y_cols_unique and x_col in df.columns:
             y_cols_unique = [x_col]

        x_data, y_data_dict = df[x_col].values, {col: df[col].values for col in y_cols_unique}
        ax = self._prepare_single_axis_plot(x_data, y_data_dict)
        self.cursor_plot_data = {'x': x_data, 'y_cols': y_data_dict}
        
        colors = self.plt.cm.tab20(np.linspace(0, 1, len(y_cols_unique)))
        step = self.sample_spin.value()
        
        for i, col in enumerate(y_cols_unique):
            line, = ax.plot(x_data[::step], y_data_dict[col][::step], label=col, color=colors[i])
            self.plotted_traces.append({'name': col, 'lines': [line], 'type': 'single'})
            
        ax.set_xlabel(x_col); ax.set_ylabel("值"); ax.legend(ncol=2, fontsize=9)
        self._finalize_plot(ax); self.statusBar().showMessage("已显示原始数据")
    
    # --- 文件处理 ---
    def open_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "选择多个CSV文件", "", "CSV文件 (*.csv);;所有文件 (*)")
        if paths: self.process_files(paths)
    
    def open_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "选择包含CSV文件的文件夹")
        if folder_path: self.process_files([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith('.csv')])
    
    def process_files(self, paths):
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            successful_loads, new_files_added = 0, False
            for path in paths:
                filename = os.path.basename(path)
                if filename not in self.loaded_files:
                    self.loaded_files[filename] = path
                    self.file_list.addItem(filename)
                    successful_loads += 1
                    new_files_added = True
            if successful_loads > 0:
                self.statusBar().showMessage(f"成功添加 {successful_loads} 个文件到列表")
                if new_files_added:
                    self.file_list.setCurrentRow(self.file_list.count() - 1)
            else:
                QMessageBox.information(self, "提示", "所有选择的文件均已在列表中。")
        finally:
            QApplication.restoreOverrideCursor()
    
    def switch_file(self, index):
        if index < 0 or self.file_list.count() == 0:
            self._clear_all_data()
            return
        filename = self.file_list.item(index).text()
        filepath = self.loaded_files.get(filename)
        if not filepath: return
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            self.data = self.safe_read_csv(filepath)
            self.current_file = filename
            self.update_combos(list(self.data.columns))
            self.reset_on_file_switch()
            self.statusBar().showMessage(f"已加载并切换到: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "文件加载失败", f"无法读取文件 '{filename}':\n{e}")
            self.loaded_files.pop(filename, None)
            self.file_list.takeItem(index)
            self.data = None
            self.current_file = None
        finally:
            QApplication.restoreOverrideCursor()
        self._update_action_buttons_state()
    
    def show_file_list_menu(self, pos: QPoint):
        from PyQt5.QtWidgets import QStyle
        if self.file_list.count() == 0: return
        menu = QMenu()
        menu.addAction(self.style().standardIcon(QStyle.SP_DialogCloseButton), "关闭选中文件", self.remove_selected_files)
        menu.addAction("关闭所有文件", self.remove_all_files)
        menu.exec_(self.file_list.mapToGlobal(pos))
    
    def remove_selected_files(self):
        selected_items = self.file_list.selectedItems()
        if not selected_items:
            current_item = self.file_list.currentItem()
            if current_item: selected_items = [current_item]
            else: return
        for item in selected_items:
            self.loaded_files.pop(item.text(), None)
            self.file_list.takeItem(self.file_list.row(item))
    
    def remove_all_files(self):
        self.loaded_files.clear()
        self.file_list.clear()
        self._clear_all_data()
    
    def _clear_all_data(self):
        self.data = None
        self.current_file = None
        self.update_combos([])
        if hasattr(self, 'result_stack'):
            self.result_stack.setCurrentIndex(self.RESULT_PLACEHOLDER_IDX)
        self.reset_on_file_switch()
        self.statusBar().showMessage("所有文件已关闭，准备就绪")
        self._update_action_buttons_state()

    def reset_on_file_switch(self):
        self.axis_control_group.setEnabled(False)
        self._disconnect_axis_limits_callback()
        self.full_x_limits, self.full_y_limits = None, None
        self.reset_cursor_state()
        self.clear_plot(show_message=False)
        self.show_text("请选择分析类型并执行分析。")
        self.btn_export.setEnabled(False)
    
    def safe_read_csv(self, path):
        try: return pd.read_csv(path, encoding='utf-8-sig', sep=None, engine='python')
        except:
            try: return pd.read_csv(path, encoding='gbk', sep=None, engine='python')
            except: return pd.read_csv(path)
    
    # --- UI状态与数据辅助方法 ---
    def _update_action_buttons_state(self):
        is_enabled = False
        if not hasattr(self, 'analysis_type'):
            if hasattr(self, 'btn_analyze'): self.btn_analyze.setEnabled(False)
            if hasattr(self, 'btn_show_osc'): self.btn_show_osc.setEnabled(False)
            return

        current_main_tab_text = self.main_tabs.tabText(self.main_tabs.currentIndex())
        
        if "理论模型设计" in current_main_tab_text:
             if hasattr(self, 'btn_analyze'): self.btn_analyze.setEnabled(False)
             if hasattr(self, 'btn_show_osc'): self.btn_show_osc.setEnabled(False)
             return

        analysis_mode = self.analysis_type.currentText()
        if analysis_mode == "传递函数分析 (Bode Plot)":
            if hasattr(self, 'tf_bode_num'):
                is_enabled = bool(self.tf_bode_num.text() and self.tf_bode_den.text())
        elif self.data is not None:
              x_selected = bool(self.x_axis_combo.currentText())
              y_selected = bool(self.multi_column_list.selectedItems()) if self.multi_column_check.isChecked() else bool(self.y_axis_combo.currentText())
              is_enabled = x_selected and y_selected

        if hasattr(self, 'btn_analyze'): self.btn_analyze.setEnabled(is_enabled)
        if hasattr(self, 'btn_show_osc'): self.btn_show_osc.setEnabled(self.data is not None and is_enabled)
    
    def update_transient_params_visibility(self):
        is_band = (self.transient_method_combo.currentText() == "带宽法")
        self.transient_band_label.setVisible(is_band)
        self.transient_band_spin.setVisible(is_band)
        self.transient_win_label.setVisible(not is_band)
        self.transient_thresh_label.setVisible(not is_band)
        self.transient_window_combo.setVisible(not is_band)
        self.transient_thresh_spin.setVisible(not is_band)

    def update_pll_params_visibility(self):
        is_band = (self.pll_method_combo.currentText() == "带宽法")
        self.pll_band_label.setVisible(is_band)
        self.pll_band_spin.setVisible(is_band)
        self.pll_win_label.setVisible(not is_band)
        self.pll_thresh_label.setVisible(not is_band)
        self.pll_window_combo.setVisible(not is_band)
        self.pll_thresh_spin.setVisible(not is_band)

    def on_multi_toggle(self):
        enabled = self.multi_column_check.isChecked()
        self.multi_column_list.setVisible(enabled)
        self.btn_select_all.setVisible(enabled)
        self.btn_clear_sel.setVisible(enabled)
        self.y_axis_combo.setEnabled(not enabled)
        self._update_action_buttons_state()

    def _guess_column_by_keywords(self, cols, keywords):
        for keyword in keywords:
            for col in cols:
                if keyword in col.lower():
                    return col
        return cols[0] if cols else ""

    def update_combos(self, cols):
        current_x = self.x_axis_combo.currentText()
        current_y = self.y_axis_combo.currentText()
        
        self.x_axis_combo.clear(); self.y_axis_combo.clear()
        self.bode_freq_combo.clear(); self.bode_gain_combo.clear(); self.bode_phase_combo.clear()
        
        if not cols: return
        
        for combo in [self.x_axis_combo, self.y_axis_combo, self.bode_freq_combo, self.bode_gain_combo, self.bode_phase_combo]:
            combo.addItems(cols)

        if current_x in cols: self.x_axis_combo.setCurrentText(current_x)
        else: self.x_axis_combo.setCurrentText(self._guess_column_by_keywords(cols, ['time', 't', '时间']))
        if current_y in cols: self.y_axis_combo.setCurrentText(current_y)
        
        self.bode_freq_combo.setCurrentText(self._guess_column_by_keywords(cols, ['freq', 'hz']))
        self.bode_gain_combo.setCurrentText(self._guess_column_by_keywords(cols, ['gain', 'db']))
        self.bode_phase_combo.setCurrentText(self._guess_column_by_keywords(cols, ['phase', 'deg']))

        self.refresh_multi_column_list()

    def refresh_multi_column_list(self):
        if self.data is None: return
        self.multi_column_list.clear()
        x_col = self.x_axis_combo.currentText()
        for col in self.data.columns:
            if col != x_col: self.multi_column_list.addItem(col)
    
    def align_numeric_df(self, columns):
        """
        对齐数据列，转换为数值型，并根据ROI设置进行筛选。
        """
        if self.data is None: return pd.DataFrame()
        
        df = self.data.copy()
        
        ### MODIFICATION START: More robust ROI filtering ###
        if self.roi_enable_check.isChecked():
            xmin = self.analysis_roi.get('xmin')
            xmax = self.analysis_roi.get('xmax')
            x_col = self.x_axis_combo.currentText()
            
            if x_col in df.columns and xmin is not None and xmax is not None and xmax > xmin:
                try:
                    x_numeric = pd.to_numeric(df[x_col], errors='coerce')
                    df = df[x_numeric.between(xmin, xmax)]
                except Exception as e:
                    print(f"Warning: Could not apply ROI filter. Error: {e}")
        ### MODIFICATION END ###

        unique_columns = list(dict.fromkeys(columns))
        
        valid_cols = [c for c in unique_columns if c in df.columns]
        if not valid_cols:
            return pd.DataFrame()

        df_subset = df[valid_cols].copy()
        for c in valid_cols:
            df_subset[c] = pd.to_numeric(df_subset[c], errors='coerce')
        
        return df_subset.dropna(how='any').astype(float)

    def subsample(self, arr, step): return arr[::step] if step > 1 else arr
    
    def show_table(self, df: pd.DataFrame):
        self.last_table_df = df.copy()
        self.result_stack.setCurrentIndex(self.RESULT_TABLE_IDX)
        self.result_table.clear()
        self.result_table.setRowCount(df.shape[0])
        self.result_table.setColumnCount(df.shape[1])
        self.result_table.setHorizontalHeaderLabels(list(df.columns))
        self.result_table.setVerticalHeaderLabels([str(i) for i in df.index])
        for r in range(df.shape[0]):
            for c in range(df.shape[1]):
                self.result_table.setItem(r, c, QTableWidgetItem(str(df.iat[r, c])))
        self.result_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.btn_export.setEnabled(True)

    def show_text(self, text):
        self.last_table_df = None
        self.result_stack.setCurrentIndex(self.RESULT_TEXT_IDX)
        self.result_text.setText(text)
        self.btn_export.setEnabled(False)

    def select_all_multi(self): self.multi_column_list.selectAll()
    def clear_multi_selection(self): self.multi_column_list.clearSelection()

    def _clear_figure(self):
        self.canvas_analysis.fig.clear()
        self.v_cursor1, self.v_cursor2, self.h_cursor1, self.h_cursor2 = None, None, None, None
        self.active_cursor = None
        self.cursor_plot_data.clear()

    def _prepare_single_axis_plot(self, x, y_data_dict):
        self.clear_plot(show_message=False)
        ax = self.canvas_analysis.fig.add_subplot(111)
        self._disconnect_axis_limits_callback()
        ax.grid(True, linestyle='--', color='#DCDCDC', alpha=0.7)
        y_values = [arr for arr in y_data_dict.values() if arr.size > 0]
        if y_values:
            y_min, y_max = (min(arr.min() for arr in y_values), max(arr.max() for arr in y_values))
            y_range = y_max - y_min if y_max > y_min else 1
            self.full_y_limits = (y_min - 0.05 * y_range, y_max + 0.05 * y_range)
        else: self.full_y_limits = (0, 1)
        self.full_x_limits = (x.min(), x.max())
        self.axis_limits_cid = ax.callbacks.connect('xlim_changed', self.on_axis_limits_changed)
        ax.callbacks.connect('ylim_changed', self.on_axis_limits_changed)
        self.axis_control_group.setEnabled(True)
        return ax

    def _finalize_plot(self, ax): self.reset_axis_limits(ax)
    
    def _disconnect_axis_limits_callback(self): 
        if self.axis_limits_cid and self.canvas_analysis.fig.axes:
            try: self.canvas_analysis.fig.axes[0].callbacks.disconnect(self.axis_limits_cid)
            except: pass
            self.axis_limits_cid = None
            
    def apply_manual_axis_limits(self):
        if not self.canvas_analysis.fig.axes: return
        ax = self.canvas_analysis.fig.axes[0]
        x_min, x_max = self.x_min_spin.value(), self.x_max_spin.value()
        y_min, y_max = self.y_min_spin.value(), self.y_max_spin.value()
        if x_min >= x_max or y_min >= y_max:
            QMessageBox.warning(self, "范围无效", "坐标轴的最小值必须小于最大值。")
            return
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        self.canvas_analysis.draw()

    def reset_axis_limits(self, ax): 
        if self.full_x_limits and self.full_y_limits:
            ax.set_xlim(self.full_x_limits)
            ax.set_ylim(self.full_y_limits)
            self.canvas_analysis.draw()

    def reset_axis_limits_view(self):
        if self.canvas_analysis.fig.axes:
            self.reset_axis_limits(self.canvas_analysis.fig.axes[0])
            
    def on_axis_limits_changed(self, ax):
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        self.x_min_spin.blockSignals(True); self.x_min_spin.setValue(x_lim[0]); self.x_min_spin.blockSignals(False)
        self.x_max_spin.blockSignals(True); self.x_max_spin.setValue(x_lim[1]); self.x_max_spin.blockSignals(False)
        self.y_min_spin.blockSignals(True); self.y_min_spin.setValue(y_lim[0]); self.y_min_spin.blockSignals(False)
        self.y_max_spin.blockSignals(True); self.y_max_spin.setValue(y_lim[1]); self.y_max_spin.blockSignals(False)

    # --- 光标逻辑 (未作修改) ---
    # ... (所有光标相关函数保持不变) ...
    # --- 光标逻辑 ---
    def reset_cursor_state(self):
        if self.cursor_events_connected: self.disconnect_cursor_events()
        self.cursor_check.blockSignals(True); self.cursor_check.setChecked(False); self.cursor_check.blockSignals(False)
        if hasattr(self,'cursor_group'): self.cursor_group.setVisible(False)
        
        if self.canvas_analysis.fig.axes:
            ax = self.canvas_analysis.fig.axes[0]
            for c in [self.v_cursor1, self.v_cursor2, self.h_cursor1, self.h_cursor2]:
                if c and c in ax.lines: c.set_visible(False)

        self.x_cursor_check.setEnabled(False)
        self.y_cursor_check.setEnabled(False)
        self.active_cursor=None
        
        if hasattr(self,'cursor_info_table'): self.cursor_info_table.clearContents()
        if hasattr(self,'cursor_stats_table'): 
            self.cursor_stats_table.clearContents()
            self.cursor_stats_table.setRowCount(0)
            self.cursor_stats_table.setColumnCount(0)
            
        if hasattr(self,'canvas_analysis'): self.canvas_analysis.draw_idle()

    def toggle_cursors(self, state):
        is_checked = (state == Qt.Checked)
        self.cursor_group.setVisible(is_checked)
        self.x_cursor_check.setEnabled(is_checked)
        self.y_cursor_check.setEnabled(is_checked)
        
        if is_checked:
            self.x_cursor_check.setChecked(True)
            self.y_cursor_check.setChecked(False)

            if not self.plotted_traces or not self.cursor_plot_data:
                QMessageBox.warning(self,"提示","请先绘制时域波形，再启用光标。")
                self.cursor_check.setChecked(False)
                return
            self.setup_cursors()
            if not self.cursor_events_connected: self.connect_cursor_events()
            self.analyze_cursor_region()
        else:
            if self.canvas_analysis.fig.axes:
                ax = self.canvas_analysis.fig.axes[0]
                for c in [self.v_cursor1, self.v_cursor2, self.h_cursor1, self.h_cursor2]:
                    if c and c in ax.lines: c.set_visible(False)
            if self.cursor_events_connected: self.disconnect_cursor_events()
            self.cursor_info_table.clearContents()
            self.cursor_stats_table.setRowCount(0)
            self.cursor_stats_table.setColumnCount(0)
            self.canvas_analysis.draw()

    def setup_cursors(self):
        if not self.canvas_analysis.fig.axes: return
        ax=self.canvas_analysis.fig.axes[0]
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()
        
        x1, x2 = x_min + 0.2 * (x_max - x_min), x_min + 0.8 * (x_max - x_min)
        y1, y2 = y_min + 0.2 * (y_max - y_min), y_min + 0.8 * (y_max - y_min)

        if self.v_cursor1 is None or self.v_cursor1 not in ax.lines: self.v_cursor1=ax.axvline(x1,c='r',lw=1.5,ls='--')
        else: self.v_cursor1.set_xdata([x1,x1])
        if self.v_cursor2 is None or self.v_cursor2 not in ax.lines: self.v_cursor2=ax.axvline(x2,c='r',lw=1.5,ls='--')
        else: self.v_cursor2.set_xdata([x2,x2])
        if self.h_cursor1 is None or self.h_cursor1 not in ax.lines: self.h_cursor1=ax.axhline(y1,c='b',lw=1.5,ls='--')
        else: self.h_cursor1.set_ydata([y1,y1])
        if self.h_cursor2 is None or self.h_cursor2 not in ax.lines: self.h_cursor2=ax.axhline(y2,c='b',lw=1.5,ls='--')
        else: self.h_cursor2.set_ydata([y2,y2])
        self.update_cursor_visibility()

    def update_cursor_visibility(self):
        if not self.cursor_check.isChecked(): return
        vis_x = self.x_cursor_check.isChecked()
        vis_y = self.y_cursor_check.isChecked()
        if self.v_cursor1: self.v_cursor1.set_visible(vis_x)
        if self.v_cursor2: self.v_cursor2.set_visible(vis_x)
        if self.h_cursor1: self.h_cursor1.set_visible(vis_y)
        if self.h_cursor2: self.h_cursor2.set_visible(vis_y)
        self.canvas_analysis.draw()
        self.analyze_cursor_region()
    
    def connect_cursor_events(self):
        self.cid_press=self.canvas_analysis.mpl_connect('button_press_event', self.on_cursor_press)
        self.cid_release=self.canvas_analysis.mpl_connect('button_release_event', self.on_cursor_release)
        self.cid_motion=self.canvas_analysis.mpl_connect('motion_notify_event', self.on_cursor_motion)
        self.cursor_events_connected=True
    
    def disconnect_cursor_events(self):
        if self.cursor_events_connected:
            self.canvas_analysis.mpl_disconnect(self.cid_press)
            self.canvas_analysis.mpl_disconnect(self.cid_release)
            self.canvas_analysis.mpl_disconnect(self.cid_motion)
            self.cursor_events_connected=False
    
    def on_cursor_press(self, event):
        if not event.inaxes or not self.cursor_check.isChecked(): return
        ax=event.inaxes
        xr, yr = ax.get_xlim()[1]-ax.get_xlim()[0], ax.get_ylim()[1]-ax.get_ylim()[0]
        if xr == 0 or yr == 0: return
        pick_tol, min_dist_px, self.active_cursor = 5, float('inf'), None
        
        if self.x_cursor_check.isChecked():
            for c in [self.v_cursor1, self.v_cursor2]:
                if not c or not c.get_visible(): continue
                dist = abs(event.xdata - c.get_xdata()[0]) * (self.canvas_analysis.width() / xr)
                if dist < pick_tol and dist < min_dist_px: self.active_cursor, min_dist_px = c, dist
        
        if self.y_cursor_check.isChecked():
            for c in [self.h_cursor1, self.h_cursor2]:
                if not c or not c.get_visible(): continue
                dist = abs(event.ydata - c.get_ydata()[0]) * (self.canvas_analysis.height() / yr)
                if dist < pick_tol and dist < min_dist_px: self.active_cursor, min_dist_px = c, dist
    
    def on_cursor_release(self, event):
        if self.active_cursor: self.active_cursor=None; self.analyze_cursor_region()
    
    def on_cursor_motion(self, event):
        if self.active_cursor is None or not event.inaxes: return
        if self.active_cursor in [self.v_cursor1, self.v_cursor2]:
            self.active_cursor.set_xdata([event.xdata, event.xdata])
        elif self.active_cursor in [self.h_cursor1, self.h_cursor2]:
            self.active_cursor.set_ydata([event.ydata, event.ydata])
        self.analyze_cursor_region(); self.canvas_analysis.draw_idle()
    
    def analyze_cursor_region(self):
        if not self.cursor_plot_data or not self.cursor_check.isChecked():
            self.cursor_info_table.clearContents()
            self.cursor_stats_table.setRowCount(0)
            return
            
        x_on, y_on = self.x_cursor_check.isChecked(), self.y_cursor_check.isChecked()
        x1, x2 = (self.v_cursor1.get_xdata()[0], self.v_cursor2.get_xdata()[0]) if x_on and self.v_cursor1 else (np.nan, np.nan)
        y1, y2 = (self.h_cursor1.get_ydata()[0], self.h_cursor2.get_ydata()[0]) if y_on and self.h_cursor1 else (np.nan, np.nan)
        
        x_min, x_max = (sorted([x1, x2]) if x_on else (float('-inf'), float('inf')))
        delta_x, delta_y = (x_max - x_min) if x_on else np.nan, abs(y1 - y2) if y_on else np.nan
        freq = (1 / delta_x) if x_on and delta_x != 0 else np.nan
        slope = (delta_y / delta_x) if x_on and y_on and delta_x != 0 else np.nan
        
        params = [("V-Cursor 1 (X1)", f"{x1:.6g}" if x_on else "N/A"), ("V-Cursor 2 (X2)", f"{x2:.6g}" if x_on else "N/A"),
                  ("H-Cursor 1 (Y1)", f"{y1:.6g}" if y_on else "N/A"), ("H-Cursor 2 (Y2)", f"{y2:.6g}" if y_on else "N/A"),
                  ("ΔX", f"{delta_x:.6g}" if x_on else "N/A"), ("ΔY", f"{delta_y:.6g}" if y_on else "N/A"),
                  ("Freq (1/ΔX)", f"{freq:.6g}" if x_on else "N/A"), ("Slope (ΔY/ΔX)", f"{slope:.6g}" if x_on and y_on else "N/A")]

        for col, (param, value) in enumerate(params):
            param_item = QTableWidgetItem(param); param_item.setFont(QFont("Microsoft YaHei", 9, QFont.Bold))
            self.cursor_info_table.setItem(0, col, param_item)
            self.cursor_info_table.setItem(1, col, QTableWidgetItem(value))
        
        full_x = self.cursor_plot_data.get('x', np.array([]))
        indices = np.where((full_x >= x_min) & (full_x <= x_max))[0] if x_on else np.arange(len(full_x))
        
        if len(indices) < 2 and x_on:
            self.cursor_stats_table.clear(); self.cursor_stats_table.setRowCount(1); self.cursor_stats_table.setColumnCount(1)
            self.cursor_stats_table.setHorizontalHeaderLabels([" "]); self.cursor_stats_table.setItem(0,0,QTableWidgetItem("选中X轴区域内数据点不足 (< 2)"))
            self.cursor_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        else:
            headers = ["曲线", "Max", "Min", "Vpp", "Mean", "Std", "RMS"]
            stats_data = []
            for name, full_y in self.cursor_plot_data.get('y_cols', {}).items():
                y_region = full_y[indices]
                if y_region.size == 0: continue
                y_max, y_min, y_mean = np.max(y_region), np.min(y_region), np.mean(y_region)
                stats_data.append([name, f"{y_max:.6g}", f"{y_min:.6g}", f"{y_max - y_min:.6g}", f"{y_mean:.6g}", f"{np.std(y_region):.6g}", f"{np.sqrt(np.mean(y_region**2)):.6g}"])
            
            self.cursor_stats_table.setRowCount(len(stats_data))
            self.cursor_stats_table.setColumnCount(len(headers))
            self.cursor_stats_table.setHorizontalHeaderLabels(headers)
            for r, row_data in enumerate(stats_data):
                for c, cell_data in enumerate(row_data):
                    self.cursor_stats_table.setItem(r, c, QTableWidgetItem(cell_data))
            self.cursor_stats_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

    # --- 导出与保存 ---
    def export_excel(self):
        if self.last_table_df is None or self.last_table_df.empty:
            QMessageBox.information(self, "提示", "暂无可导出的表格结果。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出为Excel", "分析结果.xlsx", "Excel文件 (*.xlsx)")
        if not path: return
        try:
            self.last_table_df.to_excel(path, index=True)
            self.statusBar().showMessage(f"已导出到: {path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"写文件失败：\n{e}")
    
    def save_figure(self):
        if not self.plotted_traces:
            QMessageBox.information(self, "提示", "当前没有可保存的图像。")
            return
        path, _ = QFileDialog.getSaveFileName(self, "保存图像为PNG", "分析图.png", "PNG图像 (*.png)")
        if not path: return
        try:
            self.canvas_analysis.fig.savefig(path, dpi=300, bbox_inches='tight', facecolor=self.canvas_analysis.fig.get_facecolor())
            self.statusBar().showMessage(f"图像已保存到: {path}")
        except Exception as e:
            QMessageBox.critical(self, "保存失败", f"保存图片失败：\n{e}")