# analysis_logic.py

import numpy as np
import pandas as pd
import scipy.signal as signal
from PyQt5.QtWidgets import QMessageBox
from data_fitter import DataFitter

class AnalysisMixin:
    """
    一个包含所有核心分析计算方法的Mixin类。
    这些方法依赖于主窗口类提供的UI控件和状态变量。
    """
    def analyze_advanced_fitting(self):
        """高级数据拟合功能的总入口"""
        x_col, y_col = self.x_axis_combo.currentText(), self.y_axis_combo.currentText()
        if not x_col or not y_col:
            QMessageBox.warning(self, "选择错误", "请为X轴和Y轴选择有效的数据列。")
            return
            
        df = self.align_numeric_df([x_col, y_col]).sort_values(by=x_col)
        x_data, y_data = df[x_col].values, df[y_col].values
        
        fitter = DataFitter(x_data, y_data)
        fit_method = self.fitting_method_combo.currentText()
        
        results = None
        y_fit = None
        title = ""
        report = f"======= 高级数据拟合报告 =======\n\n"

        try:
            if "线性拟合" in fit_method:
                title = "线性拟合结果"
                results = fitter.fit_linear()
                y_fit = np.polyval(results['coeffs'], x_data)
                report += f"模型: y = mx + c\n"
                report += f"斜率 m: {results['slope']:.4g}\n"
                report += f"截距 c: {results['intercept']:.4g}\n"
                self.plot_fit_results(x_data, y_data, y_fit, title)

            elif "多项式拟合" in fit_method:
                degree = self.fitting_poly_degree.value()
                title = f"{degree}阶多项式拟合结果"
                results = fitter.fit_polynomial(degree)
                y_fit = np.polyval(results['coeffs'], x_data)
                report += f"模型: {degree}阶多项式\n"
                report += f"系数 (高阶到低阶): {[f'{c:.4g}' for c in results['coefficients']]}\n"
                self.plot_fit_results(x_data, y_data, y_fit, title)
            
            elif "趋势振荡" in fit_method:
                title = "趋势振荡信号拟合结果"
                results = fitter.fit_trending_oscillation()
                if 'error' in results: raise RuntimeError(results['error'])
                y_fit = fitter._trending_oscillation_model(x_data, **results)
                report += "模型: (at+b)·sin(ωt+φ) + (ct+d)\n"
                for key, val in results.items():
                    report += f"{key}: {val:.4g}\n"
                self.plot_fit_results(x_data, y_data, y_fit, title)
            
            elif "包络线分析" in fit_method:
                method = self.fitting_env_method.currentText()
                degree = self.fitting_env_poly_degree.value()
                title = f"包络线趋势分析 ({method})"
                
                results, upper_envelope = fitter.analyze_envelope(fit_method=method, degree=degree)
                
                y_fit = np.polyval(results['coeffs'], x_data)
                report += f"模型: 包络线 {method} 拟合\n"
                if method == 'linear':
                    report += f"斜率: {results['slope']:.4g}\n"
                    report += f"截距: {results['intercept']:.4g}\n"
                else:
                    report += f"系数: {[f'{c:.4g}' for c in results['coeffs']]}\n"

                # 特殊情况：绘图时，原始数据是包络线，不是原信号
                self.plot_fit_results(x_data, upper_envelope, y_fit, title)
                self.show_text(report) # 只显示报告，不添加gof
                self.statusBar().showMessage("包络线分析完成")
                return # 包络线分析流程结束

            # 对普通拟合计算拟合优度并添加到报告中
            gof = DataFitter.calculate_goodness_of_fit(y_data, y_fit)
            report += f"\n--- 拟合优度 ---\n"
            report += f"R² (决定系数): {gof['r_squared']:.4f}\n"
            report += f"RMSE (均方根误差): {gof['rmse']:.4g}\n"
            report += f"Chi-Squared (卡方值): {gof['chi_squared']:.4g}\n"
            
            self.show_text(report)
            self.statusBar().showMessage(f"{title} 完成")

        except Exception as e:
            QMessageBox.critical(self, "拟合失败", f"执行拟合时发生错误: \n{e}")
            self.statusBar().showMessage("拟合失败")

    def analyze_data(self):
        """主分析路由函数，由'执行分析/绘图'按钮触发"""
        if not self.btn_analyze.isEnabled(): return
        self.btn_analyze.setEnabled(False)
        self.statusBar().showMessage("正在分析中，请稍候...")
        self.QApplication.processEvents()
        
        mode = self.analysis_type.currentText()
        multi = self.multi_column_check.isChecked()
        
        try:
            # 所有从UI触发的新分析都应该从一个干净的画布开始
            self.clear_plot(show_message=False)

            if mode == "负载瞬态响应分析":
                self.analyze_transient_multi() if multi else self.analyze_transient_single()
            elif mode == "环路稳定性分析 (Bode Plot)":
                self.analyze_bode_plot()
            elif mode == "传递函数分析 (Bode Plot)":
                self.analyze_tf_bode_plot()
            elif mode == "FFT / 频域分析":
                self.analyze_fft()
            elif mode == "PLL性能分析":
                self.analyze_pll_multi() if multi else self.analyze_pll_single()
            elif mode == "ADC校准分析":
                self.analyze_adc_multi() if multi else self.analyze_adc_single()
            elif mode == "高级数据拟合 (Advanced Fitting)":
                self.analyze_advanced_fitting()              
            else:
                self.show_text(f"'{mode}' 功能待实现。")
        except Exception as e:
            QMessageBox.critical(self, "分析错误", f"分析过程中发生错误：\n{e}")
            self.statusBar().showMessage("分析失败")
        finally:
            self._update_action_buttons_state() # Re-enable buttons

    def analyze_tf_bode_plot(self, tf=None, name="传递函数", clear_plot=True):
        """根据传递函数对象或UI输入绘制Bode图"""
        try:
            if tf is None:
                num_str = self.tf_bode_num.text().replace(',', ' ')
                den_str = self.tf_bode_den.text().replace(',', ' ')
                num = [float(i) for i in num_str.split()]
                den = [float(i) for i in den_str.split()]
                if not num or not den:
                    QMessageBox.warning(self, "输入错误", "分子和分母系数不能为空。"); return
                tf = signal.TransferFunction(num, den)

            w, mag, phase = signal.bode(tf, n=2000)
            freq_hz = w / (2 * np.pi)

            self.plot_bode(freq_hz, mag, phase, name=name, clear_plot=clear_plot)
            self.statusBar().showMessage(f"已绘制'{name}'的Bode图")
        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"无法生成Bode图：\n{e}")
    
    def analyze_bode_plot(self):
        """分析从CSV加载的实测Bode图数据"""
        freq_col, gain_col, phase_col = self.bode_freq_combo.currentText(), self.bode_gain_combo.currentText(), self.bode_phase_combo.currentText()
        if not all([freq_col, gain_col, phase_col]):
            QMessageBox.warning(self, "列未选择", "请为频率、增益和相位选择有效的数据列。"); return
            
        try:
            df = self.align_numeric_df([freq_col, gain_col, phase_col])
            if df.empty or df.shape[0] < 2:
                QMessageBox.warning(self, "数据错误", "选择的数据列中包含非数值或空值。"); return
        except KeyError:
            QMessageBox.critical(self, "列名错误", f"一个或多个列名不存在。"); return
            
        freq, gain, phase = df[freq_col].values, df[gain_col].values, df[phase_col].values
        name = f"实测数据 ({self.current_file or ''})"
        self.plot_bode(freq, gain, phase, name=name, clear_plot=True)
        self.statusBar().showMessage("实测Bode图绘制完成")

    def analyze_fft(self):
        """执行FFT频域分析"""
        x_col = self.x_axis_combo.currentText()
        is_multi = self.multi_column_check.isChecked()
        y_cols = [i.text() for i in self.multi_column_list.selectedItems()] if is_multi else [self.y_axis_combo.currentText()]
        
        df = self.align_numeric_df([x_col] + y_cols)
        if df.shape[0] < 2: QMessageBox.warning(self, "数据不足", "有效数据点不足 (< 2)，无法执行FFT。"); return

        y_cols_unique = [c for c in df.columns if c != x_col]
        x_data, y_data_dict = df[x_col].values, {col: df[col].values for col in y_cols_unique}
        fs = 0
        
        if self.fft_auto_fs_check.isChecked():
            time_diffs = np.diff(x_data)
            if np.mean(time_diffs) == 0:
                 QMessageBox.critical(self, "错误", "时间列数据间隔为0，无法自动计算采样率。"); return
            if np.std(time_diffs) / np.mean(time_diffs) > 0.01:
                QMessageBox.warning(self, "采样率警告", "时间列间隔不均匀，采样率可能不准。")
            fs = 1.0 / np.mean(time_diffs)
            self.fft_fs_spin.setValue(fs)
        else:
            fs = self.fft_fs_spin.value()
        
        if fs <= 0: QMessageBox.critical(self, "错误", "采样率必须为正数。"); return

        ax = self._prepare_single_axis_plot(np.array([1, fs/2]), {'dummy': np.array([1,1])}) 
        ax.clear(); ax.grid(True,which="both",ls="--")
        
        all_peaks_data = []
        colors = self.plt.cm.tab20(np.linspace(0, 1, len(y_cols_unique)))
        
        for i, col in enumerate(y_cols_unique):
            y_values = y_data_dict[col]
            N = len(y_values)
            yf = np.fft.fft(y_values)
            xf = np.fft.fftfreq(N, 1 / fs)
            xf_single = xf[:N//2]
            yf_single = 2.0/N * np.abs(yf[0:N//2])
            line, = ax.plot(xf_single, yf_single, label=col, color=colors[i])
            self.plotted_traces.append({'name': col, 'lines': [line]})
            peak_indices = np.argsort(yf_single)[-10:][::-1]
            for idx in peak_indices:
                if xf_single[idx] > 0:
                    all_peaks_data.append([col, f"{xf_single[idx]:.2f}", f"{yf_single[idx]:.6g}"])

        peak_df = pd.DataFrame(all_peaks_data, columns=["曲线", "频率 (Hz)", "幅值"])
        self.show_table(peak_df)
        
        ax.set_xscale('log'); ax.set_title("FFT 频域分析"); ax.set_xlabel("频率 (Hz)"); ax.set_ylabel("幅值")
        ax.legend(); self.canvas_analysis.draw(); self.statusBar().showMessage("FFT分析完成")

    def _calculate_dynamic_response_metrics(self, x, y, band_pct, win_size, thresh, method):
        """计算动态响应指标的辅助函数"""
        res={'mean':np.mean(y),'std':np.std(y),'pp':np.max(y)-np.min(y),'steady_value':np.nan,
             'settle_time_band':np.nan,'settle_time_method':np.nan,
             'overshoot_pct':np.nan,'undershoot_pct':np.nan, 'settle_idx_final':None,
             'peak_val': np.nan, 'peak_idx': None, 'trough_val': np.nan, 'trough_idx': None}
        settle_idx_method=None
        if len(y) > win_size:
            if method=='标准差':
                settle_idx_method=next((i for i,val in enumerate(pd.Series(y).rolling(window=win_size).std()) if val<thresh),None)
            elif method=='RMSE':
                rolling_mean=pd.Series(y).rolling(window=win_size).mean().bfill()
                rolling_rmse=np.sqrt(((pd.Series(y)-rolling_mean)**2).rolling(window=win_size).mean())
                settle_idx_method=next((i for i,val in enumerate(rolling_rmse) if val<thresh),None)
        if settle_idx_method is not None: res['settle_time_method']=x[settle_idx_method]-x[0]
        
        steady_band_ref=np.mean(y[settle_idx_method:]) if settle_idx_method is not None and settle_idx_method<len(y)-2 else np.mean(y[int(len(y)*.9):])
        if steady_band_ref == 0: return res 

        lower_b,upper_b=steady_band_ref*(1-band_pct),steady_band_ref*(1+band_pct)
        outside_indices=np.where((y<lower_b)|(y>upper_b))[0]
        settle_idx_band=outside_indices[-1]+1 if outside_indices.size>0 else 0
        
        if settle_idx_band>=len(y): settle_idx_band=None
        if settle_idx_band is not None: res['settle_time_band']=x[settle_idx_band]-x[0]
        
        res['settle_idx_final']=settle_idx_method if settle_idx_method is not None and method != '带宽法' else settle_idx_band
        
        if res['settle_idx_final'] is not None and res['settle_idx_final']<len(y)-1:
            settle_idx=res['settle_idx_final']
            res['steady_value']=np.mean(y[settle_idx:])
            transient_region_y=y[:settle_idx]
            if len(transient_region_y)>0 and res['steady_value']!=0:
                res['peak_idx'] = np.argmax(transient_region_y)
                res['peak_val'] = transient_region_y[res['peak_idx']]
                res['trough_idx'] = np.argmin(transient_region_y)
                res['trough_val'] = transient_region_y[res['trough_idx']]
                overshoot = res['peak_val'] - res['steady_value']
                undershoot = res['steady_value'] - res['trough_val']
                if overshoot>0: res['overshoot_pct']=(overshoot/abs(res['steady_value']))*100
                if undershoot>0: res['undershoot_pct']=(undershoot/abs(res['steady_value']))*100
        return res

    def analyze_transient_single(self):
        """分析单列负载瞬态响应"""
        x_col, y_col = self.x_axis_combo.currentText(), self.y_axis_combo.currentText()
        try:
            win_size = int(self.transient_window_combo.currentText())
        except ValueError:
            QMessageBox.warning(self, "参数无效", "请输入有效的整数作为滑动窗口大小。"); return

        df = self.align_numeric_df([x_col, y_col])
        if df.shape[0] < win_size:
            QMessageBox.warning(self, "数据不足", f"有效数据点({df.shape[0]}个)少于滑动窗口大小({win_size})。"); return

        if x_col == y_col: x_data, y_data = df[x_col].values, df[x_col].values
        else: x_data, y_data = df[x_col].values, df[y_col].values
            
        self.cursor_plot_data = {'x': x_data, 'y_cols': {y_col: y_data}}
        params = {'band_pct': self.transient_band_spin.value() / 100.0, 'win_size': win_size,
                  'thresh': self.transient_thresh_spin.value(), 'method': self.transient_method_combo.currentText()}
        res = self._calculate_dynamic_response_metrics(x_data, y_data, **params)
        
        settle_time_val = res['settle_time_band'] if params['method'] == '带宽法' else res['settle_time_method']
        method_details = f"方法: 带宽法 (±{self.transient_band_spin.value():.1f}%)" if params['method'] == '带宽法' else f"方法: {params['method']} (阈值<{params['thresh']:.6f}, 窗={params['win_size']})"
        
        txt = (f"======= 负载瞬态响应分析 ({y_col}) =======\n\n"
               f"--- 核心结论 ---\n恢复时间: {settle_time_val:.6f} (X轴单位)\n({method_details})\n\n"
               f"--- 动态性能 ---\n稳态电压: {res['steady_value']:.6f}\n过冲: {res['overshoot_pct']:.2f} %\n下冲: {res['undershoot_pct']:.2f} %\n\n"
               f"--- 整体统计 ---\n均值: {res['mean']:.6f}\n标准差 (σ): {res['std']:.6f}\n峰峰值: {res['pp']:.6f}\n")
        self.show_text(txt.replace('nan', 'N/A'))
        
        ax = self._prepare_single_axis_plot(x_data, {y_col: y_data})
        line, = ax.plot(x_data[::self.sample_spin.value()], y_data[::self.sample_spin.value()], label=y_col, zorder=5)
        self.plotted_traces.append({'name': y_col, 'lines': [line], 'type': 'single'})
        
        ### MODIFICATION START: Add annotations to plot ###
        if res['settle_idx_final'] is not None and res['settle_idx_final'] < len(x_data):
            settle_x = x_data[res['settle_idx_final']]
            ax.axvspan(x_data[0], settle_x, color='orange', alpha=0.2, label='动态过渡区域')
            ax.axvspan(settle_x, x_data[-1], color='lightgreen', alpha=0.3, label='稳态区域')
            ax.axvline(settle_x, color='darkgreen', linestyle='--', linewidth=1.5, label=f'恢复时间点: {settle_time_val:.4f}')

            # Annotate peak and trough
            if res['peak_idx'] is not None:
                peak_x, peak_y = x_data[res['peak_idx']], res['peak_val']
                ax.plot(peak_x, peak_y, 'rv', markersize=8, label=f'过冲点 ({peak_y:.3f})')
            if res['trough_idx'] is not None:
                trough_x, trough_y = x_data[res['trough_idx']], res['trough_val']
                ax.plot(trough_x, trough_y, 'b^', markersize=8, label=f'下冲点 ({trough_y:.3f})')
        ### MODIFICATION END ###

        ax.set_title(f"负载瞬态响应 - {y_col}"); ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.legend()
        self._finalize_plot(ax); self.statusBar().showMessage("瞬态响应单列分析完成")

    def analyze_transient_multi(self):
        """分析多列负载瞬态响应"""
        # ... (与单列逻辑类似，但循环处理多列)
        self.show_text("多列瞬态分析待实现。")
    
    def analyze_pll_single(self):
        """分析单列PLL锁定过程"""
        x_col, y_col = self.x_axis_combo.currentText(), self.y_axis_combo.currentText()
        try:
            win_size = int(self.pll_window_combo.currentText())
        except ValueError:
            QMessageBox.warning(self, "参数无效", "请输入有效的整数作为滑动窗口大小。"); return

        df = self.align_numeric_df([x_col, y_col])
        if df.shape[0] < win_size:
            QMessageBox.warning(self, "数据不足", f"有效数据点少于滑动窗口大小。"); return

        if x_col == y_col: x_data, y_data = df[x_col].values, df[x_col].values
        else: x_data, y_data = df[x_col].values, df[y_col].values
            
        self.cursor_plot_data = {'x': x_data, 'y_cols': {y_col: y_data}}
        params = {'band_pct': self.pll_band_spin.value() / 100.0, 'win_size': win_size,
                  'thresh': self.pll_thresh_spin.value(), 'method': self.pll_method_combo.currentText()}
        res = self._calculate_dynamic_response_metrics(x_data, y_data, **params)
        
        settle_time_val = res['settle_time_band'] if params['method'] == '带宽法' else res['settle_time_method']
        method_details = f"方法: 带宽法 (±{self.pll_band_spin.value():.1f}%)" if params['method'] == '带宽法' else f"方法: {params['method']} (阈值<{params['thresh']:.6f}, 窗={params['win_size']})"
        
        txt = (f"======= PLL性能分析 ({y_col}) =======\n\n"
               f"--- 核心结论 ---\n锁定时间: {settle_time_val:.6f} (X轴单位)\n({method_details})\n\n"
               f"--- 动态性能 ---\n稳态值: {res['steady_value']:.6f}\n过冲: {res['overshoot_pct']:.2f} %\n下冲: {res['undershoot_pct']:.2f} %\n\n")
        self.show_text(txt.replace('nan', 'N/A'))
        
        ax = self._prepare_single_axis_plot(x_data, {y_col: y_data})
        line, = ax.plot(x_data[::self.sample_spin.value()], y_data[::self.sample_spin.value()], label=y_col)
        self.plotted_traces.append({'name': y_col, 'lines': [line], 'type': 'single'})

        if res['settle_idx_final'] is not None and res['settle_idx_final'] < len(x_data):
            settle_x = x_data[res['settle_idx_final']]
            ax.axvspan(x_data[0], settle_x, color='orange', alpha=0.2, label='动态过渡区域')
            ax.axvspan(settle_x, x_data[-1], color='lightgreen', alpha=0.3, label='稳态(锁定)区域')
            ax.axvline(settle_x, color='darkgreen', linestyle='--', linewidth=1.5, label=f'锁定时间点: {settle_time_val:.4f}')

        ax.set_title(f"PLL性能分析 - {y_col}"); ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.legend()
        self._finalize_plot(ax); self.statusBar().showMessage("PLL单列分析完成")

    def analyze_pll_multi(self):
        """分析多列PLL锁定过程"""
        x_col, y_cols = self.x_axis_combo.currentText(), [i.text() for i in self.multi_column_list.selectedItems()]
        try:
            win_size = int(self.pll_window_combo.currentText())
        except ValueError:
            QMessageBox.warning(self, "参数无效", "请输入有效的整数作为滑动窗口大小。"); return

        df = self.align_numeric_df([x_col] + y_cols)
        if df.shape[0] < win_size:
            QMessageBox.warning(self, "数据不足", f"有效数据点少于滑动窗口大小。"); return
        
        y_cols_unique = [c for c in df.columns if c != x_col]
        x_data, y_data_dict = df[x_col].values, {col: df[col].values for col in y_cols_unique}
        ax = self._prepare_single_axis_plot(x_data, y_data_dict)
        self.cursor_plot_data = {'x': x_data, 'y_cols': {}}

        colors = self.plt.cm.tab20(np.linspace(0, 1, len(y_cols_unique)))
        rows, final_settle_x, step = [], [], self.sample_spin.value()
        params = {'band_pct': self.pll_band_spin.value()/100.0, 'win_size': win_size,
                  'thresh': self.pll_thresh_spin.value(), 'method': self.pll_method_combo.currentText()}
        settle_time_header = f"锁定时间({params['method']})"

        for i, col in enumerate(y_cols_unique):
            y_values = y_data_dict[col]
            self.cursor_plot_data['y_cols'][col] = y_values
            res = self._calculate_dynamic_response_metrics(x_data, y_values, **params)
            settle_time = res['settle_time_band'] if params['method'] == '带宽法' else res['settle_time_method']
            rows.append([col, f"{res['mean']:.6f}", f"{res['std']:.6f}", f"{res['pp']:.6f}", f"{res['steady_value']:.6f}", f"{res['overshoot_pct']:.2f}%", f"{res['undershoot_pct']:.2f}%", f"{settle_time:.6f}"])
            line, = ax.plot(x_data[::step], y_values[::step], label=col, color=colors[i])
            self.plotted_traces.append({'name': col, 'lines': [line], 'type': 'single'})
            if res['settle_idx_final'] is not None and res['settle_idx_final'] < len(x_data):
                final_settle_x.append(x_data[res['settle_idx_final']])
        
        if final_settle_x:
            latest_settle_x = max(final_settle_x)
            ax.axvspan(x_data[0], latest_settle_x, color='orange', alpha=0.2, label='最晚动态过渡区域')
            ax.axvspan(latest_settle_x, x_data[-1], color='lightgreen', alpha=0.3, label='最晚稳态(锁定)区域')
            ax.axvline(latest_settle_x, color='darkgreen', linestyle='--', linewidth=1.5, label=f'最晚锁定时间点: {latest_settle_x - x_data[0]:.4f}')

        cols = ["列名", "均值", "标准差", "峰峰值", "稳态值", "过冲(%)", "下冲(%)", settle_time_header]
        self.show_table(pd.DataFrame(rows, columns=cols).replace({'nan%': 'N/A', 'nan': 'N/A'}))
        ax.set_title("PLL性能分析 - 多列"); ax.set_xlabel(x_col); ax.legend(ncol=2, fontsize=9)
        self._finalize_plot(ax); self.statusBar().showMessage("PLL多列分析完成")
        
        ### MODIFICATION START: Removed duplicated code block ###
        # The entire logic block for analyze_pll_multi was duplicated. It has been removed.
        ### MODIFICATION END ###

    def analyze_adc_single(self):
        """分析ADC的INL/DNL"""
        x_col, y_col = self.x_axis_combo.currentText(), self.y_axis_combo.currentText()
        df = self.align_numeric_df([x_col, y_col]).sort_values(by=x_col)
        
        if df.shape[0] < 3:
            QMessageBox.warning(self, "数据不足", "ADC分析需要至少3个有效数据点。"); return

        x_ideal_raw = df[x_col].values
        y_code_raw = df[y_col].values
        
        # 1. 多项式拟合
        degree = self.adc_fit_method_combo.currentIndex() + 1
        coeffs = np.polyfit(x_ideal_raw, y_code_raw, degree)
        p_fit = np.poly1d(coeffs)
        y_fit = p_fit(x_ideal_raw)
        
        # 2. 计算INL (相对理想直线)
        # 理想直线定义为通过首尾两个端点的线
        ideal_line_coeffs = np.polyfit([x_ideal_raw[0], x_ideal_raw[-1]], [y_code_raw[0], y_code_raw[-1]], 1)
        p_ideal_line = np.poly1d(ideal_line_coeffs)
        y_ideal_line = p_ideal_line(x_ideal_raw)
        inl = (y_code_raw - y_ideal_line) # INL in LSB
        inl_max, inl_min = np.max(inl), np.min(inl)
        
        ### MODIFICATION START: More robust DNL calculation ###
        # 3. 计算DNL, 处理缺失码 (Missing Codes)
        unique_codes = sorted(df[y_col].unique())
        code_centers = df.groupby(y_col)[x_col].mean().loc[unique_codes].values
        
        dnl_max, dnl_min = np.nan, np.nan
        dnl_codes, dnl_values = [], []

        if len(code_centers) > 1:
            # 理想步宽 (LSB voltage)
            total_codes_span = unique_codes[-1] - unique_codes[0]
            total_voltage_span = x_ideal_raw[-1] - x_ideal_raw[0]
            ideal_step = total_voltage_span / total_codes_span if total_codes_span > 0 else 0

            if ideal_step > 0:
                step_widths = np.diff(code_centers)
                code_jumps = np.diff(unique_codes)
                
                # DNL = (实际码宽 / 理想码宽) - 1
                # 实际码宽 = 电压差 / code跳变数
                actual_code_widths = step_widths / code_jumps
                dnl_values = (actual_code_widths / ideal_step) - 1
                dnl_codes = unique_codes[1:]
                
                if len(dnl_values) > 0:
                    dnl_max, dnl_min = np.max(dnl_values), np.min(dnl_values)
        ### MODIFICATION END ###

        # 4. 显示文本结果
        fit_eq = self._format_poly_equation(coeffs)
        txt = (f"======= ADC校准分析 ({y_col} vs {x_col}) =======\n\n"
               f"--- 拟合结果 ({degree}阶多项式) ---\n{fit_eq}\n\n"
               f"--- 静态非线性误差 ---\n"
               f"积分非线性 (INL): +{inl_max:.3f} / {inl_min:.3f} LSB\n"
               f"微分非线性 (DNL): +{dnl_max:.3f} / {dnl_min:.3f} LSB\n")
        self.show_text(txt.replace('nan', 'N/A'))
        
        # 5. 绘图
        self._clear_figure()
        ax1 = self.canvas_analysis.fig.add_subplot(2, 2, 1)
        ax2 = self.canvas_analysis.fig.add_subplot(2, 2, 2)
        ax3 = self.canvas_analysis.fig.add_subplot(2, 1, 2)
        
        # 图1: 传递函数
        ax1.plot(x_ideal_raw, y_code_raw, 'o', markersize=3, label='实测数据')
        ax1.plot(x_ideal_raw, y_fit, '-', lw=2, label=f'{degree}阶拟合曲线')
        ax1.set_title("ADC 传递函数"); ax1.set_xlabel("理想输入"); ax1.set_ylabel("ADC输出码值"); ax1.legend(); ax1.grid(True)
        
        # 图2: INL
        ax2.plot(y_code_raw, inl)
        ax2.set_title(f"INL (+{inl_max:.2f} / {inl_min:.2f} LSB)"); ax2.set_xlabel("ADC输出码值"); ax2.set_ylabel("INL (LSB)"); ax2.grid(True)
        
        # 图3: DNL
        if dnl_codes and len(dnl_values) > 0:
            ax3.plot(dnl_codes, dnl_values)
        ax3.set_title(f"DNL (+{dnl_max:.2f} / {dnl_min:.2f} LSB)"); ax3.set_xlabel("ADC输出码值"); ax3.set_ylabel("DNL (LSB)"); ax3.grid(True)
        
        self.canvas_analysis.fig.tight_layout()
        self.canvas_analysis.draw()
        self.statusBar().showMessage("ADC分析完成")
    
    def analyze_adc_multi(self):
        self.show_text("多列ADC分析功能待实现。")
    
    def _format_poly_equation(self,p):
        equation="y = ";degree=len(p)-1
        for i,coeff in enumerate(p):
            power=degree-i
            if abs(coeff) < 1e-9: continue
            if i > 0: equation += " + " if coeff >= 0 else " - "; coeff = abs(coeff)
            elif coeff < 0: equation += "- "; coeff = abs(coeff)
            
            if power > 1: equation += f"{coeff:.4g}·x^{power}"
            elif power == 1: equation += f"{coeff:.4g}·x"
            else: equation += f"{coeff:.4g}"
        return equation