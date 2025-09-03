# analysis_logic.py

# analysis_logic.py

import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.optimize import least_squares
from PyQt5.QtWidgets import QMessageBox
from data_fitter import DataFitter

class AnalysisMixin:
    """
    一个包含所有核心分析计算方法的Mixin类。
    """
    ### MODIFICATION START: Add auto-detection algorithm ###
    def _guess_transfer_function_order(self, freq, mag, phase):
        """
        根据伯德图数据自动猜测传递函数的阶数。
        返回 (guessed_zeros, guessed_poles)。
        """
        try:
            # 使用展开的相位以避免-180/180跳变问题
            unwrapped_phase = np.unwrap(np.deg2rad(phase)) * 180 / np.pi
            
            # 1. 根据总相位偏移估算(极点数 - 零点数)
            # 对于稳定的最小相位系统，总相位变化约为 (Z-P)*90度
            # 我们从0度开始，所以 P-Z ≈ (start_phase - end_phase) / 90
            total_phase_shift = unwrapped_phase[0] - unwrapped_phase[-1]
            pole_zero_difference = int(round(total_phase_shift / 90.0))
            
            # 2. 根据相位的局部最小值(谷)来检测零点
            # 零点会提供相位裕度提升，导致相位曲线出现一个谷底然后回升
            # 我们寻找负的峰值，也就是谷
            # prominence确保谷底是显著的
            phase_valleys, _ = signal.find_peaks(-unwrapped_phase, prominence=10) # 10度的显著性
            
            guessed_zeros = 0
            if len(phase_valleys) > 0:
                for valley_idx in phase_valleys:
                    # 检查谷底之后是否有显著的相位回升
                    if valley_idx < len(unwrapped_phase) - 1:
                        phase_after_valley = unwrapped_phase[valley_idx:]
                        # 如果从谷底到最后的相位回升超过30度，我们认为这是一个零点
                        if (phase_after_valley[-1] - phase_after_valley[0]) > 30:
                            guessed_zeros += 1

            # 3. 结合信息计算最终阶数
            guessed_poles = guessed_zeros + pole_zero_difference

            # 4. 合理性检查 (根据数字电源常见模型进行约束)
            if guessed_poles < 1: guessed_poles = 1
            if guessed_poles > 4: guessed_poles = 4 # 限制最大阶数
            if guessed_zeros > 2: guessed_zeros = 2 # 限制最大零点数
            if guessed_zeros >= guessed_poles: guessed_zeros = guessed_poles - 1

            return max(0, guessed_zeros), guessed_poles

        except Exception as e:
            print(f"自动检测阶数失败: {e}")
            # 返回一个安全的默认值
            return 0, 2
    ### MODIFICATION END ###


    def _calculate_bode_metrics(self, freq, mag, phase):
        metrics = {'pm': np.nan, 'gm': np.nan, 'wc_gc': None, 'wc_pc': None, 'bw': None, 'mr': np.nan}
        
        # 新增：如果数据点过少，直接返回空结果，避免后续计算出错
        if len(freq) < 2:
            print("Warning: Not enough data points to calculate Bode metrics.")
            return metrics
            
        try:
            if np.any(mag < 0) and np.any(mag > 0):
                crossings = np.where(np.diff(np.sign(mag)))[0]
                if len(crossings) > 0:
                    idx = crossings[0]
                    f1, f2, m1, m2 = freq[idx], freq[idx+1], mag[idx], mag[idx+1]
                    metrics['wc_gc'] = f1 * (f2/f1)**(-m1/(m2-m1))
                    phase_at_gc = np.interp(metrics['wc_gc'], freq, phase)
                    metrics['pm'] = 180 + phase_at_gc
            
            phase_shifted_pos = np.unwrap(np.deg2rad(phase)) * 180 / np.pi
            phase_cross_val = -180
            
            crossings = np.where(np.diff(np.sign(phase_shifted_pos - phase_cross_val)))[0]
            
            if len(crossings) > 0:
                idx = crossings[0]
                p1, p2 = phase_shifted_pos[idx], phase_shifted_pos[idx+1]
                f1, f2 = freq[idx], freq[idx+1]
                
                metrics['wc_pc'] = f1 + (phase_cross_val - p1) * (f2 - f1) / (p2 - p1)
                mag_at_pc = np.interp(metrics['wc_pc'], freq, mag)
                metrics['gm'] = -mag_at_pc

            dc_gain = mag[0]
            if np.any(mag < (dc_gain - 3)):
                bw_candidates = freq[np.where(mag < (dc_gain-3))[0]]
                if len(bw_candidates)>0: metrics['bw'] = bw_candidates[0]
            if np.max(mag) > dc_gain: metrics['mr'] = np.max(mag)
        except Exception as e: 
            print(f"无法计算所有伯德图指标: {e}")
        return metrics

    ### MODIFICATION START: Fitter upgraded to a multi-model dispatcher ###
    def _fit_transfer_function(self, freq, mag, phase, num_zeros, num_poles):
        """
        使用最小二乘法将传递函数拟合到频率响应数据。
        v3.0: 升级为多模型调度器，可根据零极点数选择最优拟合策略。
        - 支持 2P0Z (纯二阶谐振)
        - 支持 2P1Z (二阶谐振 + ESR零点)
        - 支持 3P0Z (积分器 + 二阶谐振)
        - 其他情况回退到通用实数极点拟合器
        """
        w = 2 * np.pi * freq
        mag_linear = 10**(mag / 20.0)
        y_measured_complex = mag_linear * np.exp(1j * np.deg2rad(phase))

        # --- 模型 1: 纯二阶系统 (2 Poles, 0 Zeros) ---
        if num_poles == 2 and num_zeros == 0:
            return self._fit_sos_model(w, freq, mag, mag_linear, y_measured_complex)

        # --- 模型 2: 二阶系统 + 1个零点 (2 Poles, 1 Zero) ---
        elif num_poles == 2 and num_zeros == 1:
            return self._fit_2p1z_model(w, freq, mag, phase, mag_linear, y_measured_complex)

        # --- 模型 3: 积分器 + 二阶系统 (3 Poles, 0 Zeros) ---
        # 假设其中一个极点是积分器
        elif num_poles == 3 and num_zeros == 0:
            return self._fit_integrator_sos_model(w, freq, mag, mag_linear, y_measured_complex)
            
        # --- 回退模型: 通用实数零极点拟合器 ---
        else:
            return self._fit_generic_real_poles_model(w, y_measured_complex, num_zeros, num_poles)

    def _fit_sos_model(self, w, freq, mag, mag_linear, y_measured_complex):
        """拟合 G(s) = k * wn^2 / (s^2 + 2*zeta*wn*s + wn^2)"""
        def model(p, w_eval):
            k, wn, zeta = abs(p[0]), abs(p[1]), abs(p[2])
            num = [k * wn**2]
            den = [1, 2 * zeta * wn, wn**2]
            _w, h = signal.freqs(num, den, worN=w_eval)
            return h

        def error_func(p):
            error = model(p, w) - y_measured_complex
            return np.concatenate([np.real(error), np.imag(error)])
        
        # 初始值猜测
        guess_k = mag_linear[0]
        peak_idx = np.argmax(mag)
        guess_wn = 2 * np.pi * freq[peak_idx]
        Mr_db = mag[peak_idx] - mag[0]
        guess_zeta = 1 / (2 * 10**(Mr_db / 20.0)) if Mr_db > 0.1 else 0.7
        p0 = [guess_k, guess_wn, min(0.99, guess_zeta)]
        
        res = least_squares(error_func, p0, method='lm', max_nfev=20000)
        if not res.success: return None, "优化失败 (SOS)"
        
        k_opt, wn_opt, zeta_opt = abs(res.x[0]), abs(res.x[1]), abs(res.x[2])
        num_fit = [k_opt * wn_opt**2]; den_fit = [1, 2 * zeta_opt * wn_opt, wn_opt**2]
        return signal.TransferFunction(num_fit, den_fit), None

    def _fit_2p1z_model(self, w, freq, mag, phase, mag_linear, y_measured_complex):
        """拟合 G(s) = k * (s/wz + 1) / (s^2/wn^2 + ... + 1)"""
        def model(p, w_eval):
            k, wn, zeta, wz = abs(p[0]), abs(p[1]), abs(p[2]), abs(p[3])
            num = [k/wz, k]
            den = [1/(wn**2), 2*zeta/wn, 1]
            _w, h = signal.freqs(num, den, worN=w_eval)
            return h

        def error_func(p):
            error = model(p, w) - y_measured_complex
            return np.concatenate([np.real(error), np.imag(error)])

        # 初始值猜测
        guess_k = mag_linear[0]
        peak_idx = np.argmax(mag)
        guess_wn = 2 * np.pi * freq[peak_idx]
        Mr_db = mag[peak_idx] - mag[0]
        guess_zeta = 1 / (2 * 10**(Mr_db / 20.0)) if Mr_db > 0.1 else 0.7
        # 从相位回升处猜测零点频率
        phase_min_idx = np.argmin(phase)
        guess_wz = 2 * np.pi * freq[phase_min_idx] * 2 # 零点通常在相位最低点之后
        p0 = [guess_k, guess_wn, min(0.99, guess_zeta), guess_wz]

        res = least_squares(error_func, p0, method='lm', max_nfev=20000)
        if not res.success: return None, "优化失败 (2P1Z)"

        k_opt, wn_opt, zeta_opt, wz_opt = abs(res.x[0]), abs(res.x[1]), abs(res.x[2]), abs(res.x[3])
        num_fit = [k_opt/wz_opt, k_opt]
        den_fit = [1/(wn_opt**2), 2*zeta_opt/wn_opt, 1]
        return signal.TransferFunction(num_fit, den_fit), None

    def _fit_integrator_sos_model(self, w, freq, mag, mag_linear, y_measured_complex):
        """拟合 G(s) = k / (s * (s^2/wn^2 + 2*zeta*s/wn + 1))"""
        def model(p, w_eval):
            k, wn, zeta = abs(p[0]), abs(p[1]), abs(p[2])
            den_sos = [1/(wn**2), 2*zeta/wn, 1]
            den = np.convolve(den_sos, [1, 0]) # 乘以 's'
            num = [k]
            _w, h = signal.freqs(num, den, worN=w_eval)
            return h
        
        def error_func(p):
            error = model(p, w) - y_measured_complex
            return np.concatenate([np.real(error), np.imag(error)])

        # 初始值猜测
        # 从低频段 k_approx = mag_linear * w 估算 k
        low_freq_mask = w < w[len(w)//10]
        if np.any(low_freq_mask):
            guess_k = np.mean(mag_linear[low_freq_mask] * w[low_freq_mask])
        else:
            guess_k = mag_linear[0]*w[0]
        # 移除积分效应后找谐振峰
        mag_comp = mag + 20*np.log10(freq) 
        peak_idx = np.argmax(mag_comp)
        guess_wn = 2 * np.pi * freq[peak_idx]
        Mr_db = mag_comp[peak_idx] - np.mean(mag_comp[:5])
        guess_zeta = 1 / (2 * 10**(Mr_db / 20.0)) if Mr_db > 0.1 else 0.7
        p0 = [guess_k, guess_wn, min(0.99, guess_zeta)]
        
        res = least_squares(error_func, p0, method='lm', max_nfev=20000)
        if not res.success: return None, "优化失败 (INT+SOS)"

        k_opt, wn_opt, zeta_opt = abs(res.x[0]), abs(res.x[1]), abs(res.x[2])
        den_sos = [1/(wn_opt**2), 2*zeta_opt/wn_opt, 1]
        den_fit = np.convolve(den_sos, [1, 0])
        num_fit = [k_opt]
        return signal.TransferFunction(num_fit, den_fit), None

    def _fit_generic_real_poles_model(self, w, y_measured_complex, num_zeros, num_poles):
        """通用模型，仅支持实数零极点，无法拟合谐振峰"""
        def model(p, w_eval):
            k = p[0]
            zeros_re = p[1 : 1 + num_zeros] if num_zeros > 0 else []
            poles_re = p[1 + num_zeros :]
            num = np.poly(zeros_re) * k if num_zeros > 0 else [k]
            den = np.poly(poles_re)
            _w, h = signal.freqs(num, den, worN=w_eval)
            return h
        
        def error_func(p):
            error = model(p, w) - y_measured_complex
            return np.concatenate([np.real(error), np.imag(error)])
        
        p0 = [np.abs(y_measured_complex[0])]
        if num_zeros > 0: p0.extend([-1e4] * num_zeros)
        p0.extend([-1e3] * num_poles)

        res = least_squares(error_func, p0, method='lm', max_nfev=20000)
        if not res.success:
            return None, "优化失败。注意：通用模型不支持复数极点，无法拟合谐振峰。"
        
        p_opt = res.x
        k_opt = p_opt[0]
        zeros_opt = p_opt[1:1+num_zeros] if num_zeros > 0 else []
        poles_opt = p_opt[1+num_zeros:]
        num_fit = np.poly(zeros_opt) * k_opt if num_zeros > 0 else [k_opt]
        den_fit = np.poly(poles_opt)
        return signal.TransferFunction(num_fit, den_fit), None
    ### MODIFICATION END ###

    def analyze_data(self):
        if not self.btn_analyze.isEnabled(): return
        self.btn_analyze.setEnabled(False)
        self.statusBar().showMessage("正在分析中，请稍候...")
        self.QApplication.processEvents()
        mode = self.analysis_type.currentText()
        multi = self.multi_column_check.isChecked()
        try:
            if "Bode Plot" not in mode: self.clear_plot(show_message=False)
            if mode == "负载瞬态响应分析": self.analyze_transient_multi() if multi else self.analyze_transient_single()
            elif mode == "环路稳定性分析 (Bode Plot)": self.analyze_bode_plot()
            elif mode == "传递函数分析 (Bode Plot)": self.analyze_tf_bode_plot()
            elif mode == "FFT / 频域分析": self.analyze_fft()
            elif mode == "PLL性能分析": self.analyze_pll_multi() if multi else self.analyze_pll_single()
            elif mode == "ADC校准分析": self.analyze_adc_multi() if multi else self.analyze_adc_single()
            elif mode == "高级数据拟合 (Advanced Fitting)": self.analyze_advanced_fitting()
            else: self.show_text(f"'{mode}' 功能待实现。")
        except Exception as e:
            QMessageBox.critical(self, "分析错误", f"分析过程中发生错误：\n{e}")
            self.statusBar().showMessage("分析失败")
        finally: 
            self._update_action_buttons_state()

    # ... (文件其余部分代码保持不变) ...
    # ... (为了简洁省略，请只替换上面修改的部分) ...
    # ... (The rest of the file remains unchanged) ...
    
    def analyze_tf_bode_plot(self, tf=None, name="传递函数", clear_plot=True):
        try:
            if tf is None:
                num = [float(i) for i in self.tf_bode_num.text().replace(',', ' ').split()]
                den = [float(i) for i in self.tf_bode_den.text().replace(',', ' ').split()]
                if not num or not den: 
                    QMessageBox.warning(self, "输入错误", "分子和分母系数不能为空。"); return
                tf = signal.TransferFunction(num, den)

            w, mag, phase = signal.bode(tf, n=2000)
            freq_hz = w / (2 * np.pi)
            metrics = self._calculate_bode_metrics(freq_hz, mag, phase)
            self._report_bode_metrics(metrics, title=f"理论传递函数分析: {name}")
            self.plot_bode(freq_hz, mag, phase, name=name, clear_plot=clear_plot, metrics=metrics)
            self.statusBar().showMessage(f"已绘制'{name}'的Bode图并完成分析")
        except Exception as e:
            QMessageBox.critical(self, "计算错误", f"无法生成Bode图：\n{e}")

    def analyze_bode_plot(self):
        """主分析函数，集成了自动检测逻辑。"""
        # 1. 获取数据 (重用上一版已优化的数据获取逻辑)
        mode = self.bode_col_select_mode.currentText()
        df = self.data
        if df is None or df.empty: 
            QMessageBox.warning(self, "无数据", "请先加载一个数据文件。"); return
        try:
            freq_col_name = ""
            if mode == "按名称":
                freq_col, gain_col, phase_col = self.bode_freq_combo.currentText(), self.bode_gain_combo.currentText(), self.bode_phase_combo.currentText()
                if not all([freq_col, gain_col, phase_col]) or any(c not in df.columns for c in [freq_col, gain_col, phase_col]):
                    QMessageBox.warning(self, "列选择错误", "请为频率、增益和相位选择有效的列名。"); return
                df_aligned = self.align_numeric_df([freq_col, gain_col, phase_col]).sort_values(by=freq_col)
                freq_col_name = freq_col
            else: # 按位置
                f_idx, g_idx, p_idx = self.bode_freq_combo.currentIndex(), self.bode_gain_combo.currentIndex(), self.bode_phase_combo.currentIndex()
                if any(i < 0 for i in [f_idx, g_idx, p_idx]) or max(f_idx, g_idx, p_idx) >= len(df.columns):
                    QMessageBox.warning(self, "列位置错误", "选择的列位置超出了数据范围。"); return
                selected_cols = list(df.columns[[f_idx, g_idx, p_idx]])
                freq_col_name = selected_cols[0]
                df_aligned = self.align_numeric_df(selected_cols).sort_values(by=freq_col_name)

            df_aligned = df_aligned[df_aligned[freq_col_name] > 0]
            if df_aligned.shape[0] < 5:
                QMessageBox.warning(self, "数据不足", "有效数据点过少(少于5个)，无法进行分析。"); return

            freq, gain, phase = df_aligned.iloc[:,0].values, df_aligned.iloc[:,1].values, df_aligned.iloc[:,2].values
        except Exception as e:
            QMessageBox.critical(self, "数据处理错误", f"提取伯德图数据时出错: {e}"); return
            
        # 2. 绘制原始伯德图
        self.clear_plot(show_message=False)
        metrics = self._calculate_bode_metrics(freq, gain, phase)
        self._report_bode_metrics(metrics, title=f"实测数据分析: {self.current_file}")
        name = f"实测数据 ({self.current_file or ''})"
        self.plot_bode(freq, gain, phase, name=name, clear_plot=False, metrics=metrics)
        
        # 3. 如果勾选了拟合，则调用拟合函数 (该函数内部会处理是否自动检测)
        if self.fitting_group.isChecked():
            self._perform_tf_fit(freq, gain, phase)
        else:
            self.statusBar().showMessage("Bode图分析完成")


    def _report_bode_metrics(self, metrics, title="伯德图分析报告"):
        report = f"======= {title} =======\n\n"
        gc_freq_str = f"{metrics['wc_gc']:.2f} Hz" if metrics['wc_gc'] else "未找到"
        pc_freq_str = f"{metrics['wc_pc']:.2f} Hz" if metrics['wc_pc'] else "未找到"
        pm_str = f"{metrics['pm']:.2f} deg" if not np.isnan(metrics['pm']) else "N/A"
        gm_str = f"{metrics['gm']:.2f} dB" if not np.isnan(metrics['gm']) else "N/A (稳定)"
        bw_str = f"{metrics['bw']:.2f} Hz" if metrics['bw'] else "N/A"
        mr_str = f"{metrics['mr']:.2f} dB" if not np.isnan(metrics['mr']) else "无谐振峰"
        
        report += f"增益穿越频率 (ω_gc): {gc_freq_str}\n"
        report += f"相位裕度 (PM): {pm_str}\n\n"
        report += f"相位穿越频率 (ω_pc): {pc_freq_str}\n"
        report += f"增益裕度 (GM): {gm_str}\n\n"
        report += f"带宽 (-3dB): {bw_str}\n"
        report += f"谐振峰值 (Mr): {mr_str}\n"
        
        self.show_text(report)

    def analyze_advanced_fitting(self):
        x_col, y_col = self.x_axis_combo.currentText(), self.y_axis_combo.currentText()
        if not x_col or not y_col:
            QMessageBox.warning(self, "选择错误", "请为X轴和Y轴选择有效的数据列。"); return
            
        df = self.align_numeric_df([x_col, y_col]).sort_values(by=x_col)
        x_data, y_data = df[x_col].values, df[y_col].values
        
        fitter = DataFitter(x_data, y_data)
        fit_method = self.fitting_method_combo.currentText()
        
        results, y_fit, title, report = None, None, "", f"======= 高级数据拟合报告 =======\n\n"

        try:
            if "线性拟合" in fit_method:
                title = "线性拟合结果"; results = fitter.fit_linear()
                y_fit = np.polyval(results['coeffs'], x_data)
                report += f"模型: y = mx + c\n斜率 m: {results['slope']:.4g}\n截距 c: {results['intercept']:.4g}\n"
                self.plot_fit_results(x_data, y_data, y_fit, title)

            elif "多项式拟合" in fit_method:
                degree = self.fitting_poly_degree.value()
                title = f"{degree}阶多项式拟合结果"; results = fitter.fit_polynomial(degree)
                y_fit = np.polyval(results['coeffs'], x_data)
                report += f"模型: {degree}阶多项式\n系数: {[f'{c:.4g}' for c in results['coefficients']]}\n"
                self.plot_fit_results(x_data, y_data, y_fit, title)
            
            elif "趋势振荡" in fit_method:
                title = "趋势振荡信号拟合结果"; results = fitter.fit_trending_oscillation()
                if 'error' in results: raise RuntimeError(results['error'])
                y_fit = fitter._trending_oscillation_model(x_data, **results)
                report += "模型: (at+b)·sin(ωt+φ) + (ct+d)\n"
                for key, val in results.items(): report += f"{key}: {val:.4g}\n"
                self.plot_fit_results(x_data, y_data, y_fit, title)
            
            elif "包络线分析" in fit_method:
                method, degree = self.fitting_env_method.currentText(), self.fitting_env_poly_degree.value()
                title = f"包络线趋势分析 ({method})"
                results, upper_envelope = fitter.analyze_envelope(fit_method=method, degree=degree)
                y_fit = np.polyval(results['coeffs'], x_data)
                report += f"模型: 包络线 {method} 拟合\n"
                if method == 'linear': report += f"斜率: {results['slope']:.4g}\n截距: {results['intercept']:.4g}\n"
                else: report += f"系数: {[f'{c:.4g}' for c in results['coeffs']]}\n"
                self.plot_fit_results(x_data, upper_envelope, y_fit, title)
                self.show_text(report); self.statusBar().showMessage("包络线分析完成"); return

            gof = DataFitter.calculate_goodness_of_fit(y_data, y_fit)
            report += f"\n--- 拟合优度 ---\nR²: {gof['r_squared']:.4f}\nRMSE: {gof['rmse']:.4g}\nChi-Squared: {gof['chi_squared']:.4g}\n"
            
            self.show_text(report); self.statusBar().showMessage(f"{title} 完成")
        except Exception as e:
            QMessageBox.critical(self, "拟合失败", f"执行拟合时发生错误: \n{e}"); self.statusBar().showMessage("拟合失败")

    def analyze_fft(self):
        x_col, is_multi = self.x_axis_combo.currentText(), self.multi_column_check.isChecked()
        y_cols = [i.text() for i in self.multi_column_list.selectedItems()] if is_multi else [self.y_axis_combo.currentText()]
        df = self.align_numeric_df([x_col] + y_cols)
        if df.shape[0] < 2: 
            QMessageBox.warning(self, "数据不足", "有效数据点不足 (< 2)，无法执行FFT。"); return
            
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
        
        if fs <= 0: 
            QMessageBox.critical(self, "错误", "采样率必须为正数。"); return

        ax = self._prepare_single_axis_plot(np.array([1, fs/2]), {'dummy': np.array([1,1])})
        ax.clear(); ax.grid(True,which="both",ls="--")
        
        all_peaks_data, colors = [], self.plt.cm.tab20(np.linspace(0, 1, len(y_cols_unique)))
        
        for i, col in enumerate(y_cols_unique):
            y_values = y_data_dict[col]
            N = len(y_values)
            yf, xf = np.fft.fft(y_values), np.fft.fftfreq(N, 1 / fs)
            xf_single, yf_single = xf[:N//2], 2.0/N * np.abs(yf[0:N//2])
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
        res={'mean':np.mean(y),'std':np.std(y),'pp':np.max(y)-np.min(y),'steady_value':np.nan,'settle_time_band':np.nan,'settle_time_method':np.nan,'overshoot_pct':np.nan,'undershoot_pct':np.nan, 'settle_idx_final':None,'peak_val': np.nan, 'peak_idx': None, 'trough_val': np.nan, 'trough_idx': None}
        settle_idx_method=None
        if len(y) > win_size:
            if method=='标准差': settle_idx_method=next((i for i,val in enumerate(pd.Series(y).rolling(window=win_size).std()) if val<thresh),None)
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
                res['peak_idx'] = np.argmax(transient_region_y); res['peak_val'] = transient_region_y[res['peak_idx']]
                res['trough_idx'] = np.argmin(transient_region_y); res['trough_val'] = transient_region_y[res['trough_idx']]
                overshoot, undershoot = res['peak_val'] - res['steady_value'], res['steady_value'] - res['trough_val']
                if overshoot>0: res['overshoot_pct']=(overshoot/abs(res['steady_value']))*100
                if undershoot>0: res['undershoot_pct']=(undershoot/abs(res['steady_value']))*100
        return res

    def analyze_transient_single(self):
        x_col, y_col = self.x_axis_combo.currentText(), self.y_axis_combo.currentText()
        try: win_size = int(self.transient_window_combo.currentText())
        except ValueError: QMessageBox.warning(self, "参数无效", "请输入有效的整数作为滑动窗口大小。"); return
        df = self.align_numeric_df([x_col, y_col])
        if df.shape[0] < win_size: QMessageBox.warning(self, "数据不足", f"有效数据点({df.shape[0]}个)少于滑动窗口大小({win_size})。"); return
        x_data, y_data = df[x_col].values, df[y_col].values
        self.cursor_plot_data = {'x': x_data, 'y_cols': {y_col: y_data}}
        params = {'band_pct': self.transient_band_spin.value() / 100.0, 'win_size': win_size, 'thresh': self.transient_thresh_spin.value(), 'method': self.transient_method_combo.currentText()}
        res = self._calculate_dynamic_response_metrics(x_data, y_data, **params)
        settle_time_val = res['settle_time_band'] if params['method'] == '带宽法' else res['settle_time_method']
        method_details = f"方法: 带宽法 (±{self.transient_band_spin.value():.1f}%)" if params['method'] == '带宽法' else f"方法: {params['method']} (阈值<{params['thresh']:.6f}, 窗={params['win_size']})"
        txt = (f"======= 负载瞬态响应分析 ({y_col}) =======\n\n--- 核心结论 ---\n恢复时间: {settle_time_val:.6f} (X轴单位)\n({method_details})\n\n--- 动态性能 ---\n稳态电压: {res['steady_value']:.6f}\n过冲: {res['overshoot_pct']:.2f} %\n下冲: {res['undershoot_pct']:.2f} %\n\n--- 整体统计 ---\n均值: {res['mean']:.6f}\n标准差 (σ): {res['std']:.6f}\n峰峰值: {res['pp']:.6f}\n")
        self.show_text(txt.replace('nan', 'N/A'))
        ax = self._prepare_single_axis_plot(x_data, {y_col: y_data})
        line, = ax.plot(x_data[::self.sample_spin.value()], y_data[::self.sample_spin.value()], label=y_col, zorder=5)
        self.plotted_traces.append({'name': y_col, 'lines': [line], 'type': 'single'})
        if res['settle_idx_final'] is not None and res['settle_idx_final'] < len(x_data):
            settle_x = x_data[res['settle_idx_final']]
            ax.axvspan(x_data[0], settle_x, color='orange', alpha=0.2, label='动态过渡区域')
            ax.axvspan(settle_x, x_data[-1], color='lightgreen', alpha=0.3, label='稳态区域')
            ax.axvline(settle_x, color='darkgreen', linestyle='--', linewidth=1.5, label=f'恢复时间点: {settle_time_val:.4f}')
            if res['peak_idx'] is not None:
                peak_x, peak_y = x_data[res['peak_idx']], res['peak_val']
                ax.plot(peak_x, peak_y, 'rv', markersize=8, label=f'过冲点 ({peak_y:.3f})')
            if res['trough_idx'] is not None:
                trough_x, trough_y = x_data[res['trough_idx']], res['trough_val']
                ax.plot(trough_x, trough_y, 'b^', markersize=8, label=f'下冲点 ({trough_y:.3f})')
        ax.set_title(f"负载瞬态响应 - {y_col}"); ax.set_xlabel(x_col); ax.set_ylabel(y_col); ax.legend()
        self._finalize_plot(ax); self.statusBar().showMessage("瞬态响应单列分析完成")

    def analyze_transient_multi(self):
        x_col, y_cols = self.x_axis_combo.currentText(), [i.text() for i in self.multi_column_list.selectedItems()]
        try: win_size = int(self.transient_window_combo.currentText())
        except ValueError: QMessageBox.warning(self, "参数无效", "请输入有效的整数作为滑动窗口大小。"); return
        df = self.align_numeric_df([x_col] + y_cols)
        if df.shape[0] < win_size: QMessageBox.warning(self, "数据不足", f"有效数据点少于滑动窗口大小。"); return
        
        y_cols_unique = [c for c in df.columns if c != x_col]
        x_data, y_data_dict = df[x_col].values, {col: df[col].values for col in y_cols_unique}
        ax = self._prepare_single_axis_plot(x_data, y_data_dict)
        self.cursor_plot_data = {'x': x_data, 'y_cols': {}}

        colors = self.plt.cm.tab20(np.linspace(0, 1, len(y_cols_unique)))
        rows, final_settle_x, step = [], [], self.sample_spin.value()
        params = {'band_pct': self.transient_band_spin.value()/100.0, 'win_size': win_size, 'thresh': self.transient_thresh_spin.value(), 'method': self.transient_method_combo.currentText()}
        settle_time_header = f"恢复时间({params['method']})"

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
            ax.axvspan(latest_settle_x, x_data[-1], color='lightgreen', alpha=0.3, label='最晚稳态区域')
            ax.axvline(latest_settle_x, color='darkgreen', linestyle='--', linewidth=1.5, label=f'最晚恢复时间点: {latest_settle_x - x_data[0]:.4f}')

        cols = ["列名", "均值", "标准差", "峰峰值", "稳态值", "过冲(%)", "下冲(%)", settle_time_header]
        self.show_table(pd.DataFrame(rows, columns=cols).replace({'nan%': 'N/A', 'nan': 'N/A'}))
        ax.set_title("负载瞬态响应 - 多列"); ax.set_xlabel(x_col); ax.legend(ncol=2, fontsize=9)
        self._finalize_plot(ax); self.statusBar().showMessage("瞬态响应多列分析完成")
        
    def analyze_pll_single(self):
        x_col, y_col = self.x_axis_combo.currentText(), self.y_axis_combo.currentText()
        try: win_size = int(self.pll_window_combo.currentText())
        except ValueError: QMessageBox.warning(self, "参数无效", "请输入有效的整数作为滑动窗口大小。"); return
        df = self.align_numeric_df([x_col, y_col])
        if df.shape[0] < win_size: QMessageBox.warning(self, "数据不足", f"有效数据点少于滑动窗口大小。"); return
        x_data, y_data = df[x_col].values, df[y_col].values
        self.cursor_plot_data = {'x': x_data, 'y_cols': {y_col: y_data}}
        params = {'band_pct': self.pll_band_spin.value() / 100.0, 'win_size': win_size, 'thresh': self.pll_thresh_spin.value(), 'method': self.pll_method_combo.currentText()}
        res = self._calculate_dynamic_response_metrics(x_data, y_data, **params)
        settle_time_val = res['settle_time_band'] if params['method'] == '带宽法' else res['settle_time_method']
        method_details = f"方法: 带宽法 (±{self.pll_band_spin.value():.1f}%)" if params['method'] == '带宽法' else f"方法: {params['method']} (阈值<{params['thresh']:.6f}, 窗={params['win_size']})"
        txt = (f"======= PLL性能分析 ({y_col}) =======\n\n--- 核心结论 ---\n锁定时间: {settle_time_val:.6f} (X轴单位)\n({method_details})\n\n--- 动态性能 ---\n稳态值: {res['steady_value']:.6f}\n过冲: {res['overshoot_pct']:.2f} %\n下冲: {res['undershoot_pct']:.2f} %\n\n")
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
        x_col, y_cols = self.x_axis_combo.currentText(), [i.text() for i in self.multi_column_list.selectedItems()]
        try: win_size = int(self.pll_window_combo.currentText())
        except ValueError: QMessageBox.warning(self, "参数无效", "请输入有效的整数作为滑动窗口大小。"); return
        df = self.align_numeric_df([x_col] + y_cols)
        if df.shape[0] < win_size: QMessageBox.warning(self, "数据不足", f"有效数据点少于滑动窗口大小。"); return
        y_cols_unique = [c for c in df.columns if c != x_col]
        x_data, y_data_dict = df[x_col].values, {col: df[col].values for col in y_cols_unique}
        ax = self._prepare_single_axis_plot(x_data, y_data_dict)
        self.cursor_plot_data = {'x': x_data, 'y_cols': {}}
        colors = self.plt.cm.tab20(np.linspace(0, 1, len(y_cols_unique)))
        rows, final_settle_x, step = [], [], self.sample_spin.value()
        params = {'band_pct': self.pll_band_spin.value()/100.0, 'win_size': win_size, 'thresh': self.pll_thresh_spin.value(), 'method': self.pll_method_combo.currentText()}
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

    def analyze_adc_single(self):
        x_col, y_col = self.x_axis_combo.currentText(), self.y_axis_combo.currentText()
        df = self.align_numeric_df([x_col, y_col]).sort_values(by=x_col)
        if df.shape[0] < 3: QMessageBox.warning(self, "数据不足", "ADC分析需要至少3个有效数据点。"); return
        x_ideal_raw, y_code_raw = df[x_col].values, df[y_col].values
        degree = self.adc_fit_method_combo.currentIndex() + 1
        coeffs = np.polyfit(x_ideal_raw, y_code_raw, degree)
        y_fit = np.poly1d(coeffs)(x_ideal_raw)
        ideal_line_coeffs = np.polyfit([x_ideal_raw[0], x_ideal_raw[-1]], [y_code_raw[0], y_code_raw[-1]], 1)
        y_ideal_line = np.poly1d(ideal_line_coeffs)(x_ideal_raw)
        inl = (y_code_raw - y_ideal_line)
        inl_max, inl_min = np.max(inl), np.min(inl)
        unique_codes = sorted(df[y_col].unique())
        code_centers = df.groupby(y_col)[x_col].mean().loc[unique_codes].values
        dnl_max, dnl_min, dnl_codes, dnl_values = np.nan, np.nan, [], []
        if len(code_centers) > 1:
            total_codes_span = unique_codes[-1] - unique_codes[0]
            total_voltage_span = x_ideal_raw[-1] - x_ideal_raw[0]
            ideal_step = total_voltage_span / total_codes_span if total_codes_span > 0 else 0
            if ideal_step > 0:
                actual_code_widths = np.diff(code_centers) / np.diff(unique_codes)
                dnl_values = (actual_code_widths / ideal_step) - 1
                dnl_codes = unique_codes[1:]
                if len(dnl_values) > 0: dnl_max, dnl_min = np.max(dnl_values), np.min(dnl_values)
        fit_eq = self._format_poly_equation(coeffs)
        txt = (f"======= ADC校准分析 ({y_col} vs {x_col}) =======\n\n--- 拟合结果 ({degree}阶多项式) ---\n{fit_eq}\n\n"
               f"--- 静态非线性误差 ---\n积分非线性 (INL): +{inl_max:.3f} / {inl_min:.3f} LSB\n"
               f"微分非线性 (DNL): +{dnl_max:.3f} / {dnl_min:.3f} LSB\n")
        self.show_text(txt.replace('nan', 'N/A'))
        self._clear_figure()
        ax1 = self.canvas_analysis.fig.add_subplot(2, 2, 1); ax2 = self.canvas_analysis.fig.add_subplot(2, 2, 2); ax3 = self.canvas_analysis.fig.add_subplot(2, 1, 2)
        ax1.plot(x_ideal_raw, y_code_raw, 'o', markersize=3, label='实测数据'); ax1.plot(x_ideal_raw, y_fit, '-', lw=2, label=f'{degree}阶拟合曲线')
        ax1.set_title("ADC 传递函数"); ax1.set_xlabel("理想输入"); ax1.set_ylabel("ADC输出码值"); ax1.legend(); ax1.grid(True)
        ax2.plot(y_code_raw, inl); ax2.set_title(f"INL (+{inl_max:.2f} / {inl_min:.2f} LSB)"); ax2.set_xlabel("ADC输出码值"); ax2.set_ylabel("INL (LSB)"); ax2.grid(True)
        if dnl_codes and len(dnl_values) > 0: ax3.plot(dnl_codes, dnl_values)
        ax3.set_title(f"DNL (+{dnl_max:.2f} / {dnl_min:.2f} LSB)"); ax3.set_xlabel("ADC输出码值"); ax3.set_ylabel("DNL (LSB)"); ax3.grid(True)
        self.canvas_analysis.fig.tight_layout(); self.canvas_analysis.draw(); self.statusBar().showMessage("ADC分析完成")

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