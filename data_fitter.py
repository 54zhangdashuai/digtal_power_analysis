# data_fitter.py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.signal import hilbert
from sklearn.metrics import r2_score, mean_squared_error

class DataFitter:
    """
    一个功能全面、易于扩展的数据拟合工具，
    特别擅长处理带有趋势的振荡信号。
    """
    def __init__(self, x_data, y_data):
        if not isinstance(x_data, np.ndarray) or not isinstance(y_data, np.ndarray):
            raise TypeError("x_data and y_data must be NumPy arrays.")
        if x_data.shape != y_data.shape:
            raise ValueError("x_data and y_data must have the same shape.")
        self.x_data = x_data
        self.y_data = y_data

    # --- 第一部分：核心框架与基础拟合功能 ---

    def fit_linear(self):
        """执行标准线性回归。"""
        # 新增：检查数据点是否足够
        if len(self.x_data) < 2:
            return {'error': '线性拟合需要至少2个数据点。', 'slope': np.nan, 'intercept': np.nan, 'r_squared': np.nan, 'coeffs': []}
            
        coeffs = np.polyfit(self.x_data, self.y_data, 1)
        slope, intercept = coeffs
        y_predicted = np.polyval(coeffs, self.x_data)
        r2 = r2_score(self.y_data, y_predicted)
        return {'slope': slope, 'intercept': intercept, 'r_squared': r2, 'coeffs': coeffs}

    def fit_polynomial(self, degree):
        """执行多项式回归。"""
        # 新增：检查数据点是否足够
        if len(self.x_data) < degree + 1:
            return {
                'error': f'进行{degree}阶多项式拟合至少需要 {degree + 1} 个数据点，当前只有 {len(self.x_data)} 个。',
                'coefficients': [], 'r_squared': np.nan, 'coeffs': []
            }

        coeffs = np.polyfit(self.x_data, self.y_data, degree)
        y_predicted = np.polyval(coeffs, self.x_data)
        r2 = r2_score(self.y_data, y_predicted)
        return {'coefficients': coeffs, 'r_squared': r2, 'coeffs': coeffs}

    def fit_custom_model(self, model_func, initial_guess):
        """使用scipy.optimize.curve_fit对用户自定义模型进行拟合。"""
        try:
            optimal_params, covariance_matrix = curve_fit(
                model_func, self.x_data, self.y_data, p0=initial_guess, maxfev=5000
            )
            return {'optimal_params': optimal_params, 'covariance_matrix': covariance_matrix}
        except RuntimeError:
            return {'error': 'Fit failed to converge with the given initial guess.'}

    # --- 第二部分：高级功能——趋势振荡信号分析 ---

    @staticmethod
    def _trending_oscillation_model(t, a, b, omega, phi, c, d):
        """复合模型函数定义"""
        return (a * t + b) * np.sin(omega * t + phi) + (c * t + d)

    def fit_trending_oscillation(self):
        """
        使用复合模型直接拟合带有趋势的振荡信号，并包含智能初始值猜测。
        """
        n = len(self.y_data)
        # 新增：检查数据点数量
        if n < 10:  # 对于复杂模型，需要更多的数据点
            return {'error': '趋势振荡拟合需要至少10个数据点。'}

        try:
            # 新增：更稳健的时间间隔dt计算
            dt = np.mean(np.diff(self.x_data)) if n > 1 else 0
            if not np.isfinite(dt) or dt <= 0:
                dt = (self.x_data[-1] - self.x_data[0]) / (n - 1) if n > 1 else 0
                if not np.isfinite(dt) or dt <= 0:
                        return {'error': '无法从X轴数据中确定有效的时间间隔(dt)。'}

            yf = np.fft.fft(self.y_data)
            xf = np.fft.fftfreq(n, dt)[:n//2]
            
            if len(xf) > 1:
                guess_omega = 2 * np.pi * xf[1 + np.argmax(np.abs(yf[1:n//2]))]
            else:
                guess_omega = 2 * np.pi / (self.x_data[-1] - self.x_data[0]) if (self.x_data[-1] - self.x_data[0]) > 0 else 1

            window_size = max(5, int(n / ( (self.x_data[-1]-self.x_data[0]) * guess_omega / (2 * np.pi)) / 2))
            if window_size % 2 == 0: window_size += 1
            baseline_trend = pd.Series(self.y_data).rolling(window=window_size, center=True, min_periods=1).mean().values
            
            valid_indices = ~np.isnan(baseline_trend)
            trend_coeffs = np.polyfit(self.x_data[valid_indices], baseline_trend[valid_indices], 1)
            guess_c, guess_d = trend_coeffs

            detrended_signal = self.y_data - (guess_c * self.x_data + guess_d)
            analytic_signal = hilbert(detrended_signal)
            amplitude_envelope = np.abs(analytic_signal)
            
            amp_coeffs = np.polyfit(self.x_data, amplitude_envelope, 1)
            guess_a, guess_b = amp_coeffs

            guess_phi = 0.0
            initial_guess = [guess_a, guess_b, guess_omega, guess_phi, guess_c, guess_d]

        except Exception:
            initial_guess = [0, np.std(self.y_data), 2*np.pi*len(self.x_data)/(self.x_data[-1]-self.x_data[0])/2, 0, 0, np.mean(self.y_data)]

        fit_result = self.fit_custom_model(self._trending_oscillation_model, initial_guess)
        
        if 'error' in fit_result:
            return fit_result
        
        p_opt = fit_result['optimal_params']
        param_names = ['a', 'b', 'omega', 'phi', 'c', 'd']
        return dict(zip(param_names, p_opt))

    def analyze_envelope(self, fit_method='linear', degree=1):
        """
        基于希尔伯特变换提取信号包络线，并对其趋势进行拟合。
        """
        analytic_signal = hilbert(self.y_data - np.mean(self.y_data))
        upper_envelope = np.abs(analytic_signal)
        
        envelope_fitter = DataFitter(self.x_data, upper_envelope)
        
        if fit_method.lower() == 'linear':
            fit_result = envelope_fitter.fit_linear()
        elif fit_method.lower() == 'polynomial':
            fit_result = envelope_fitter.fit_polynomial(degree)
        else:
            raise ValueError("fit_method must be 'linear' or 'polynomial'")
        
        # --- MODIFIED: 同时返回拟合结果和包络线数据 ---
        return fit_result, upper_envelope

    # --- 第三部分：评估与可视化 ---

    @staticmethod
    def calculate_goodness_of_fit(y_true, y_predicted):
        """计算并返回一系列拟合优度指标。"""
        if len(y_true) != len(y_predicted):
            return {'error': 'Input arrays must have the same length.'}
            
        residuals = y_true - y_predicted
        ss_res = np.sum(residuals**2)
        chi_squared = ss_res
        
        return {
            'r_squared': r2_score(y_true, y_predicted),
            'rmse': np.sqrt(mean_squared_error(y_true, y_predicted)),
            'chi_squared': chi_squared
        }