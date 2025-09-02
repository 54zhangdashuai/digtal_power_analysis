# logic/control_logic.py
import numpy as np
import scipy.signal as signal

def design_compensator(params: dict) -> dict:
    """设计一个数字PID补偿器 (简化版Lead补偿器)"""
    try:
        fc, pm_req = params['fc'], params['pm_req']
        plant_gain_db, plant_phase = params['plant_gain_db'], params['plant_phase']
        
        plant_gain_linear = 10**(plant_gain_db / 20.0)
        
        # 计算所需相位提升
        phase_boost = pm_req - (180 + plant_phase) + 5 # 5度安全裕度
        
        # Lead补偿器逻辑
        alpha = (1 + np.sin(np.deg2rad(phase_boost))) / (1 - np.sin(np.deg2rad(phase_boost)))
        wc = 2 * np.pi * fc
        
        wz = wc / np.sqrt(alpha)
        wp = wc * np.sqrt(alpha)
        
        # Gc(s) = K * (s/wz + 1) / (s/wp + 1)
        num_c_analog = [1/wz, 1]
        den_c_analog = [1/wp, 1]
        
        # 计算K值以保证穿越频率处的总增益为1 (0dB)
        _, mag_c, _ = signal.bode((num_c_analog, den_c_analog), w=[wc])
        k = 1 / (10**(mag_c[0]/20) * plant_gain_linear)
        
        num_c_analog = (np.array(num_c_analog) * k).tolist()
        
        comp_tf = signal.TransferFunction(num_c_analog, den_c_analog)
        
        b0, b1 = num_c_analog
        a0, a1 = den_c_analog # a0 is 1
        
        # 生成C代码 (浮点)
        c_code_float = (
            f"// Compensator Coefficients (Float)\n"
            f"// fc={fc} Hz, PM_req={pm_req} deg\n"
            f"float b0 = {b1:.6f};\n"
            f"float b1 = {b0:.6f};\n"
            f"float a1 = {a1:.6f};\n"
            f"// y[n] = b0*x[n] + b1*x[n-1] - a1*y[n-1]\n"
        )
        # 生成C代码 (Q15 - 示例)
        q_factor = 2**15
        c_code_q15 = (
            f"// Compensator Coefficients (Q15 format)\n"
            f"#define B0_Q15 ({int(b1 * q_factor)})\n"
            f"#define B1_Q15 ({int(b0 * q_factor)})\n"
            f"#define A1_Q15 ({int(a1 * q_factor)})\n"
        )

        return {
            "coeffs": {"b": num_c_analog, "a": den_c_analog},
            "c_code_float": c_code_float,
            "c_code_q15": c_code_q15,
            "compensator_tf": comp_tf,
            "error": None
        }
        
    except Exception as e:
        return {'error': f"补偿器计算失败: {e}"}