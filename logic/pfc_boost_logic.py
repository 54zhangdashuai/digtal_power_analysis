# logic/pfc_boost_logic.py
import math
import numpy as np

def calculate_pfc_boost(params: dict) -> dict:
    """
    CCM Boost PFC 关键参数计算。
    返回一个包含计算结果和简化传递函数模型的字典。
    """
    try:
        vac_min, vac_max, f_line = params['vac_min'], params['vac_max'], params['f_line']
        vout, pout, fs = params['vout'], params['pout'], params['fs']
        efficiency, delta_il_pct = params['efficiency'], params['delta_il_pct']
        
        ### MODIFICATION START: Added input validation ###
        if not (0 < vac_min <= vac_max and f_line > 0 and pout > 0 and fs > 0 and 0 < efficiency <= 100):
            return {'error': "输入参数范围无效。请检查电压、频率、功率和效率。"}
        ### MODIFICATION END ###

        # --- 核心计算 ---
        pin = pout / (efficiency / 100.0)
        v_in_peak_min = vac_min * math.sqrt(2)
        v_in_peak_max = vac_max * math.sqrt(2) # Calculate for check
        
        if vout <= v_in_peak_max:
            return {'error': f"输出电压Vout({vout}V)必须高于最大峰值输入电压({v_in_peak_max:.1f}V)。"}
            
        i_in_peak_min = (math.sqrt(2) * pin) / vac_min
        d_max = 1 - (v_in_peak_min / vout)
        
        # 电感计算
        delta_il = i_in_peak_min * (delta_il_pct / 100.0)
        if delta_il <= 0:
            return {'error': "电感电流纹波必须大于0。"}
        l_val = (v_in_peak_min * d_max) / (delta_il * fs)
        
        # 输出电容计算 (基于二次线频纹波)
        # 假设纹波电压为输出的1%
        vout_ripple = vout * 0.01
        c_out_val = pout / (2 * math.pi * (2 * f_line) * vout * vout_ripple)
        
        # 器件应力
        v_stress_sw_diode = vout
        i_stress_sw_diode = i_in_peak_min
        
        # --- 传递函数建模 (简化) ---
        # PFC的精确模型很复杂。这里使用一个等效的二阶模型来演示功能。
        # Gvd(s) = (Vout/D') / (s^2*LC + s*L/R + 1)
        r_load = vout**2 / pout
        d_prime_avg = (1 - d_max) # 使用最大占空比进行估算
        if d_prime_avg <= 0:
             return {'error': "计算出的 D' 小于等于0，无法生成传递函数。"}
        dc_gain = vout / d_prime_avg
        
        num = [dc_gain]
        den = [l_val * c_out_val, l_val / r_load, 1]

        results = {
            "input_peak_current_A": i_in_peak_min,
            "max_duty_cycle": d_max,
            "inductor_uH": l_val * 1e6,
            "output_capacitor_uF": c_out_val * 1e6,
            "switch_stress_V": v_stress_sw_diode,
            "diode_stress_V": v_stress_sw_diode,
            "switch_peak_current_A": i_stress_sw_diode,
            "plant_tf_num": num,
            "plant_tf_den": den,
            "error": None
        }
        return results
        
    except (KeyError, ZeroDivisionError) as e:
        return {'error': f"输入参数错误或不完整: {e}"}