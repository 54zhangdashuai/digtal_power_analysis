# logic/buck_logic.py
import math

def calculate_buck(params: dict) -> dict:
    """
    Buck变换器的简化计算。
    接收一个包含设计参数的字典，返回一个包含计算结果的字典。
    """
    try:
        vin, vout, iout = params['vin'], params['vout'], params['iout']
        fs, delta_il_pct, delta_vout_pct = params['fs'], params['delta_il_pct'], params['delta_vout_pct']

        ### MODIFICATION START: Added input validation ###
        if vin <= vout or vout <= 0 or iout <= 0 or fs <= 0:
            return {'error': "输入参数无效 (必须满足 Vin > Vout > 0, Iout > 0, fs > 0)。"}
        ### MODIFICATION END ###

        # 核心计算
        d = vout / vin
        delta_il = iout * (delta_il_pct / 100.0)
        
        # 电感计算
        l_val = (vin - vout) * d / (fs * delta_il)
        
        # 电容计算
        delta_vout = vout * (delta_vout_pct / 100.0)
        c_val = delta_il / (8 * fs * delta_vout)
        
        # 器件应力计算
        i_peak = iout + delta_il / 2.0
        i_rms_cap = delta_il / (2 * math.sqrt(3))
        v_stress_sw = vin
        
        # 创建被控对象传递函数 G_vd(s) = Vin / (s^2*LC + s*(L/R) + 1)
        R = vout / iout
        num = [vin]
        den = [l_val * c_val, l_val / R, 1]

        results = {
            "duty_cycle": d,
            "inductor_uH": l_val * 1e6,
            "capacitor_uF": c_val * 1e6,
            "i_peak_A": i_peak,
            "i_rms_cap_A": i_rms_cap,
            "v_stress_sw_V": v_stress_sw,
            "plant_tf_num": num,
            "plant_tf_den": den,
            "error": None
        }
        return results

    except (KeyError, ZeroDivisionError) as e:
        return {'error': f"输入参数错误或不完整: {e}"}