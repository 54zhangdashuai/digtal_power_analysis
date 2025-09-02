# logic/llc_resonant_logic.py
import math
import numpy as np

def calculate_llc(params: dict) -> dict:
    """
    基于FHA的LLC谐振变换器计算。
    返回谐振腔参数和用于绘图的增益曲线数据。
    """
    try:
        vin, vout, pout, fr = params['vin'], params['vout'], params['pout'], params['fr']
        n, q, k = params['n'], params['q'], params['k']

        # 核心计算
        wr = 2 * math.pi * fr
        
        # 1. 等效负载电阻
        rac = (8 * n**2 * vout**2) / (math.pi**2 * pout)
        
        # 2. 谐振腔参数
        lr = (q * rac) / wr
        cr = 1 / (wr**2 * lr)
        lm = k * lr
        
        # 3. 计算增益曲线 M = f(fn)
        fn_array = np.linspace(0.1, 3.0, 500) # 归一化频率轴
        
        # Gain formula M(fn) = | (k * fn^2) / ( (k+1)*fn^2 - 1 + j*Q*k*fn*(fn^2-1) ) |
        # 使用复数计算幅值
        term_real = (k + 1) * fn_array**2 - 1
        term_imag = q * k * fn_array * (fn_array**2 - 1)
        
        gain_complex = (k * fn_array**2) / (term_real + 1j * term_imag)
        gain_array = np.abs(gain_complex)

        results = {
            "equiv_resistance_rac_ohm": rac,
            "resonant_inductor_lr_uH": lr * 1e6,
            "resonant_capacitor_cr_nF": cr * 1e9,
            "magnetizing_inductor_lm_uH": lm * 1e6,
            "gain_curve_fn": fn_array,
            "gain_curve_M": gain_array,
            "error": None
        }
        return results
        
    except (KeyError, ZeroDivisionError) as e:
        return {'error': f"输入参数错误或不完整: {e}"}