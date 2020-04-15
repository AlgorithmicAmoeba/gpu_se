

class Inputs:
    """Creates fake inputs for the glucose feed from past data
    """

    def __call__(self, t):
        t_batch = 30

        Cn_in = 0.625 * 10 / 60  # (g/L) / (g/mol) = mol/L
        if t < t_batch + 66:
            CgFg = 0.141612826257827
        elif t < t_batch + 101:
            CgFg = 0.21241923938674
        elif t < t_batch + 137:
            CgFg = 0.283225652515653
        else:
            CgFg = 0.354032065644566

        # if t < t_batch + 66:
        #     CgFg = 0.141612826257827
        # else:
        #     CgFg = 0.21241923938674
        Cg_in = 314.19206 / 180  # (g/L) / (g/mol) = mol/L
        Cb_in = 10  # mol/L
        Fm_in = 0

        if t < t_batch:
            Fg_in = 0
            Fn_in = 0
            Fb_in = 0
            F_out = 0
        else:
            Fg_in = CgFg / 180 / Cg_in  # (g/h) / (g/mol) / (mol/L) = L/h
            Fn_in = 0.625 / 1000 / Cn_in / 60  # (mg/h) / (mg/g) / (mol/L) / (g/mol) = L/h
            Fb_in = 0.00006  # L/h
            F_out = Fg_in + Fn_in + Fb_in + Fm_in

        Qco_in = 8.67 / 1000 * 60  # (ml / min) / (ml/L) * (min/h) = L/h
        Fco_in = 87 * Qco_in / 8.314 / 298  # (kPa) * (L/h) / (L*kPa/mol/K) / (K) = mol/h
        Cco_in = 8.7  # mol CO2 / mol total

        Qo_in = 99.92 / 1000 * 60  # (ml / min) / (ml/L) * (min/h) = L/h
        Fo_in = 87 * Qo_in / 8.314 / 298  # (kPa) * (L/h) / (L*kPa/mol/K) / (K) = mol/h
        Co_in = 21  # mol CO2 / mol total

        Fg_out = Fco_in + Fo_in

        T_amb = 25
        Q = 5 / 9

        return Fg_in, Cg_in, Fco_in, Cco_in, Fo_in, Co_in, Fg_out, Cn_in, Fn_in, Fb_in, Cb_in, Fm_in, F_out, T_amb, Q
