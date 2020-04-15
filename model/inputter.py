

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
        Ca_in = 10 # mol/L
        Cb_in = 10  # mol/L
        Fm_in = 0

        if t < t_batch:
            Fg_in = 0
            Fa_in = 0
            Fb_in = 0
            F_out = 0
        else:
            Fg_in = CgFg / 180 / Cg_in  # (g/h) / (g/mol) / (mol/L) = L/h
            Fn_in = 0.625 / 1000 / Cn_in / 60  # (mg/h) / (mg/g) / (mol/L) / (g/mol) = L/h
            Fa_in = 0
            Fb_in = 0.00006  # L/h
            F_out = Fg_in + Fn_in + Fb_in + Fm_in

        T_amb = 25
        Q = 5 / 9

        return Fg_in, Cg_in, Fa_in, Ca_in, Fb_in, Cb_in, Fm_in, F_out, T_amb, Q
