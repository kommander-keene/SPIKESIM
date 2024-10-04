class YamadaParams:
    def __init__(self, A_bias, B_abs, a_diff_abs, y_G, y_Q, y_I):
        self.A_bias = A_bias
        self.B_bias = B_abs
        self.a_abs = a_diff_abs
        self.y_G = y_G
        self.y_Q = y_Q
        self.y_I = y_I
    def to_args(self):
        return (self.A_bias, self.B_bias, self.a_abs, self.y_G, self.y_Q, self.y_I)