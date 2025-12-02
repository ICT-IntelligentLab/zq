import numpy as np
from gnuradio import gr
import pmt
import time
import threading
####################################
# Channel PDU (External h + noise)
####################################
class channel_pdu_external(gr.basic_block):
    """
    PDU 信道：乘 h + 加噪声
    输入：
        - in: IQ PDU (c32vector)
        - set_h: 外部复数 h (PMT complex)
        - set_noise: 外部噪声功率 (float)
    输出：
        - out: IQ PDU
    """
    def __init__(self, verbose=False):
        gr.basic_block.__init__(self,
            name="Channel PDU",
            in_sig=None, out_sig=None)

        self.verbose = verbose
        self.h = 1.0+0j
        self.noiseVar = 0.0

        # 消息端口
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle_in)

        self.message_port_register_in(pmt.intern("set_h"))
        self.set_msg_handler(pmt.intern("set_h"), self.handle_h)

        self.message_port_register_in(pmt.intern("set_noise"))
        self.set_msg_handler(pmt.intern("set_noise"), self.handle_noise)

        self.message_port_register_out(pmt.intern("out"))

    def handle_h(self, msg):
        self.h = pmt.to_complex(msg)  # msg 是 PMT complex
        if self.verbose:
            print("[Channel] Set h =", self.h)

    def handle_noise(self, msg):
        self.noiseVar = float(pmt.to_python(msg))
        if self.verbose:
            print("[Channel] Set noiseVar =", self.noiseVar)

    def handle_in(self, msg):
        meta = pmt.car(msg)
        payload = pmt.cdr(msg)
        iq = np.array(pmt.c32vector_elements(payload), dtype=np.complex64)
        n = len(iq)

        # 生成噪声
        if self.noiseVar > 0:
            noise = np.sqrt(self.noiseVar/2)*(np.random.randn(n)+1j*np.random.randn(n))
        else:
            noise = np.zeros(n, dtype=np.complex64)

        out_iq = iq*self.h + noise

        if self.verbose:
            print("[Channel] output IQ:", out_iq)

        out_msg = pmt.cons(meta, pmt.init_c32vector(len(out_iq), out_iq))
        self.message_port_pub(pmt.intern("out"), out_msg)