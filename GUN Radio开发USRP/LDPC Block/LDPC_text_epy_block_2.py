import numpy as np
from gnuradio import gr
import pmt

####################################
# QPSK Demod PDU
####################################
class qpsk_demod_pdu(gr.basic_block):
    """
    QPSK Demod PDU
    输入：IQ PDU
    输出：bit 或 LLR PDU
    """
    def __init__(self, output_type='llr', verbose=False):
        gr.basic_block.__init__(self,
            name="QPSK Demod PDU",
            in_sig=None, out_sig=None)

        self.output_type = output_type.lower()
        self.verbose = verbose
        self.h = 1.0+0j
        self.noiseVar = 1.0

        # 消息端口
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle_msg)
        self.message_port_register_out(pmt.intern("out"))

        self.message_port_register_in(pmt.intern("set_h"))
        self.set_msg_handler(pmt.intern("set_h"), self.handle_h)
        self.message_port_register_in(pmt.intern("set_noise"))
        self.set_msg_handler(pmt.intern("set_noise"), self.handle_noise)

    def handle_h(self, msg):
        self.h = pmt.to_complex(msg)
      
    def handle_noise(self, msg):
        self.noiseVar = float(pmt.to_python(msg))
      
    def handle_msg(self, msg):
        meta = pmt.car(msg)
        payload = pmt.cdr(msg)
        iq = np.array(pmt.c32vector_elements(payload), dtype=np.complex64)
        n = len(iq)

        # 信道补偿
        r = iq * np.conj(self.h)

        demod = np.zeros(2*n, dtype=np.float32)
        demod[0::2] = 2*np.real(r) / self.noiseVar
        demod[1::2] = 2*np.imag(r) / self.noiseVar

        if self.output_type == 'bit':
            bits = (demod < 0).astype(np.uint8)
            out_msg = pmt.cons(meta, pmt.init_u8vector(len(bits), bits))
        else:
            out_msg = pmt.cons(meta, pmt.init_f32vector(len(demod), demod))

        # verbose 打印 PDU 内容
        if self.verbose:
            payload = pmt.cdr(out_msg)
            if self.output_type == 'bit':
                data = np.array(pmt.u8vector_elements(payload), dtype=np.uint8)
            else:
                data = np.array(pmt.f32vector_elements(payload), dtype=np.float32)
            print("[Demod] Output:", data.tolist())

        # 发送消息
        self.message_port_pub(pmt.intern("out"), out_msg)
