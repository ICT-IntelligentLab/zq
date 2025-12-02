# -*- coding: utf-8 -*-
import numpy as np
import pmt
from gnuradio import gr

####################################
# PDU QPSK Modulator
####################################
class qpsk_mod_pdu(gr.basic_block):
    """
    PDU QPSK Mapper
    输入：PDU（u8 类型，内容为0/1比特）
    输出：PDU（complex64）
    """

    def __init__(self, print_enable=False):
        gr.basic_block.__init__(self,
            name="PDU QPSK Modulator",
            in_sig=None,
            out_sig=None)

        self.print_enable = print_enable

        # 设置端口
        self.message_port_register_in(pmt.intern("in"))
        self.set_msg_handler(pmt.intern("in"), self.handle_pdu)
        self.message_port_register_out(pmt.intern("out"))

    def handle_pdu(self, msg):
        meta = pmt.car(msg)
        vec = pmt.cdr(msg)

        # 转 numpy
        bits = np.array(pmt.u8vector_elements(vec), dtype=np.uint8)

        # 确保偶数个比特
        if len(bits) % 2 != 0:
            bits = bits[:-1]

        # 拆成两比特一组
        bit_pairs = bits.reshape((-1, 2))

        # Gray 映射
        mapping = {
            (0,0): (1+1j),
            (0,1): (1-1j),
            (1,1): (-1-1j),
            (1,0): (-1+1j),
        }

        syms = np.array([mapping[tuple(b)] for b in bit_pairs], dtype=np.complex64)

        if self.print_enable:
            print(f"[QPSK Mod] Output symbols: {syms}")

        # 输出 PDU
        out_vec = pmt.init_c32vector(len(syms), syms.tolist())
        out_msg = pmt.cons(meta, out_vec)
        self.message_port_pub(pmt.intern("out"), out_msg)




