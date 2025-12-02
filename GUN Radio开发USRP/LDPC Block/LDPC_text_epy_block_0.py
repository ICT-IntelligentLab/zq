import numpy as np
from gnuradio import gr
import pmt
import time
import threading

####################################
# PDU Bit + Channel + Noise Power Source
####################################
class pdu_bit_h_noise(gr.basic_block):
    """
    每隔 interval_s 秒发送 num_bits 比特，同时生成一个复高斯信道 h
    并根据给定 SNR 和信号功率计算噪声功率
    """
    def __init__(self, interval_s=1.0, num_bits=10, snr_db=10.0, signal_power=1.0, print_bits=True):
        gr.basic_block.__init__(self,
            name="pdu_bit_h_noise_source",
            in_sig=None, out_sig=None)

        self.interval_s = float(interval_s)
        self.num_bits = int(num_bits)
        self.snr_db = float(snr_db)
        self.signal_power = float(signal_power)
        self.print_bits = bool(print_bits)

        # 消息端口
        self.message_port_register_out(pmt.intern("bits"))
        self.message_port_register_out(pmt.intern("h"))
        self.message_port_register_out(pmt.intern("noise_power"))

        # 开启线程
        self.keep_running = True
        self.thread = threading.Thread(target=self._run)
        self.thread.daemon = True
        self.thread.start()

    def _run(self):
        while self.keep_running:
            # -------------------
            # 生成比特
            # -------------------
            bits = np.random.randint(0, 2, self.num_bits).astype(np.uint8)
            if self.print_bits:
                print("#######################")

                print("[Bit Source] Bits:", bits.tolist())

            # ⚠️ 这里改为发送 PDU
            bits_msg = pmt.init_u8vector(len(bits), bits)
            bits_pdu = pmt.cons(pmt.PMT_NIL, bits_msg)  # 包装成 PDU
            self.message_port_pub(pmt.intern("bits"), bits_pdu)

            # -------------------
            # 生成复高斯信道 h
            # -------------------
            h = (np.random.randn() + 1j*np.random.randn()) / np.sqrt(2)
            h_msg = pmt.from_complex(h)  # 直接发送复数 PMT
            self.message_port_pub(pmt.intern("h"), h_msg)

            # -------------------
            # 计算噪声功率
            # -------------------
            snr_linear = 10**(self.snr_db / 10)
            noise_power = self.signal_power / snr_linear
            noise_msg = pmt.from_double(float(noise_power))
            self.message_port_pub(pmt.intern("noise_power"), noise_msg)

            time.sleep(self.interval_s)

    def stop(self):
        self.keep_running = False
        self.thread.join()
        return super().stop()








