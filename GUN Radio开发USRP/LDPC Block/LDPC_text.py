#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# SPDX-License-Identifier: GPL-3.0
#
# GNU Radio Python Flow Graph
# Title: LDPC_text
# Author: abc
# GNU Radio version: 3.10.12.0

from PyQt5 import Qt
from gnuradio import qtgui
from gnuradio import gr
from gnuradio.filter import firdes
from gnuradio.fft import window
import sys
import signal
from PyQt5 import Qt
from argparse import ArgumentParser
from gnuradio.eng_arg import eng_float, intx
from gnuradio import eng_notation
import LDPC_text_epy_block_0 as epy_block_0  # embedded python block
import LDPC_text_epy_block_1 as epy_block_1  # embedded python block
import LDPC_text_epy_block_2 as epy_block_2  # embedded python block
import LDPC_text_epy_block_3 as epy_block_3  # embedded python block
import threading



class LDPC_text(gr.top_block, Qt.QWidget):

    def __init__(self):
        gr.top_block.__init__(self, "LDPC_text", catch_exceptions=True)
        Qt.QWidget.__init__(self)
        self.setWindowTitle("LDPC_text")
        qtgui.util.check_set_qss()
        try:
            self.setWindowIcon(Qt.QIcon.fromTheme('gnuradio-grc'))
        except BaseException as exc:
            print(f"Qt GUI: Could not set Icon: {str(exc)}", file=sys.stderr)
        self.top_scroll_layout = Qt.QVBoxLayout()
        self.setLayout(self.top_scroll_layout)
        self.top_scroll = Qt.QScrollArea()
        self.top_scroll.setFrameStyle(Qt.QFrame.NoFrame)
        self.top_scroll_layout.addWidget(self.top_scroll)
        self.top_scroll.setWidgetResizable(True)
        self.top_widget = Qt.QWidget()
        self.top_scroll.setWidget(self.top_widget)
        self.top_layout = Qt.QVBoxLayout(self.top_widget)
        self.top_grid_layout = Qt.QGridLayout()
        self.top_layout.addLayout(self.top_grid_layout)

        self.settings = Qt.QSettings("gnuradio/flowgraphs", "LDPC_text")

        try:
            geometry = self.settings.value("geometry")
            if geometry:
                self.restoreGeometry(geometry)
        except BaseException as exc:
            print(f"Qt GUI: Could not restore geometry: {str(exc)}", file=sys.stderr)
        self.flowgraph_started = threading.Event()

        ##################################################
        # Variables
        ##################################################
        self.samp_rate = samp_rate = 32000

        ##################################################
        # Blocks
        ##################################################

        self.epy_block_3 = epy_block_3.channel_pdu_external(verbose=True)
        self.epy_block_2 = epy_block_2.qpsk_demod_pdu(output_type='bit', verbose=True)
        self.epy_block_1 = epy_block_1.qpsk_mod_pdu(print_enable=True)
        self.epy_block_0 = epy_block_0.pdu_bit_h_noise(interval_s=2, num_bits=10, snr_db=10, signal_power=2, print_bits=True)


        ##################################################
        # Connections
        ##################################################
        self.msg_connect((self.epy_block_0, 'bits'), (self.epy_block_1, 'in'))
        self.msg_connect((self.epy_block_0, 'h'), (self.epy_block_2, 'set_h'))
        self.msg_connect((self.epy_block_0, 'noise_power'), (self.epy_block_2, 'set_noise'))
        self.msg_connect((self.epy_block_1, 'out'), (self.epy_block_3, 'in'))
        self.msg_connect((self.epy_block_3, 'out'), (self.epy_block_2, 'in'))


    def closeEvent(self, event):
        self.settings = Qt.QSettings("gnuradio/flowgraphs", "LDPC_text")
        self.settings.setValue("geometry", self.saveGeometry())
        self.stop()
        self.wait()

        event.accept()

    def get_samp_rate(self):
        return self.samp_rate

    def set_samp_rate(self, samp_rate):
        self.samp_rate = samp_rate




def main(top_block_cls=LDPC_text, options=None):

    qapp = Qt.QApplication(sys.argv)

    tb = top_block_cls()

    tb.start()
    tb.flowgraph_started.set()

    tb.show()

    def sig_handler(sig=None, frame=None):
        tb.stop()
        tb.wait()

        Qt.QApplication.quit()

    signal.signal(signal.SIGINT, sig_handler)
    signal.signal(signal.SIGTERM, sig_handler)

    timer = Qt.QTimer()
    timer.start(500)
    timer.timeout.connect(lambda: None)

    qapp.exec_()

if __name__ == '__main__':
    main()
