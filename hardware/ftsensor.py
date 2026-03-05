"""Prints the readings of a Bota Systems Serial sensor.

Usage: python bota_serial_example.py <port>

This example expects a device layout according to
_expected_device_layout, see below.
"""

import sys
import struct
import time
import threading

from collections import namedtuple

import serial
from crc import Calculator, Configuration

class BotaSerialSensorError(Exception):
    def __init__(self, message):
        super(BotaSerialSensorError, self).__init__(message)
        self.message = message

class BotaSerialSensor:

    BOTA_PRODUCT_CODE = 123456
    BAUDERATE = 460800
    SINC_LENGTH = 512
    CHOP_ENABLE = 0
    FAST_ENABLE = 0
    FIR_DISABLE = 1
    TEMP_COMPENSATION = 0 # 0: Disabled (recommended), 1: Enabled
    USE_CALIBRATION = 1 # 1: calibration matrix active, 0: raw measurements
    DATA_FORMAT = 0 # 0: binary, 1: CSV
    BAUDERATE_CONFIG = 4 # 0: 9600, 1: 57600, 2: 115200, 3: 230400, 4: 460800
    FRAME_HEADER = b'\xAA'
    # Note that the time step is set according to the sinc filter size!
    time_step = 0.01

    def __init__(self, port):
        self._port = port
        self._ser = serial.Serial()
        self._pd_thread_stop_event = threading.Event()
        DeviceSet = namedtuple('DeviceSet', 'name product_code config_func')
        self._expected_device_layout = {0: DeviceSet('BFT-SENS-SER-M8', self.BOTA_PRODUCT_CODE, self.bota_sensor_setup)}
        self._status = None
        self._fx = 0.0
        self._fy = 0.0
        self._fz = 0.0
        self._mx = 0.0
        self._my = 0.0
        self._mz = 0.0
        self._timestamp = 0.0
        self._temperature = 0.0


    def bota_sensor_setup(self):
        print("Trying to setup the sensor.")
        # Wait for streaming of data
        out = self._ser.read_until(bytes('App Init', 'ascii'))
        if not self.contains_bytes(bytes('App Init', 'ascii'), out):
            print("Sensor not streaming, check if correct port selected!")
            return False
        time.sleep(0.5)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()

        # Go to CONFIG mode
        cmd = bytes('C', 'ascii')
        self._ser.write(cmd)
        out = self._ser.read_until(bytes('r,0,C,0', 'ascii'))
        if not self.contains_bytes(bytes('r,0,C,0', 'ascii'), out):
            print("Failed to go to CONFIG mode.")
            return False

        # Communication setup
        comm_setup = f"c,{self.TEMP_COMPENSATION},{self.USE_CALIBRATION},{self.DATA_FORMAT},{self.BAUDERATE_CONFIG}"
        #print(comm_setup)
        cmd = bytes(comm_setup, 'ascii')
        self._ser.write(cmd)
        out = self._ser.read_until(bytes('r,0,c,0', 'ascii'))
        if not self.contains_bytes(bytes('r,0,c,0', 'ascii'), out):
            print("Failed to set communication setup.")
            return False
        self.time_step = 0.00001953125*self.SINC_LENGTH
        print("Timestep: {}".format(self.time_step))

        # Filter setup
        filter_setup = f"f,{self.SINC_LENGTH},{self.CHOP_ENABLE},{self.FAST_ENABLE},{self.FIR_DISABLE}"
        #print(filter_setup)
        cmd = bytes(filter_setup, 'ascii')
        self._ser.write(cmd)
        out = self._ser.read_until(bytes('r,0,f,0', 'ascii'))
        if not self.contains_bytes(bytes('r,0,f,0', 'ascii'), out):
            print("Failed to set filter setup.")
            return False

        # Go to RUN mode
        cmd = bytes('R', 'ascii')
        self._ser.write(cmd)
        out = self._ser.read_until(bytes('r,0,R,0', 'ascii'))
        if not self.contains_bytes(bytes('r,0,R,0', 'ascii'), out):
            print("Failed to go to RUN mode.")
            return False

        return True

    def contains_bytes(self, subsequence, sequence):
        return subsequence in sequence

    def _processdata_thread(self):
        while not self._pd_thread_stop_event.is_set():
            frame_synced = False
            crc16X25Configuration = Configuration(16, 0x1021, 0xFFFF, 0xFFFF, True, True)
            crc_calculator = Calculator(crc16X25Configuration)

            while not frame_synced and not self._pd_thread_stop_event.is_set():
                possible_header = self._ser.read(1)
                if self.FRAME_HEADER == possible_header:
                    #print(possible_header)
                    data_frame = self._ser.read(34)
                    crc16_ccitt_frame = self._ser.read(2)

                    crc16_ccitt = struct.unpack_from('H', crc16_ccitt_frame, 0)[0]
                    checksum = crc_calculator.checksum(data_frame)
                    if checksum == crc16_ccitt:
                        print("Frame synced")
                        frame_synced = True
                    else:
                        self._ser.read(1)

            while frame_synced and not self._pd_thread_stop_event.is_set():            
                start_time = time.perf_counter()
                frame_header = self._ser.read(1)

                if frame_header != self.FRAME_HEADER:
                    print("Lost sync")
                    frame_synced = False
                    break

                data_frame = self._ser.read(34)
                crc16_ccitt_frame = self._ser.read(2)

                crc16_ccitt = struct.unpack_from('H', crc16_ccitt_frame, 0)[0]
                checksum = crc_calculator.checksum(data_frame)
                if checksum != crc16_ccitt:
                    print("CRC mismatch received")
                    break

                self._status = struct.unpack_from('H', data_frame, 0)[0]

                self._fx = struct.unpack_from('f', data_frame, 2)[0]
                self._fy = struct.unpack_from('f', data_frame, 6)[0]
                self._fz = struct.unpack_from('f', data_frame, 10)[0]
                self._mx = struct.unpack_from('f', data_frame, 14)[0]
                self._my = struct.unpack_from('f', data_frame, 18)[0]
                self._mz = struct.unpack_from('f', data_frame, 22)[0]

                self._timestamp = struct.unpack_from('I', data_frame, 26)[0]

                self._temperature = struct.unpack_from('f', data_frame, 30)[0]
                
                time_diff = time.perf_counter() - start_time
 

    def _my_loop(self):

        try:
            while 1:
                print('Run my loop')

                print("Status {}".format(self._status))

                print("Fx {}".format(self._fx))
                print("Fy {}".format(self._fy))
                print("Fz {}".format(self._fz))
                print("Mx {}".format(self._mx))
                print("My {}".format(self._my))
                print("Mz {}".format(self._mz))

                print("Timestamp {}".format(self._timestamp))

                print("Temperature {}\n".format(self._temperature))

                time.sleep(1.0)

        except KeyboardInterrupt:
            # ctrl-C abort handling
            print('stopped')


    def run(self):

        self._ser.baudrate = self.BAUDERATE
        self._ser.port = self._port
        self._ser.timeout = 10

        try:
            self._ser.open()
            print("Opened serial port {}".format(self._port))
        except:
            raise BotaSerialSensorError('Could not open port')

        if not self._ser.is_open:
            raise BotaSerialSensorError('Could not open port')

        if not self.bota_sensor_setup():
            print('Could not setup sensor!')
            return

        #check_thread = threading.Thread(target=self._check_thread)
        #check_thread.start()
        proc_thread = threading.Thread(target=self._processdata_thread)
        proc_thread.start()
        
        device_running = True

        if device_running:
            self._my_loop()

        self._pd_thread_stop_event.set()
        proc_thread.join()
        #check_thread.join()

        self._ser.close()

        if not device_running:
            raise BotaSerialSensorError('Device is not running')

    @staticmethod
    def _sleep(duration, get_now=time.perf_counter):
        now = get_now()
        end = now + duration
        while now < end:
            now = get_now()



class CustomBotaSerialSensor(BotaSerialSensor):
    def __init__(self, port, alpha=0.7):

        super().__init__(port)
        

        self._prev_fx = 0
        self._prev_fy = 0
        self._prev_fz = 0
        self._prev_mx = 0
        self._prev_my = 0
        self._prev_mz = 0
        self.alpha = alpha  

    def run(self):
        raise NotImplementedError("Please try function start() and end()")
    def start(self):
        """
        Custom run method to modify the behavior of the parent class.
        This version reads the sensor data in the main thread instead of a separate loop.
        """
        self._ser.baudrate = self.BAUDERATE
        self._ser.port = self._port
        self._ser.timeout = 10

        try:
            self._ser.open()
            print("Opened serial port {}".format(self._port))
        except:
            raise BotaSerialSensorError('Could not open port')

        if not self._ser.is_open:
            raise BotaSerialSensorError('Could not open port')

        if not self.bota_sensor_setup():
            print('Could not setup sensor!')
            return
        
        if hasattr(self, 'proc_thread'):
            print("Attribute is defined.")
            raise NotImplementedError("Only support one sensor one thread! You have started somewhere already.")
        else:
            # print("Attribute is not defined.")
            self.proc_thread = threading.Thread(target=self._processdata_thread)
            self.proc_thread.start()
            self._sleep(1.0)
            self._record_bias()

    def end(self):
        self._pd_thread_stop_event.set()
        if hasattr(self, 'proc_thread'):
            self.proc_thread.join()
            del self.proc_thread
        else:
            raise NotImplementedError("Didn't start or the thread has ended already")
        self._ser.close()
    def _record_bias(self):

        fx_sum = 0
        fy_sum = 0
        fz_sum = 0
        mx_sum = 0
        my_sum = 0
        mz_sum = 0


        for _ in range(10):
            fx_sum += self._fx
            fy_sum += self._fy
            fz_sum += self._fz
            mx_sum += self._mx
            my_sum += self._my
            mz_sum += self._mz
            time.sleep(0.1)  


        self._fx_bias = fx_sum / 10
        self._fy_bias = fy_sum / 10
        self._fz_bias = fz_sum / 10
        self._mx_bias = mx_sum / 10
        self._my_bias = my_sum / 10
        self._mz_bias = mz_sum / 10
    # def _record_bias(self):
    #     self._fx_bias = self._fx
    #     self._fy_bias = self._fy
    #     self._fz_bias = self._fz
    #     self._mx_bias = self._mx
    #     self._my_bias = self._my
    #     self._mz_bias = self._mz

    def get_wrench(self):
        return [
            self._fx-self._fx_bias,
            self._fy-self._fy_bias,
            self._fz-self._fz_bias,
            self._mx-self._mx_bias,
            self._my-self._my_bias,
            self._mz-self._mz_bias,
        ]
    

    def get_wrench_alpha(self):

        fx = self._fx - self._fx_bias
        fy = self._fy - self._fy_bias
        fz = self._fz - self._fz_bias
        mx = self._mx - self._mx_bias
        my = self._my - self._my_bias
        mz = self._mz - self._mz_bias


        filtered_fx = self.alpha * fx + (1 - self.alpha) * self._prev_fx
        filtered_fy = self.alpha * fy + (1 - self.alpha) * self._prev_fy
        filtered_fz = self.alpha * fz + (1 - self.alpha) * self._prev_fz
        filtered_mx = self.alpha * mx + (1 - self.alpha) * self._prev_mx
        filtered_my = self.alpha * my + (1 - self.alpha) * self._prev_my
        filtered_mz = self.alpha * mz + (1 - self.alpha) * self._prev_mz


        self._prev_fx = filtered_fx
        self._prev_fy = filtered_fy
        self._prev_fz = filtered_fz
        self._prev_mx = filtered_mx
        self._prev_my = filtered_my
        self._prev_mz = filtered_mz


        return [
            filtered_fx,
            filtered_fy,
            filtered_fz,
            filtered_mx,
            filtered_my,
            filtered_mz,
        ]
    
    def get_wrench_deadband(self, deadband_thresholds=[1., 1.0, 1.0, 0.02, 0.02, 0.02]):


        filtered_values = self.get_wrench_alpha()
        deadband_values = [
            value if abs(value) >= deadband_thresholds[i] else 0.0
            for i, value in enumerate(filtered_values)
        ]

        return deadband_values




# if __name__ == '__main__':

#     print('bota_serial_example started')

#     if len(sys.argv) > 1:
#         try:
#             bota_sensor_1 = BotaSerialSensor(sys.argv[1])
#             bota_sensor_1.run()
#         except BotaSerialSensorError as expt:
#             print('bota_serial_example failed: ' + expt.message)
#             sys.exit(1)
#     else:
#         print('usage: bota_serial_example portname')
#         sys.exit(1)

if __name__ == '__main__':

    print('bota_serial_example started')

    if len(sys.argv) > 1:
        try:
            bota_sensor_1 = CustomBotaSerialSensor(sys.argv[1])
            bota_sensor_1.start()
        except BotaSerialSensorError as expt:
            print('bota_serial_example failed: ' + expt.message)
            sys.exit(1)
    else:
        print('usage: bota_serial_example portname')
        sys.exit(1)

    
    try:
        while 1:
            print('Run my loop')

            print("Status {}".format(bota_sensor_1._status))

            print("Fx {}".format(bota_sensor_1._fx))
            print("Fy {}".format(bota_sensor_1._fy))
            print("Fz {}".format(bota_sensor_1._fz))
            print("Mx {}".format(bota_sensor_1._mx))
            print("My {}".format(bota_sensor_1._my))
            print("Mz {}".format(bota_sensor_1._mz))
            print("bias z {}".format(bota_sensor_1._fx_bias))

            print("Timestamp {}".format(bota_sensor_1._timestamp))

            print("Temperature {}\n".format(bota_sensor_1._temperature))
            print("wrench {}\n".format(bota_sensor_1.get_wrench()))

            time.sleep(1.0)

    except KeyboardInterrupt:
        # ctrl-C abort handling
        print('stopped')

    bota_sensor_1.end()
