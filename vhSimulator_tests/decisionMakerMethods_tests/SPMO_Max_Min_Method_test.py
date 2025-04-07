import unittest
from vhSimulator import SPMO_Max_Min_Method

class TestSPMO_Max_Min_Method(unittest.TestCase):
    
    def setUp(self):
        self.test_data = [
            {'Network': 'NBIoT-5g-1', 'Status': 'Online', 'Distance': 4303.039157647008, 'RSSI': -99.38, 'SNR': 8.12, 'Throughput': 0.067, 'Protocol': 'NB-IoT-5G', 'PC': 0.25, 'MC': 5},
            {'Network': 'LoRa-1', 'Status': 'Online', 'Distance': 1951.7812986186484, 'RSSI': -87.75, 'SNR': 8.82, 'Throughput': 0.041, 'Protocol': 'LoRa-868', 'PC': 0.05, 'MC': 1},
            {'Network': 'LoRa-2', 'Status': 'Online', 'Distance': 9483.326839420037, 'RSSI': -112.88, 'SNR': 0.52, 'Throughput': 0.004, 'Protocol': 'LoRa-868', 'PC': 0.05, 'MC': 1},
            {'Network': 'LTE-4g-1', 'Status': 'Online', 'Distance': 17042.94762466105, 'RSSI': -90.15, 'SNR': 8.85, 'Throughput': 42.776, 'Protocol': 'LTE-4G', 'PC': 1.05, 'MC': 3},
            {'Network': 'WiFi-Max-1', 'Status': 'Online', 'Distance': 4589.336234625973, 'RSSI': -86.53, 'SNR': 7.4, 'Throughput': 11.675, 'Protocol': 'WiFi-Max', 'PC': 0.8, 'MC': 2}
        ]

    def test_get_maximum_value(self):
        method = SPMO_Max_Min_Method("MaxMethod", "Throughput", max_value=True)
        method.inputs = self.test_data
        result = method.get_maximum_value()
        self.assertEqual(result['Network'], "LTE-4g-1")
    
    def test_get_minimum_value(self):
        method = SPMO_Max_Min_Method("MinMethod", "Throughput", max_value=False)
        method.inputs = self.test_data
        result = method.get_minimum_value()
        self.assertEqual(result['Network'], "LoRa-2")
    
    def test_makeDecision_max(self):
        method = SPMO_Max_Min_Method("MaxMethod", "Throughput", max_value=True)
        method.inputs = self.test_data
        result = method.makeDecision()
        self.assertEqual(result['Network'], "LTE-4g-1")
    
    def test_makeDecision_min(self):
        method = SPMO_Max_Min_Method("MinMethod", "Throughput", max_value=False)
        method.inputs = self.test_data
        result = method.makeDecision()
        self.assertEqual(result['Network'], "LoRa-2")

if __name__ == "__main__":
    unittest.main()