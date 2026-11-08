import time
from toptica.lasersdk.client import Client, NetworkConnection

class DLCProSimple:
    def __init__(self, host_or_label: str, timeout_s: float = 5.0):
        self.conn = NetworkConnection(host_or_label, timeout=timeout_s)
        self.client = None

    def open(self):
        if self.client is None:
            self.client = Client(self.conn)
            self.client.open()
        return self

    def close(self):
        if self.client is not None:
            try:
                self.client.close()
            finally:
                self.client = None

    # --------- emission ----------
    def emission_on(self):
        self.client.set("emission", True)

    def emission_off(self):
        self.client.set("emission", False)

    # --------- wavelength (nm) ----------
    def set_wavelength_nm(self, nm: float):
        self.client.set("laser1:ctl:wavelength-set", float(nm))

    def get_wavelength_nm(self) -> float:
        return float(self.client.get("laser1:ctl:wavelength-act"))

    def set_piezo_V(self, V: float) -> float:
        self.client.set("laser1:dl:pc:voltage-set", float(V))

    def get_piezo_V(self) -> float:
        return float(self.client.get("laser1:dl:pc:voltage-act"))
    
    def get_piezo_V_setpoint(self) -> float:
        return float(self.client.get("laser1:dl:pc:voltage-set"))
    
    def set_power(self, P: float):
        self.client.set("laser1:power-stabilization:setpoint", float(P))


# --------- example usage ----------
if __name__ == "__main__":
    HOST = "192.168.1.200"  # your label or IP

    drv = DLCProSimple(HOST).open()
    try:
        # Turn emission on ONLY if interlocks are satisfied and you intend to output light
        # drv.emission_on()

        # Set wavelength (nm)
        # drv.set_wavelength_nm(1547.7)        # waits until settled by default
        # print("λ =", drv.get_wavelength_nm(), "nm")
        drv.set_power(30)

        # drv.emission_off()
    finally:
        drv.close()
