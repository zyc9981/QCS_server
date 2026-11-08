import struct
import serial
from typing import Optional, Dict

class GouMaxTOF:
    """
    Simple UART driver for GouMax C-band 4-in-1 Tunable Filter.

    Public API:
      - set_wavelength(ch: int, wl_nm: float) -> None
      - get_wavelength(ch: int) -> float

    Notes:
      * ch is 1..4
      * wl_nm is in nanometers (1525.00 .. 1570.00 typical per spec)
      * Device protocol uses big-endian 16-bit words and a 1-byte 0xAA head.
    """

    # ---- Protocol constants (big-endian "words") ----
    HEAD = 0xAA

    # Command words (two 16-bit words each)
    CMD_SET_WL  = (0x5354, 0x574C)  # "ST", "WL"
    CMD_GET_WL  = (0x5244, 0x574C)  # "RD", "WL"

    # Length words (in *words*) for the payload range the spec defines
    LEN_SET_WL = 0x0009  # words in fields 4..12 (set)
    LEN_GET_WL = 0x0001  # words in field 4       (read)

    # Expected response length words for fields 4..13
    LEN_RESP_WL = 0x000A

    # Error codes per spec
    ERR_NO_ERROR        = 0x0000
    ERR_UNKNOWN_CMD     = 0x0001
    ERR_WL_OOR          = 0x0002
    ERR_CTRL_OOR        = 0x0004
    ERR_CHECKSUM        = 0x0009

    # Practical wavelength limits from the datasheet (nm)
    WL_MIN_NM = 1525.00
    WL_MAX_NM = 1570.00

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 1.0):
        """
        Example: GouMaxTOF('COM6')  or  GouMaxTOF('/dev/ttyUSB0')
        """
        self.ser = serial.Serial(port=port, baudrate=baudrate, bytesize=serial.EIGHTBITS,
                                 parity=serial.PARITY_NONE, stopbits=serial.STOPBITS_ONE,
                                 timeout=timeout, xonxoff=False, rtscts=False, dsrdtr=False)

    def close(self):
        try:
            self.ser.close()
        except Exception:
            pass

    # ---------------------- Public API ----------------------

    def set_wavelength(self, ch: int, wl_nm: float) -> None:
        """
        Set the wavelength for a single channel (1..4) in nm.
        """
        self._validate_channel(ch)
        self._validate_wavelength_nm(wl_nm)
        wl_pm = int(round(wl_nm * 1000.0))  # protocol uses pm as 32-bit int
        ctrl = 1 << (ch - 1)                # b0->CH1, b1->CH2, b2->CH3, b3->CH4

        # Build the four 32-bit wavelength slots (pm); only the selected channel is nonzero.
        wl_words = []
        for c in range(1, 5):
            val = wl_pm if c == ch else 0
            wl_words.extend(self._u32_to_two_u16(val))

        pkt = self._build_packet(self.CMD_SET_WL, self.LEN_SET_WL, [ctrl] + wl_words)
        resp = self._xfer(pkt)
        self._parse_and_check_resp(resp, self.CMD_SET_WL, expect_ctrl=ctrl)

    def get_wavelength(self, ch: int) -> float:
        """
        Read back the wavelength for channel (1..4), in nm.
        """
        self._validate_channel(ch)
        ctrl = 1 << (ch - 1)

        pkt = self._build_packet(self.CMD_GET_WL, self.LEN_GET_WL, [ctrl])
        resp = self._xfer(pkt)
        words = self._parse_and_check_resp(resp, self.CMD_GET_WL, expect_ctrl=ctrl)

        # Response layout after fixed header:
        # [error, ctrl, (CH1_hi, CH1_lo), (CH2_hi, CH2_lo), (CH3_hi, CH3_lo), (CH4_hi, CH4_lo)]
        # words indexes: 0:error, 1:ctrl, 2..9: wavelengths (8 words)
        ch_hi = words[2 + 2*(ch-1)]
        ch_lo = words[2 + 2*(ch-1) + 1]
        wl_pm = self._two_u16_to_u32(ch_hi, ch_lo)
        return wl_pm / 1000.0

    # --------------------- Low-level helpers ---------------------

    def _build_packet(self, cmd_words, length_word, payload_words):
        """
        Packet bytes:
          [0xAA] + W1 + W2 + LEN + payload(words) + CHECKSUM
        CHECKSUM is sum of bytes for fields 1..(before checksum), modulo 2^16.
        All words are big-endian.
        """
        w1, w2 = cmd_words
        # Serialize words after head to compute checksum
        body = struct.pack(">HHH", w1, w2, length_word)
        if payload_words:
            body += struct.pack(">" + "H"*len(payload_words), *payload_words)

        checksum = self._checksum_bytes(body)  # sum of bytes modulo 2^16
        pkt = bytes([self.HEAD]) + body + struct.pack(">H", checksum)
        return pkt

    def _xfer(self, pkt: bytes) -> bytes:
        self._flush_io()
        self.ser.write(pkt)

        # Read response head (1 byte) + 3 header words (6 bytes)
        header = self._read_exact(7)
        if header[0] != self.HEAD:
            raise IOError(f"Bad head byte: 0x{header[0]:02X}")

        # Parse header words (big-endian)
        cmd1, cmd2, length_words = struct.unpack(">HHH", header[1:7])

        # Then read payload (length_words * 2 bytes) + checksum (2 bytes)
        payload_plus_ck = self._read_exact(length_words*2 + 2)
        resp = header + payload_plus_ck

        # Verify checksum
        body = resp[1:-2]  # fields 1..before checksum (exclude head and last 2 bytes)
        rx_ck = struct.unpack(">H", resp[-2:])[0]
        calc_ck = self._checksum_bytes(body)
        if rx_ck != calc_ck:
            raise IOError(f"Checksum mismatch (rx=0x{rx_ck:04X}, calc=0x{calc_ck:04X})")

        return resp

    def _parse_and_check_resp(self, resp: bytes, expected_cmd, expect_ctrl: Optional[int] = None):
        """
        Returns the response words starting at Field 4 (i.e., error, ctrl, then data words).
        """
        # Unpack header
        _, = resp[:1]             # head
        cmd1, cmd2, length = struct.unpack(">HHH", resp[1:7])
        if (cmd1, cmd2) != expected_cmd:
            raise IOError(f"Unexpected response command 0x{cmd1:04X} 0x{cmd2:04X}")

        # Extract payload words (length words), then checksum already verified
        payload = resp[7:7 + length*2]
        words = list(struct.unpack(">" + "H"*length, payload))

        # First word is error code
        err = words[0]
        if err != self.ERR_NO_ERROR:
            err_map = {
                self.ERR_UNKNOWN_CMD: "Unknown command",
                self.ERR_WL_OOR: "Wavelength out of range / invalid for channel bit",
                self.ERR_CTRL_OOR: "Channel control out of range",
                self.ERR_CHECKSUM: "Checksum error",
            }
            raise IOError(f"Device error 0x{err:04X}: {err_map.get(err, 'Unspecified error')}")

        # Second word is channel control echo
        ctrl = words[1]
        if expect_ctrl is not None and (ctrl & 0x000F) != (expect_ctrl & 0x000F):
            raise IOError(f"Channel control mismatch (resp=0x{ctrl:04X}, expected=0x{expect_ctrl:04X})")

        return words  # [error, ctrl, ...data words...]

    # --------------------- Utilities ---------------------

    @staticmethod
    def _u32_to_two_u16(val: int):
        if not (0 <= val <= 0xFFFFFFFF):
            raise ValueError("u32 out of range")
        hi = (val >> 16) & 0xFFFF
        lo = val & 0xFFFF
        return [hi, lo]

    @staticmethod
    def _two_u16_to_u32(hi: int, lo: int) -> int:
        return ((hi & 0xFFFF) << 16) | (lo & 0xFFFF)

    @staticmethod
    def _checksum_bytes(b: bytes) -> int:
        # Sum of bytes (not words) for fields 1..(before checksum), modulo 2^16
        return sum(b) & 0xFFFF

    def _read_exact(self, n: int) -> bytes:
        buf = bytearray()
        while len(buf) < n:
            chunk = self.ser.read(n - len(buf))
            if not chunk:
                raise TimeoutError(f"Serial read timeout (wanted {n}, got {len(buf)})")
            buf.extend(chunk)
        return bytes(buf)

    def _flush_io(self):
        try:
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()
        except Exception:
            pass

    def _validate_channel(self, ch: int):
        if ch not in (1,2,3,4):
            raise ValueError("Channel must be 1..4")

    def _validate_wavelength_nm(self, wl_nm: float):
        if not (self.WL_MIN_NM <= wl_nm <= self.WL_MAX_NM):
            raise ValueError(f"wavelength must be in [{self.WL_MIN_NM:.2f}, {self.WL_MAX_NM:.2f}] nm")


tof = GouMaxTOF(port="COM6")  # or "/dev/ttyUSB0"

# Set CH2 to 1550.12 nm
tof.set_wavelength(1, 1551.50)
tof.set_wavelength(2, 1552.50)
tof.set_wavelength(3, 1553.50)
tof.set_wavelength(4, 1554.50)

# Read back CH2
print("CH2 =", tof.get_wavelength(2), "nm")

tof.close()