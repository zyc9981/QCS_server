import queue
import threading
import time

from controller_common import open_arduino, readline_str


class RxCommandsMixin:
    def _rx_wait_for_line_blocking(self, expected_text: str, timeout_s: float, context: str):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            line = readline_str(self.rx)
            if not line:
                continue

            text = line.strip()
            if not text:
                continue

            if text.startswith("ERR"):
                raise RuntimeError(f"{context}: {text}")

            if text == expected_text:
                return

        raise TimeoutError(f"{context}: timed out waiting for '{expected_text}'")

    def rx_read_stream_sample_blocking(self, timeout_s: float = 0.75):
        deadline = time.time() + timeout_s
        with self.rx_serial_lock:
            while time.time() < deadline:
                line = readline_str(self.rx)
                if not line:
                    continue

                text = line.strip()
                if not text:
                    continue

                if text.startswith("ERR"):
                    raise RuntimeError(text)

                values = self.rx_parse_values(line)
                if values is not None:
                    return values

        raise TimeoutError("RX stream sample timed out")

    def rx_start_stream_blocking(self, timeout_s: float = 1.0):
        with self.rx_serial_lock:
            try:
                self.rx.reset_input_buffer()
            except Exception:
                pass

            self.rx.write(b"STREAM_START\n")
            self.rx.flush()

            deadline = time.time() + timeout_s
            while time.time() < deadline:
                line = readline_str(self.rx)
                if not line:
                    continue

                text = line.strip()
                if not text:
                    continue

                if text.startswith("ERR"):
                    raise RuntimeError(text)

                values = self.rx_parse_values(line)
                if values is not None:
                    self.rx_streaming = True
                    return values

        raise TimeoutError("STREAM_START timed out")

    def rx_stop_stream_blocking(self, timeout_s: float = 1.0):
        with self.rx_serial_lock:
            self.rx.write(b"STREAM_STOP\n")
            self.rx.flush()
            self._rx_wait_for_line_blocking("Done", timeout_s=timeout_s, context="STREAM_STOP")
        self.rx_streaming = False

    def rx_set_threshold(self, threshold: int, timeout_s: float = 2.0):
        clamped_threshold = max(0, min(4095, int(round(threshold))))
        with self.rx_serial_lock:
            self.rx.write(b"SET_THRESHOLD\n")
            self.rx.flush()
            self.rx.write(f"{clamped_threshold}\n".encode("utf-8"))
            self.rx.flush()
            self._rx_wait_for_line_blocking("Done", timeout_s=timeout_s, context="SET_THRESHOLD")
        self.rx_threshold_value = clamped_threshold
        return clamped_threshold

    def rx_apply_saved_decode_settings(self):
        detector_index = int(getattr(self, "rx_decode_detector_index", 1))
        applied_detector_index = self.rx_set_decode_pd(detector_index)

        threshold_value = getattr(self, "rx_threshold_value", None)
        if threshold_value is None:
            return applied_detector_index, None

        applied_threshold = self.rx_set_threshold(threshold_value)
        return applied_detector_index, applied_threshold

    def rx_set_decode_pd(self, detector_index: int, timeout_s: float = 2.0):
        clamped_detector_index = max(0, min(int(detector_index), int(getattr(self, "rx_detector_count", 1)) - 1))
        with self.rx_serial_lock:
            self.rx.write(b"SET_DECODE_PD\n")
            self.rx.flush()
            self.rx.write(f"{clamped_detector_index}\n".encode("utf-8"))
            self.rx.flush()
            self._rx_wait_for_line_blocking("Done", timeout_s=timeout_s, context="SET_DECODE_PD")

        self.rx_decode_detector_index = clamped_detector_index
        channel_number = getattr(self, "rx_pd_to_channel", {}).get(clamped_detector_index)
        if channel_number is not None:
            self.rx_decode_channel_number = int(channel_number)

        return clamped_detector_index

    def rx_clear_line_queue(self):
        line_queue = getattr(self, "rx_line_queue", None)
        if line_queue is None:
            return

        try:
            while True:
                line_queue.get_nowait()
        except queue.Empty:
            pass

    def rx_start_reader_thread(self):
        reader_thread = getattr(self, "rx_reader_thread", None)
        if reader_thread is not None and reader_thread.is_alive():
            return

        self.rx_reader_stop.clear()
        self.rx_reader_thread = threading.Thread(
            target=self.rx_reader_loop,
            name="rx-reader",
            daemon=True,
        )
        self.rx_reader_thread.start()

    def rx_stop_reader_thread(self, join_timeout_s=1.0):
        reader_thread = getattr(self, "rx_reader_thread", None)
        stop_event = getattr(self, "rx_reader_stop", None)
        if reader_thread is None or stop_event is None:
            return

        stop_event.set()
        if reader_thread.is_alive():
            reader_thread.join(timeout=join_timeout_s)
        self.rx_reader_thread = None

    def rx_reader_loop(self):
        while not self.rx_reader_stop.is_set():
            if not getattr(self, "rx_polling", False):
                time.sleep(getattr(self, "rx_reader_idle_sleep_s", 0.01))
                continue

            try:
                with self.rx_serial_lock:
                    if not self.rx_polling or self.rx_reader_stop.is_set():
                        continue
                    line = readline_str(self.rx)
            except Exception:
                if self.rx_reader_stop.is_set():
                    break
                time.sleep(getattr(self, "rx_reader_idle_sleep_s", 0.01))
                continue

            if line:
                self.rx_line_queue.put(line)

    def rx_reset_plot_data(self):
        self.rx_history.clear()
        spd_history = getattr(self, "spd_history", None)
        if spd_history is not None:
            spd_history.clear()
        self.rx_plot_dirty = False
        self.spd_plot_dirty = False
        self.rx_next_plot_update = 0.0
        for line in self.rx_lines:
            if line is not None:
                line.set_data([], [])
        for line in getattr(self, "spd_lines", []):
            if line is not None:
                line.set_data([], [])
        hist_lines = getattr(self, "qutag_hist_lines", None)
        if hist_lines is None:
            hist_lines = [getattr(self, "qutag_hist_line", None)]
        for hist_line in hist_lines:
            if hist_line is not None:
                hist_line.set_data([], [])
        for axis in getattr(self, "rx_axes", []):
            axis.set_xlim(0, getattr(self, "rx_plot_window_s", 40.0))
            axis.set_ylim(0, 4096)
        hist_axis = getattr(self, "qutag_hist_axis", None)
        if hist_axis is not None:
            hist_axis.set_xlim(0, self.qutag_hist_x_max_ns())
            hist_axis.set_ylim(0, 1)
        for canvas in getattr(self, "rx_canvases", []):
            canvas.draw_idle()

    def rx_flush_log_lines(self, force=False):
        pending_lines = getattr(self, "rx_pending_log_lines", None)
        if not pending_lines:
            return

        now = time.monotonic()
        next_flush = getattr(self, "rx_next_log_flush", 0.0)
        if not force and now < next_flush:
            return

        self.rx_log.insert("end", "\n".join(pending_lines) + "\n")
        pending_lines.clear()

        max_lines = getattr(self, "rx_log_max_lines", 0)
        if max_lines > 0:
            line_count = int(self.rx_log.index("end-1c").split(".")[0])
            overflow = line_count - max_lines
            if overflow > 0:
                self.rx_log.delete("1.0", f"{overflow + 1}.0")

        self.rx_log.see("end")
        self.rx_next_log_flush = now + getattr(self, "rx_log_flush_interval_s", 0.1)

    def rx_reset_runtime_state(self):
        self.rx_plot_dirty = False
        self.spd_plot_dirty = False
        self.qutag_hist_plot_dirty = False
        self.rx_next_plot_update = 0.0
        self.spd_next_poll = 0.0
        self.qutag_hist_next_poll = 0.0
        self.rx_next_log_flush = 0.0
        self.rx_clear_line_queue()
        pending_lines = getattr(self, "rx_pending_log_lines", None)
        if pending_lines is not None:
            pending_lines.clear()
        spd_history = getattr(self, "spd_history", None)
        if spd_history is not None:
            spd_history.clear()

    def rx_poll(self):
        new_data = False
        max_lines = getattr(self, "rx_max_lines_per_poll", 200)
        lines_processed = 0

        while lines_processed < max_lines:
            try:
                line = self.rx_line_queue.get_nowait()
            except queue.Empty:
                break

            values = self.rx_parse_values(line)
            if values is not None:
                values = self.rx_apply_pd_background_offsets(values)
                self.rx_pending_log_lines.append(self.rx_format_values_for_log(values))
                self.rx_add_sample(values)
                new_data = True
            else:
                stripped_line = line.rstrip()
                if stripped_line:
                    self.rx_pending_log_lines.append(stripped_line)

            lines_processed += 1

        if new_data:
            self.rx_plot_dirty = True

        self.rx_flush_log_lines()

        now = time.monotonic()
        if getattr(self, "rx_streaming", False) and now >= getattr(self, "spd_next_poll", 0.0):
            if self.spd_collect_sample():
                self.spd_plot_dirty = True
            self.spd_next_poll = now + getattr(self, "spd_poll_interval_s", 0.1)

        if now >= getattr(self, "qutag_hist_next_poll", 0.0):
            if self.qutag_hist_collect_sample():
                self.qutag_hist_plot_dirty = True
            self.qutag_hist_next_poll = now + self.qutag_hist_integration_time_s()

        plot_dirty = (
            self.rx_plot_dirty
            or getattr(self, "spd_plot_dirty", False)
            or getattr(self, "qutag_hist_plot_dirty", False)
        )
        if plot_dirty and now >= getattr(self, "rx_next_plot_update", 0.0):
            if self.rx_plot_dirty:
                self.rx_update_plot()
                self.rx_plot_dirty = False
            if getattr(self, "spd_plot_dirty", False):
                self.spd_update_plot()
                self.spd_plot_dirty = False
            if getattr(self, "qutag_hist_plot_dirty", False):
                self.qutag_hist_update_plot()
                self.qutag_hist_plot_dirty = False
            self.rx_next_plot_update = now + getattr(self, "rx_plot_interval_s", 0.1)

        try:
            self.rx_win.after(20, self.rx_poll)
        except Exception:
            pass

    def rx_stream_start(self):
        self.rx_streaming = True
        self.rx_reset_runtime_state()
        self.rx_reset_plot_data()
        self.rx_start_reader_thread()
        try:
            with self.rx_serial_lock:
                self.rx.reset_input_buffer()
        except Exception:
            pass
        self.rx_log_print("[RX] STREAM_START")
        try:
            with self.rx_serial_lock:
                self.rx.write(b"STREAM_START\n")
                self.rx.flush()
        except Exception:
            pass

    def rx_stream_stop(self):
        self.rx_streaming = False
        self.rx_flush_log_lines(force=True)
        self.rx_log_print("[RX] STREAM_STOP")
        try:
            with self.rx_serial_lock:
                self.rx.write(b"STREAM_STOP\n")
                self.rx.flush()
        except Exception:
            pass

    def rx_decode_mode(self):
        detector_index = int(getattr(self, "rx_decode_detector_index", 1))
        channel_number = getattr(self, "rx_pd_to_channel", {}).get(detector_index)
        threshold_value = getattr(self, "rx_threshold_value", None)

        if channel_number is None:
            target_label = f"PD{detector_index}"
        else:
            target_label = f"Channel {int(channel_number)} (PD{detector_index})"

        if threshold_value is None:
            self.rx_log_print(f"[RX] DECODE_MODE on {target_label} (Arduino will block; reconnect to exit)")
        else:
            self.rx_log_print(
                f"[RX] DECODE_MODE on {target_label} with threshold {int(threshold_value)} "
                f"(Arduino will block; reconnect to exit)"
            )
        try:
            with self.rx_serial_lock:
                self.rx.write(b"DECODE_MODE\n")
                self.rx.flush()
        except Exception:
            pass

    def rx_qdcp_decode_mode(self):
        detector_index = int(getattr(self, "rx_decode_detector_index", 1))
        channel_number = getattr(self, "rx_pd_to_channel", {}).get(detector_index)
        threshold_value = getattr(self, "rx_threshold_value", None)

        if channel_number is None:
            target_label = f"PD{detector_index}"
        else:
            target_label = f"Channel {int(channel_number)} (PD{detector_index})"

        if threshold_value is None:
            self.rx_log_print(f"[RX] QDCP_DECODE_MODE on {target_label} (Arduino will block; reconnect to exit)")
        else:
            self.rx_log_print(
                f"[RX] QDCP_DECODE_MODE on {target_label} with threshold {int(threshold_value)} "
                f"(Arduino will block; reconnect to exit)"
            )
        try:
            with self.rx_serial_lock:
                self.rx.write(b"QDCP_DECODE_MODE\n")
                self.rx.flush()
        except Exception:
            pass

    def rx_reconnect(self):
        self.rx_polling = False
        self.rx_stop_reader_thread()
        try:
            with self.rx_serial_lock:
                self.rx.close()
        except Exception:
            pass

        time.sleep(0.5)
        with self.rx_serial_lock:
            self.rx = open_arduino(self.rx_port, 115200, 0.05)
            while True:
                line = readline_str(self.rx)
                if line.strip() == "RX Ready.":
                    break
        self.rx_streaming = False
        self.rx_reset_runtime_state()
        self.rx_reset_plot_data()

        try:
            applied_detector_index, applied_threshold = self.rx_apply_saved_decode_settings()
            channel_number = getattr(self, "rx_pd_to_channel", {}).get(applied_detector_index)
            if applied_threshold is None:
                self.rx_log_print(f"[RX] Restored decode PD{applied_detector_index} after reconnect.")
            elif channel_number is None:
                self.rx_log_print(
                    f"[RX] Restored decode PD{applied_detector_index}, threshold {int(applied_threshold)} after reconnect."
                )
            else:
                self.rx_log_print(
                    f"[RX] Restored decode Channel {int(channel_number)} (PD{applied_detector_index}), "
                    f"threshold {int(applied_threshold)} after reconnect."
                )
        except Exception as exc:
            self.rx_log_print(f"[RX] Warning: could not restore decode settings after reconnect: {exc}")

        self.rx_polling = True
        self.rx_start_reader_thread()
        self.rx_log_print(f"[RX] Reconnected to {self.rx_port} (Ready.)")

    def rx_parse_values(self, line: str):
        parts = line.strip().split()
        expected_count = getattr(self, "rx_detector_count", len(self.rx_lines))
        if len(parts) < expected_count:
            return None
        try:
            values = [float(part) for part in parts[:expected_count]]
        except ValueError:
            return None

        return values

    def rx_format_values_for_log(self, values):
        return " ".join(f"{float(value):g}" for value in values)

    def rx_get_pd_background_offset(self, pd_index: int):
        background_offsets = getattr(self, "rx_pd_background_offsets", {})
        return float(background_offsets.get(int(pd_index), 0.0))

    def rx_apply_pd_background_offset_value(self, value, pd_index: int):
        return int(max(0.0, value - self.rx_get_pd_background_offset(pd_index)))

    def rx_apply_pd_background_offsets(self, values, start_pd_index: int = 0):
        return [
            self.rx_apply_pd_background_offset_value(float(value), int(start_pd_index) + index)
            for index, value in enumerate(values)
        ]

    def rx_add_sample(self, values):
        now = time.time()
        self.rx_history.append((now, values))

        cutoff = now - getattr(self, "rx_plot_window_s", 30.0)
        while self.rx_history and self.rx_history[0][0] < cutoff:
            self.rx_history.popleft()

    def spd_collect_sample(self):
        time_tagger = getattr(self, "time_tagger", None)
        hydraharp = getattr(self, "hydraharp", None)
        time_tagger_indices = list(getattr(self, "spd_counter_indices", []))
        hydraharp_indices = list(getattr(self, "hydraharp_channel_indices", []))
        unavailable = float("nan")
        time_tagger_values = [unavailable] * len(time_tagger_indices)
        hydraharp_values = [unavailable] * len(hydraharp_indices)
        has_sample = False

        if time_tagger is not None:
            try:
                data = self.read_time_tagger_counts()
                if data is not None:
                    self.time_tagger_latest_counts = dict(data)
                    self.time_tagger_latest_counts_monotonic = time.monotonic()
                    time_tagger_values = [
                        float(data[channel]) for channel in time_tagger_indices
                    ]
                    has_sample = True
                self.spd_time_tagger_error_logged = False
            except (IndexError, KeyError, TypeError, ValueError) as exc:
                log_print = getattr(self, "rx_log_print", None)
                if log_print is not None and not getattr(self, "spd_time_tagger_error_logged", False):
                    log_print(f"[Time Tagger X] live count read failed: {exc}")
                    self.spd_time_tagger_error_logged = True
            except Exception as exc:
                log_print = getattr(self, "rx_log_print", None)
                if log_print is not None and not getattr(self, "spd_time_tagger_error_logged", False):
                    log_print(f"[Time Tagger X] live count read failed: {exc}")
                    self.spd_time_tagger_error_logged = True

        if hydraharp is not None and not getattr(self, "hydraharp_raw_recording", False):
            try:
                counts_all, _times = hydraharp.timeTrace.getData()
                # snAPI returns counts per second.  Scale its rates to the
                # Time Tagger X Counter bin so both plots show counts per bin.
                count_bin_s = max(
                    0.0,
                    float(getattr(self, "time_tagger_count_bin_width_ms", 50.0)) / 1000.0,
                )
                hydraharp_values = [
                    float(counts_all[channel_index, -1]) * count_bin_s
                    for channel_index in hydraharp_indices
                ]
                has_sample = True
                self.spd_hydraharp_error_logged = False
            except (IndexError, TypeError, ValueError) as exc:
                log_print = getattr(self, "rx_log_print", None)
                if log_print is not None and not getattr(self, "spd_hydraharp_error_logged", False):
                    log_print(f"[HydraHarp] time-trace read failed: {exc}")
                    self.spd_hydraharp_error_logged = True
            except Exception as exc:
                log_print = getattr(self, "rx_log_print", None)
                if log_print is not None and not getattr(self, "spd_hydraharp_error_logged", False):
                    log_print(f"[HydraHarp] time-trace read failed: {exc}")
                    self.spd_hydraharp_error_logged = True

        if not has_sample:
            return False

        self.spd_add_sample(time_tagger_values + hydraharp_values)
        return True

    def spd_add_sample(self, values):
        spd_history = getattr(self, "spd_history", None)
        if spd_history is None:
            return

        now = time.time()
        spd_history.append((now, values))

        cutoff = now - getattr(self, "rx_plot_window_s", 30.0)
        while spd_history and spd_history[0][0] < cutoff:
            spd_history.popleft()

    def spd_update_plot(self):
        spd_history = getattr(self, "spd_history", None)
        if not spd_history:
            return

        spd_lines = getattr(self, "spd_lines", [])
        plot_axes = {line.axes for line in spd_lines if line is not None}
        plot_axes = {
            axis for axis in plot_axes
            if self.rx_axis_redraw_enabled(axis)
        }
        if not plot_axes:
            return

        start_time = spd_history[0][0]
        xs = [timestamp - start_time for (timestamp, _values) in spd_history]
        max_y = 0.0

        for index, line in enumerate(spd_lines):
            if line is None or line.axes not in plot_axes:
                continue

            ys = [values[index] for (_timestamp, values) in spd_history]
            line.set_data(xs, ys)

            finite_ys = [value for value in ys if value == value]
            if finite_ys:
                max_y = max(max_y, max(finite_ys))

        y_max = max(4096.0, max_y * 1.1)
        for axis in plot_axes:
            axis.set_xlim(0, getattr(self, "rx_plot_window_s", 30.0))
            axis.set_ylim(0, y_max)

        self.rx_draw_axes(plot_axes)

    def qutag_hist_direction_label(self):
        start_channel = int(getattr(self, "qutag_hist_start_channel", 1))
        stop_channel = int(getattr(self, "qutag_hist_stop_channel", 2))
        return f"{start_channel}->{stop_channel}"

    def qutag_hist_get_pairs(self):
        pairs = getattr(self, "qutag_hist_pairs", None)
        if pairs:
            return [(int(start), int(stop)) for start, stop in pairs]

        start_channel = int(getattr(self, "qutag_hist_start_channel", 1))
        stop_channel = int(getattr(self, "qutag_hist_stop_channel", 2))
        return [(start_channel, stop_channel)]

    def qutag_hist_get_pair_options(self):
        options = getattr(self, "qutag_hist_pair_options", None)
        if options:
            return [(int(start), int(stop)) for start, stop in options]

        return [(1, 2)]

    def qutag_hist_pair_option_label(self, hist_pair):
        start_channel, stop_channel = hist_pair
        return f"{int(start_channel)}-{int(stop_channel)}"

    def qutag_hist_parse_pair_option_label(self, pair_text):
        text = str(pair_text).strip().replace(" ", "").replace("->", "-")
        start_text, stop_text = text.split("-", 1)
        return int(start_text), int(stop_text)

    def qutag_hist_plot_label(self):
        labels = [self.qutag_hist_line_label(hist_pair) for hist_pair in self.qutag_hist_get_pairs()]
        return "Time Tagger X Histogram " + ", ".join(labels)

    def qutag_hist_line_label(self, hist_pair=None):
        if hist_pair is None:
            start_channel = int(getattr(self, "qutag_hist_start_channel", 1))
            stop_channel = int(getattr(self, "qutag_hist_stop_channel", 2))
        else:
            start_channel, stop_channel = hist_pair
        return f"Ch {start_channel} -> Ch {stop_channel}"

    def qutag_hist_set_pairs(self, hist_pairs, configure=True):
        normalized_pairs = [(int(start), int(stop)) for start, stop in hist_pairs]
        if not normalized_pairs:
            raise ValueError("Choose at least one coincidence histogram case.")

        allowed_pairs = set(self.qutag_hist_get_pair_options())
        invalid_pairs = [pair for pair in normalized_pairs if pair not in allowed_pairs]
        if invalid_pairs:
            invalid_text = ", ".join(self.qutag_hist_pair_option_label(pair) for pair in invalid_pairs)
            raise ValueError(f"Unsupported histogram case: {invalid_text}")

        if len(set(normalized_pairs)) != len(normalized_pairs):
            raise ValueError("Choose different coincidence histogram cases.")

        self.qutag_hist_pairs = normalized_pairs
        self.qutag_hist_start_channel, self.qutag_hist_stop_channel = normalized_pairs[0]
        self.qutag_hist_configured = False
        self.qutag_hist_data = {}
        self.qutag_hist_stats = {}
        self.qutag_hist_index = {}

        if hasattr(self, "rx_plot_labels"):
            self.rx_plot_labels["hist"] = self.qutag_hist_plot_label()

        axis = getattr(self, "qutag_hist_axis", None)
        lines = list(getattr(self, "qutag_hist_lines", []))
        if axis is not None:
            while len(lines) < len(normalized_pairs):
                line, = axis.plot([], [], label="")
                lines.append(line)

        for line_index, line in enumerate(lines):
            if line_index < len(normalized_pairs):
                line.set_data([], [])
                line.set_label(self.qutag_hist_line_label(normalized_pairs[line_index]))
                line.set_visible(True)
            else:
                line.set_data([], [])
                line.set_visible(False)

        self.qutag_hist_lines = lines
        self.qutag_hist_line = lines[0] if lines else None

        if axis is not None:
            legend = axis.legend(
                loc="upper left",
                frameon=False,
                fontsize=8,
                borderaxespad=0.2,
                handlelength=1.2,
                handletextpad=0.4,
                labelspacing=0.2,
            )
            if hasattr(self, "_style_legend"):
                self._style_legend(legend)
            axis.set_xlim(0, self.qutag_hist_x_max_ns())
            axis.set_ylim(0, 1)
            self.rx_draw_axes({axis})

        if configure:
            self.qutag_hist_configure()

    def qutag_hist_integration_time_s(self):
        return max(0.02, float(getattr(self, "qutag_hist_integration_time_ms", 500.0)) / 1000.0)

    def qutag_hist_x_max_ns(self):
        bin_width_ps = float(getattr(self, "qutag_hist_bin_width_ps", 1))
        bin_count = int(getattr(self, "qutag_hist_bin_count", 3000))
        return max(bin_width_ps * max(bin_count, 1) / 1000.0, bin_width_ps / 1000.0)

    def qutag_hist_configure(self):
        if getattr(self, "time_tagger", None) is None:
            return False

        hist_pairs = self.qutag_hist_get_pairs()
        bin_width_ps = int(getattr(self, "qutag_hist_bin_width_ps", 1))
        bin_count = int(getattr(self, "qutag_hist_bin_count", 3000))

        try:
            previous_histograms = getattr(self, "time_tagger_histograms", {})
            self.time_tagger_histograms = {}
            for histogram in previous_histograms.values():
                try:
                    histogram.stop()
                    getattr(self, "time_tagger_marker", lambda message: None)("Acquisition stopped: previous Histogram")
                except Exception:
                    pass

            for start_channel, stop_channel in hist_pairs:
                hist_pair = (start_channel, stop_channel)
                self.time_tagger_histograms[hist_pair] = self.create_time_tagger_histogram(
                    start_channel,
                    stop_channel,
                    bin_width_ps,
                    bin_count,
                )
        except Exception as exc:
            log_print = getattr(self, "rx_log_print", None)
            if log_print is not None and not getattr(self, "qutag_hist_error_logged", False):
                log_print(f"[Time Tagger X Histogram] setup failed: {exc}")
                self.qutag_hist_error_logged = True
            self.qutag_hist_configured = False
            return False

        self.qutag_hist_configured = True
        self.qutag_hist_error_logged = False
        self.qutag_hist_next_poll = time.monotonic() + self.qutag_hist_integration_time_s()
        return True

    def qutag_hist_collect_sample(self):
        if getattr(self, "time_tagger", None) is None:
            return False
        if getattr(self, "qutag_piezoscan_running", False):
            return False
        if not getattr(self, "qutag_hist_configured", False) and not self.qutag_hist_configure():
            return False

        hist_pairs = self.qutag_hist_get_pairs()
        hist_data = {}
        hist_stats = {}
        hist_index = {}
        histograms = []

        try:
            for start_channel, stop_channel in hist_pairs:
                hist_pair = (start_channel, stop_channel)
                histogram = self.time_tagger_histograms[hist_pair]
                hist_data[hist_pair] = histogram.getData()
                hist_index[hist_pair] = histogram.getIndex()
                hist_stats[hist_pair] = {}
                histograms.append(histogram)

            # Each plotted histogram represents exactly one integration
            # interval.  Clearing starts the next interval immediately after
            # this completed data snapshot is taken.
            for histogram in histograms:
                histogram.clear()
        except Exception as exc:
            log_print = getattr(self, "rx_log_print", None)
            if log_print is not None and not getattr(self, "qutag_hist_error_logged", False):
                log_print(f"[Time Tagger X Histogram] data read failed: {exc}")
                self.qutag_hist_error_logged = True
            self.qutag_hist_configured = False
            return False

        self.qutag_hist_error_logged = False
        self.qutag_hist_data = hist_data
        self.qutag_hist_stats = hist_stats
        self.qutag_hist_index = hist_index
        return True

    def qutag_hist_update_plot(self):
        axis = getattr(self, "qutag_hist_axis", None)
        lines = getattr(self, "qutag_hist_lines", None)
        if lines is None:
            lines = [getattr(self, "qutag_hist_line", None)]
        data = getattr(self, "qutag_hist_data", None)
        if axis is None or data is None:
            return
        if not self.rx_axis_redraw_enabled(axis):
            return

        bin_width_ps = float(getattr(self, "qutag_hist_bin_width_ps", 1))
        indices = getattr(self, "qutag_hist_index", {})
        max_y = 0.0
        for hist_pair, line in zip(self.qutag_hist_get_pairs(), lines):
            if line is None:
                continue
            y_values = [float(value) for value in data.get(hist_pair, [])]
            x_values = [
                float(value) / 1000.0
                for value in indices.get(
                    hist_pair,
                    [index * bin_width_ps for index in range(len(y_values))],
                )
            ]
            line.set_data(x_values, y_values)
            if y_values:
                max_y = max(max_y, max(y_values))

        axis.set_title("")
        axis.set_xlabel("")
        axis.set_ylabel("")
        axis.set_xlim(0, self.qutag_hist_x_max_ns())
        axis.set_ylim(0, max(1.0, max_y * 1.1))
        self.rx_draw_axes({axis})

    def qutag_hist_set_integration_time_ms(self, integration_time_ms):
        integration_time_ms = float(integration_time_ms)
        if integration_time_ms <= 0:
            raise ValueError("Hist Integration Time must be > 0 ms.")

        self.qutag_hist_integration_time_ms = integration_time_ms
        self.qutag_hist_next_poll = time.monotonic() + self.qutag_hist_integration_time_s()

        # Restart the active measurement so the next plot contains one full
        # interval at the newly selected integration time.
        for histogram in getattr(self, "time_tagger_histograms", {}).values():
            histogram.clear()

        axis = getattr(self, "qutag_hist_axis", None)
        if axis is not None:
            axis.set_ylabel("")
            axis.set_xlim(0, self.qutag_hist_x_max_ns())
            self.rx_draw_axes({axis})

    def qutag_hist_set_params(self, bin_width_ps, bin_count):
        bin_width_ps = int(bin_width_ps)
        bin_count = int(bin_count)
        if bin_width_ps <= 0:
            raise ValueError("Hist Bin Width must be > 0 ps.")
        if bin_count < 2 or bin_count > 1000000:
            raise ValueError("Hist Bin Count must be in 2..1000000.")

        self.qutag_hist_bin_width_ps = bin_width_ps
        self.qutag_hist_bin_count = bin_count
        self.qutag_hist_configured = False
        self.qutag_hist_data = {}
        self.qutag_hist_stats = {}
        self.qutag_hist_index = {}

        lines = getattr(self, "qutag_hist_lines", [getattr(self, "qutag_hist_line", None)])
        for line in lines:
            if line is not None:
                line.set_data([], [])

        axis = getattr(self, "qutag_hist_axis", None)
        if axis is not None:
            axis.set_xlim(0, self.qutag_hist_x_max_ns())
            axis.set_ylim(0, 1)
            self.rx_draw_axes({axis})

        self.qutag_hist_configure()

    def rx_update_plot(self):
        if not self.rx_history:
            return

        start_time = self.rx_history[0][0]
        xs = [timestamp - start_time for (timestamp, _values) in self.rx_history]
        plot_axes = set()

        for index, line in enumerate(self.rx_lines):
            if line is None:
                continue
            if not self.rx_axis_redraw_enabled(line.axes):
                continue

            ys = [values[index] for (_timestamp, values) in self.rx_history]
            line.set_data(xs, ys)
            plot_axes.add(line.axes)

        for axis in plot_axes:
            axis.set_xlim(0, getattr(self, "rx_plot_window_s", 30.0))
            axis.set_ylim(0, 4096)

        self.rx_draw_axes(plot_axes)

    def rx_axis_redraw_enabled(self, axis):
        axis_plot_key = getattr(self, "rx_axis_plot_key", {})
        plot_key = axis_plot_key.get(axis)
        if plot_key is None:
            return True
        redraw_enabled = getattr(self, "rx_plot_redraw_enabled", {})
        return redraw_enabled.get(plot_key, True)

    def rx_draw_axes(self, axes):
        axis_to_canvas = getattr(self, "rx_axis_to_canvas", {})
        drawn_canvases = set()

        for axis in axes:
            canvas = axis_to_canvas.get(axis)
            if canvas is None or canvas in drawn_canvases:
                continue
            canvas.draw_idle()
            drawn_canvases.add(canvas)
