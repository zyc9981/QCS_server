from qcs_analyzer._implementation import *
import numpy as np
from numba import njit
import matplotlib.pyplot as plt
import time


def u0_estimator(t_ref_0, t_ref_1):
    """Return the fractional global drift of reference stream 1 relative to 0."""
    n = min(len(t_ref_0), len(t_ref_1))
    return (t_ref_1[n - 1] - t_ref_0[n - 1]) / (t_ref_0[n - 1] - t_ref_0[0])


def qcs_ref_estimator(
    tau,
    t,
    # order=5,
    order=1,
    N_sample=None,
    t_eval_ps=None,
):
    """Return residual standard deviation and fitted tau at the requested time.

    The polynomial is fit using seconds centered on the selected samples rather
    than absolute picosecond timestamps.  This keeps the polynomial basis well
    conditioned without changing the physical time represented by the fit.
    """
    if N_sample != None:
        N_sample = int(np.min([np.size(tau), N_sample]))
    else:
        N_sample = np.size(tau)
    if N_sample < 2:
        raise ValueError("tau and t must contain at least two samples for a linear fit.")

    tau = np.asarray(tau).ravel()[:N_sample]
    t = np.asarray(t).ravel()[:N_sample]

    if t_eval_ps == None:
        t_eval_ps = (t[0] + t[-1]) / 2

    t_origin_ps = np.mean(t)
    t_fit_s = (t - t_origin_ps) * 1e-12
    t_eval_fit_s = (t_eval_ps - t_origin_ps) * 1e-12
    fit = np.polyfit(t_fit_s, tau, order)
    tau_at_point = np.polyval(fit, t_eval_fit_s)

    return tau_at_point, np.std(tau - np.polyval(fit, t_fit_s)), N_sample


def qcs_ref_estimator2(
    tau,
    t,
    N_sample=None,
    t_eval_ps=None,
):
    """Return residual standard deviation and fitted tau at the time-span center."""
    if N_sample != None:
        N_sample = int(np.min([np.size(tau), N_sample]))
    else:
        N_sample = np.size(tau)
    if N_sample < 2:
        raise ValueError("tau and t must contain at least two samples for a linear fit.")

    tau = np.asarray(tau).ravel()[:N_sample]
    t = np.asarray(t).ravel()[:N_sample]

    tau_at_point = np.mean(tau)

    return tau_at_point, np.std(tau - tau_at_point), N_sample


def g2_histogram_numba(
    acquisition_a,
    acquisition_b,
    edges,
    *,
    Ta_s,
    chA,
    chB,
    time_bin_ps_init,
):
    """
    Build a coincidence histogram and return a fit_g2-compatible dictionary.

    Requirements:
        - Acquisitions are sorted, finite, 1-D NumPy timestamp arrays.
        - Timestamps and edges use picoseconds.
        - Edges are uniformly spaced.
        - Bins include their lower edge and exclude their upper edge.
    """
    acquisition_a = np.asarray(acquisition_a)
    acquisition_b = np.asarray(acquisition_b)
    edges = np.asarray(edges, dtype=np.float64)

    if acquisition_a.ndim != 1 or acquisition_b.ndim != 1:
        raise ValueError("Acquisitions must be one-dimensional.")

    if edges.ndim != 1 or len(edges) < 2:
        raise ValueError("edges must contain at least two boundaries.")

    widths = np.diff(edges)
    time_bin_ps = widths[0]

    if (
        not np.all(np.isfinite(edges))
        or np.any(widths <= 0)
        or not np.allclose(widths, time_bin_ps, rtol=1e-8, atol=0)
    ):
        raise ValueError("edges must be finite, increasing, and uniform.")

    if not np.isfinite(Ta_s) or Ta_s <= 0:
        raise ValueError("Ta_s must be finite and positive.")

    if not np.isfinite(time_bin_ps_init) or time_bin_ps_init <= 0:
        raise ValueError("time_bin_ps_init must be finite and positive.")

    # Create the compiled helper once, then reuse it on subsequent calls.
    if not hasattr(g2_histogram_numba, "_count_pairs"):

        @njit(fastmath=False)
        def count_pairs(t_a, t_b, bin_edges):
            n_bins = len(bin_edges) - 1
            bin_width = bin_edges[1] - bin_edges[0]
            lower = bin_edges[0]
            upper = bin_edges[-1]

            counts = np.zeros(n_bins, dtype=np.int64)
            n_b = len(t_b)
            j_start = 0

            for i in range(len(t_a)):
                timestamp_a = t_a[i]

                while j_start < n_b:
                    if t_b[j_start] - timestamp_a >= lower:
                        break
                    j_start += 1

                if j_start == n_b:
                    break

                j = j_start
                while j < n_b:
                    delay = t_b[j] - timestamp_a

                    if delay >= upper:
                        break

                    k = int(np.floor((delay - lower) / bin_width))

                    # Resolve floating-point rounding at bin boundaries.
                    k = min(max(k, 0), n_bins - 1)

                    while k > 0 and delay < bin_edges[k]:
                        k -= 1
                    while k < n_bins - 1 and delay >= bin_edges[k + 1]:
                        k += 1

                    counts[k] += 1
                    j += 1

            return counts

        g2_histogram_numba._count_pairs = count_pairs

    counts = g2_histogram_numba._count_pairs(
        acquisition_a, acquisition_b, edges
    )
    centers = edges[:-1] + widths / 2.0

    return {
        "centers": centers.copy(),
        "histogram": counts / time_bin_ps * time_bin_ps_init,
        "histogram_raw": counts.copy(),
        "T_acq_s": float(Ta_s),
        "chA": chA,
        "chB": chB,
    }


def qcs_kernal(
    t_ps,
    ch,
    ch_0=1,
    ch_1=6,
    ch_2=2,
    ch_ref_0=3,
    ch_ref_1=7,

    tau_ps_init=61470,
    tau_rt_ps_init=107407,
    Ta_s_init=0.1,
    tau_span_ps_init=40e3,
    u0_init = 0.0,
    time_bin_ps_init=200,
    Ta_ref_s=0.1,

    Ta_s_max=1,
    Ta_rt_s_max=10,
    time_bin_ps_min=10,
    fit_curve="g2calib_1",
    fit_parameters={
        "height": 40,
        # "gamma": 0.7,
        # "sigma": 0.1,
        "sigma": 100,
        "offset": 0.01,
    },
    # sour_freq=100e6,
    fwhm_calib_ps=1000,
    second_order_fit=False,
    show_fit=False,
    Freq=50e6,
):
    """
    Chronologically scan paired-channel data in adaptive Ta intervals.

    Leave correction, fitting, and Ta-update logic in the marked sections.
    """
    t_ps = np.asarray(t_ps).ravel()
    ch = np.asarray(ch).ravel()

    if len(t_ps) == 0 or len(ch) == 0:
        raise ValueError("t_ps and ch must not be empty.")
    if len(t_ps) != len(ch):
        raise ValueError("t_ps and ch must have the same length.")

    t_end_ps = t_ps / Ta_s_init

    fit_curve = FIT_CURVE_ALIASES.get(str(fit_curve).strip().lower(), str(fit_curve).strip().lower())
    if fit_curve not in FIT_CURVES:
        available = ", ".join(sorted(FIT_CURVES))
        raise ValueError(f"Unknown fit_curve '{fit_curve}'. Choose one of: {available}.")
    try:
        fit_parameter_guess = {} if fit_parameters is None else dict(fit_parameters)
    except (TypeError, ValueError) as exc:
        raise TypeError("fit_parameters must be a dictionary of initial fit values.") from exc

    # Keep each channel chronological.
    t_0 = np.sort(t_ps[ch == ch_0])
    t_ref_0 = np.sort(t_ps[ch == ch_ref_0])
    t_1 = np.sort(t_ps[ch == ch_1])
    t_ref_1 = np.sort(t_ps[ch == ch_ref_1])
    t_2 = np.sort(t_ps[ch == ch_2])

    if len(t_0) == 0 or len(t_1) == 0 or len(t_2) == 0:
        raise ValueError("All channels need at least one event.")

    # State carried from one adaptive interval to the next.
    state = {
        "Ta_s": float(Ta_s_init),
        "Ta_rt_s": float(Ta_s_init),
        "time_bin_ps": float(time_bin_ps_init),
        "time_bin_rt_ps": float(time_bin_ps_init),
        "tau_ps": float(tau_ps_init),
        "tau_rt_ps": float(tau_rt_ps_init),
        "epsilon_ps": float(0.0),
        # "tau_span_ps": float(tau_span_ps_init),
        "tau_span_ps": np.rint(float(1e12)/Freq) * 1.6,
        "u0": float(u0_init),
        "T_lastmid_ps": float(0.0),
        "T_lastu0_ps": float(0.0),
        "T_rt_lastmid_ps": float(0.0),
        "u0_rt": float(0.0),

        # Second_order_fit parameters
        "u1": float(0.0),
        # "u0_lastmid": float(0.0)
    }

    # Temporary: calculate u0_init from classical recorded pulses
    state["u0"] = u0_estimator(t_ref_0, t_ref_1)
    # state["u0_lastmid"] = state["u0"]
    # print(state["u0"])

    T_file_s = (np.max(t_ps) - np.min(t_ps)) * 1e-12
    buffer_size = int(np.floor(T_file_s / Ta_s_init))
    point_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ps_records = np.full(buffer_size, np.nan, dtype=float)
    tau_rt_ps_records = np.full(buffer_size, np.nan, dtype=float)
    epsilon_ps_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ref_ps_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ref_uncertainty_ps_records = np.full(buffer_size, np.nan, dtype=float)
    Ta_s_records = np.full(buffer_size, np.nan, dtype=float)
    Ta_rt_s_records = np.full(buffer_size, np.nan, dtype=float)
    fwhm_records = np.full(buffer_size, np.nan, dtype=float)
    sigma_records = np.full(buffer_size, np.nan, dtype=float)
    sigma_rt_records = np.full(buffer_size, np.nan, dtype=float)
    tau_s_uncertainty_records = np.full(buffer_size, np.nan, dtype=float)
    CC_count_records = np.full(buffer_size, np.nan, dtype=float)
    CC_count_rt_records = np.full(buffer_size, np.nan, dtype=float)
    # S_peak_records = np.full(buffer_size, np.nan, dtype=float)
    # fwhm_uncertainty_records = np.full(buffer_size, np.nan, dtype=float)
    coincidence_records = np.full(buffer_size, np.nan, dtype=float)
    u0_records = np.full(buffer_size, np.nan, dtype=float)
    u1_records = np.full(buffer_size, np.nan, dtype=float)

    # fit_results = [None] * buffer_size
    # g2_results = [None] * buffer_size
    # errors = [None] * buffer_size

    acquisition_index = 0
    acquisition_results = []

    window_start_ps = min(t_0[0], t_1[0], t_2[0])
    recording_end_ps = max(t_0[-1], t_1[-1], t_2[-1])

    window_rt_start_ps = window_start_ps
    flag_rt = False

    index_0 = 0
    index_0rt = 0
    index_1 = 0
    index_2 = 0
    index_ref = 0

    while window_start_ps < recording_end_ps:
        # cpu_time_start = time.time()

        Ta_s = state["Ta_s"]
        Ta_rt_s = state["Ta_rt_s"]

        remaining_s = (recording_end_ps - window_start_ps) * 1e-12
        if remaining_s < Ta_s:
            # Do not fit, correct, save, or update the state for this partial tail.
            break

        if not np.isfinite(Ta_s) or Ta_s <= 0:
            raise ValueError("State Ta_s must be positive.")

        if not np.isfinite(Ta_rt_s) or Ta_rt_s <= 0:
                    raise ValueError("State Ta_rt_s must be positive.")

        # ------------------------------------------------------------
        # STEP 1:
        # Data acquisition and correction
        # ------------------------------------------------------------
        window_end_ps = window_start_ps + Ta_s * 1e12
        if int(np.rint((window_end_ps - window_rt_start_ps) / (Ta_s_init * 1e12))) * Ta_s_init >= Ta_rt_s:
            # state["Ta_rt_s"] = window_end_ps - window_rt_start_ps
            end_index_2 = np.searchsorted(t_2, window_end_ps, side="left")
            window_rt_start_ps = window_end_ps
            flag_rt = True

        # Collect only events inside this chronological interval.
        end_index_0 = np.searchsorted(t_0, window_end_ps, side="left")
        end_index_1 = np.searchsorted(t_1, window_end_ps, side="left")

        acquisition_t_0 = t_0[index_0:end_index_0]
        acquisition_t_0rt = t_0[index_0rt:end_index_0]
        acquisition_t_1 = t_1[index_1:end_index_1]
        x_ps = acquisition_t_1 - acquisition_t_1[0]
        T_ps = state["Ta_s"] * 1e12 / 2
        u0s = state["u0"] + state["u1"] *  (acquisition_t_1[0] - state["T_lastu0_ps"])
        # u0s = state["u0"] + state["u1"] *  ((window_start_ps + window_end_ps) / 2 - state["T_lastu0_ps"])
        # acquisition_t_1 = acquisition_t_1 - (acquisition_t_1 - acquisition_t_1[0]) * state["u0"] + state["u0"] * state["Ta_s"] * 1e12 / 2
        # acquisition_t_1 = acquisition_t_1 - state["u0"] * x_ps - 0.5 * state["u1"] * (x_ps ** 2) + state["u0"] * T_ps + 0.5 * state["u1"] * (T_ps ** 2)
        acquisition_t_1 = acquisition_t_1 - u0s * x_ps - 0.5 * state["u1"] * (x_ps ** 2) + u0s * T_ps + 0.5 * state["u1"] * (T_ps ** 2)
        acquisition_t_2 = t_2[index_2:end_index_2]
        # acquisition_t_2 = acquisition_t_2 - (acquisition_t_2 - acquisition_t_2[0]) * state["u0_rt"] + state["u0_rt"] * state["Ta_rt_s"] * 1e12 / 2

        acquisition = {
            "start_ps": window_start_ps,
            "end_ps": window_end_ps,
            "start_rt_ps": window_rt_start_ps,
            # "end_rt_ps": window_end_ps,
            "point_ps": (window_start_ps + window_end_ps) / 2,
            "Ta_s": Ta_s,
            "t_0": acquisition_t_0,
            "t_0rt": acquisition_t_0rt,
            "t_1": acquisition_t_1,
            "t_2": acquisition_t_2,
            # "state_before": state.copy(),
        }

        # Collect ref events inside this chronological interval.

        ref_width_ps = Ta_ref_s * 1e12
        available_start_ps = t_ref_0[0]
        available_end_ps = t_ref_0[-1]

        if ref_width_ps >= available_end_ps - available_start_ps:
            # Reference duration is shorter than the requested window.
            ref_start_ps = available_start_ps
            ref_end_ps = available_end_ps
        else:
            # Center when possible; shift inward near either boundary.
            ref_center_ps = np.clip(
                acquisition["point_ps"],
                available_start_ps + ref_width_ps / 2,
                available_end_ps - ref_width_ps / 2,
            )
            ref_start_ps = ref_center_ps - ref_width_ps / 2
            ref_end_ps = ref_center_ps + ref_width_ps / 2

        ref_start_index = np.searchsorted(t_ref_0, ref_start_ps, side="left")
        ref_end_index = np.searchsorted(t_ref_0, ref_end_ps, side="right")

        acquisition_ref_0 = t_ref_0[ref_start_index:ref_end_index]
        acquisition_ref_1 = t_ref_1[ref_start_index:ref_end_index]

        # ref_start_ps = acquisition["point_ps"] - Ta_ref_s * 1e12 / 2
        # ref_end_ps = acquisition["point_ps"] + Ta_ref_s * 1e12 / 2
        # ref_start_index = np.searchsorted(t_ref_0, ref_start_ps, side="left")
        # ref_end_index = np.searchsorted(t_ref_0, ref_end_ps, side="left")
        # acquisition_ref_0 = t_ref_0[ref_start_index:ref_end_index]
        # acquisition_ref_1 = t_ref_1[ref_start_index:ref_end_index]
        # # acquisition_ref_1 = acquisition_ref_1 - (acquisition_ref_1 - acquisition_ref_1[0]) * state["u0"]

        # ------------------------------------------------------------
        # STEP 2:
        # Create start-stop g2 processing results.
        # ------------------------------------------------------------
        tau_predicted_ps = state["tau_ps"] + (acquisition["point_ps"] - state["T_lastmid_ps"]) * state["u0"]
        t_lo = tau_predicted_ps - state["tau_span_ps"] * 0.5
        t_hi = tau_predicted_ps + state["tau_span_ps"] * 0.5
        t_lo = tau_ps_init - state["tau_span_ps"] * 0.5
        t_hi = tau_ps_init + state["tau_span_ps"] * 0.5
        time_bin_ps = state["time_bin_ps"]
        edges = np.arange(t_lo, t_hi + time_bin_ps, time_bin_ps)
        if len(edges) < 2:
            raise ValueError("Window too small for given bin width. Increase span or decrease bin.")
        centers = edges[:-1] + time_bin_ps / 2
        start_stop_results = np.zeros(len(centers), dtype=np.int64)

        # j_start = 0
        # for t_i in acquisition_t_0:
        #     while j_start < len(acquisition_t_1) and (acquisition_t_1[j_start] - t_i) < t_lo:
        #         j_start += 1
    
        #     j = j_start
        #     while j < len(acquisition_t_1):
        #         dt = acquisition_t_1[j] - t_i
    
        #         if dt > t_hi:
        #             break
    
        #         kbin = int(np.floor((dt - t_lo) / time_bin_ps))
        #         if 0 <= kbin < len(start_stop_results):
        #             start_stop_results[kbin] += 1
        #             # numPairs += 1
    
        #         j += 1
        
        # results = {
        #     "centers": centers.copy(),
        #     "histogram": start_stop_results.copy() / state["time_bin_ps"] * time_bin_ps_init,
        #     "histogram_raw": start_stop_results.copy(),
        #     "T_acq_s": Ta_s,
        #     "chA": ch_0,
        #     "chB": ch_1,
        # }

        results = g2_histogram_numba(
            acquisition_t_0,
            acquisition_t_1,
            edges,
            Ta_s=Ta_s,
            chA=ch_0,
            chB=ch_1,
            time_bin_ps_init=time_bin_ps_init,
        )

        if flag_rt == True:
            tau_rt_predicted_ps = state["tau_rt_ps"]
            # t_lo_rt = tau_rt_predicted_ps - state["tau_span_ps"] * 0.5
            # t_hi_rt = tau_rt_predicted_ps + state["tau_span_ps"] * 0.5
            t_lo_rt = tau_rt_ps_init - state["tau_span_ps"] * 0.5
            t_hi_rt = tau_rt_ps_init + state["tau_span_ps"] * 0.5
            time_bin_rt_ps = state["time_bin_rt_ps"]
            edges_rt = np.arange(t_lo_rt, t_hi_rt + time_bin_rt_ps, time_bin_rt_ps)
            if len(edges_rt) < 2:
                raise ValueError("Window too small for given bin width. Increase span or decrease bin.")
            centers_rt = edges_rt[:-1] + time_bin_rt_ps / 2
            start_stop_results_rt = np.zeros(len(centers_rt), dtype=np.int64)

            # j_start = 0
            # for t_i in acquisition_t_0rt:
            #     while j_start < len(acquisition_t_2) and (acquisition_t_2[j_start] - t_i) < t_lo_rt:
            #         j_start += 1
        
            #     j = j_start
            #     while j < len(acquisition_t_2):
            #         dt = acquisition_t_2[j] - t_i
        
            #         if dt > t_hi_rt:
            #             break
        
            #         kbin = int(np.floor((dt - t_lo_rt) / time_bin_rt_ps))
            #         if 0 <= kbin < len(start_stop_results_rt):
            #             start_stop_results_rt[kbin] += 1
            #             # numPairs += 1
        
            #         j += 1
            
            # results_rt = {
            #     "centers": centers_rt.copy(),
            #     "histogram": start_stop_results_rt.copy() / state["time_bin_rt_ps"] * time_bin_ps_init,
            #     "histogram_raw": start_stop_results_rt.copy(),
            #     "T_acq_s": Ta_rt_s,
            #     "chA": ch_0,
            #     "chB": ch_2,
            # }

            results_rt = g2_histogram_numba(
                acquisition_t_0rt,
                acquisition_t_2,
                edges_rt,
                Ta_s=Ta_rt_s,
                chA=ch_0,
                chB=ch_2,
                time_bin_ps_init=time_bin_ps_init,
            )

        # ------------------------------------------------------------
        # STEP 3:
        # Estimate time offset and frequency skew from g2 fit
        # ------------------------------------------------------------
        try:
            fit_parameters = fit_parameter_guess.copy()
            fit_parameters["center"] = float(
                results["centers"][np.argmax(results["histogram"])]
            )

            fit = fit_g2(
                results,
                curve=fit_curve,
                parameters=fit_parameters,
                # fit_window=state["tau_span_ps"],
                fit_window=np.rint(float(1e12)/Freq) * 1.2,
                show=show_fit,
            )

        except Exception as exc:
            raise

            print(f"Time-dependent g2 fit failed at t_stop={(acquisition["point_ps"]*1e-12):.6g} s: {exc}")
            fit = None
            exc_value = exc
        else:
            exc_value = None

        if flag_rt == True:
            try:
                fit_parameters = fit_parameter_guess.copy()
                fit_parameters["center"] = float(
                    results_rt["centers"][np.argmax(results_rt["histogram"])]
                )
    
                # print(fit_parameters["center"])
                fit_rt = fit_g2(
                    results_rt,
                    curve=fit_curve,
                    parameters=fit_parameters,
                    # fit_window=state["tau_span_ps"],
                    fit_window=np.rint(float(1e12)/Freq) * 1.2,
                    show=show_fit,
                    # show=True,
                )
    
            except Exception as exc:
                raise
    
                print(f"Time-dependent g2 fit failed at t_stop={(acquisition["point_ps"]*1e-12):.6g} s: {exc}")
                fit = None
                exc_value = exc
            else:
                exc_value = None

        # ------------------------------------------------------------
        # STEP 4:
        # Update tau, u0, and records
        # ------------------------------------------------------------

        point_records[acquisition_index] = acquisition["point_ps"]
        tau_ps_records[acquisition_index] = fit["center"]
        tau_rt_ps_records[acquisition_index] = fit_rt["center"]
        Ta_s_records[acquisition_index] = state["Ta_s"]
        Ta_rt_s_records[acquisition_index] = state["Ta_rt_s"]
        epsilon_ps_records[acquisition_index] = fit["center"] - fit_rt["center"] / 2
        # fwhm_records[acquisition_index] = fwhm
        sigma_records[acquisition_index] = fit["parameters"]["sigma"]
        sigma_rt_records[acquisition_index] = fit_rt["parameters"]["sigma"]
        tau_s_uncertainty_records[acquisition_index] = fit["center_uncertainty"]
        # fwhm_uncertainty_records[acquisition_index] = fit["fwhm_uncertainty"]
        CC_count_records[acquisition_index] = np.sum(results["histogram_raw"])
        CC_count_rt_records[acquisition_index] = np.sum(results_rt["histogram_raw"])
        # S_peak_records[acquisition_index] = np.sqrt(np.sum(results["histogram"]) ** 2 * np.size(results["centers"]) / np.size(acquisition_t_0) / np.size(acquisition_t_1))

        tau_ref_ps, tau_ref_uncertainty_ps, N_ref_samples = qcs_ref_estimator(
            acquisition_ref_1 - acquisition_ref_0,
            acquisition_ref_0,
            t_eval_ps=acquisition["point_ps"],
        )

        tau_ref_ps_records[acquisition_index] = tau_ref_ps
        tau_ref_uncertainty_ps_records[acquisition_index] = tau_ref_uncertainty_ps / np.sqrt(N_ref_samples)
        T_u0 = 0.0

        if acquisition_index > 0:
            # udrift = (fit["center"] - tau_predicted_ps) / (acquisition["point_ps"] - state["T_lastmid_ps"])
            # u0 = state["u0"] + udrift
            # state["u0"] = u0

            # First order correction
            u0 = (fit["center"] - state["tau_ps"]) / (acquisition["point_ps"] - state["T_lastmid_ps"])
            udrift = u0 - state["u0"]
            u0_last = state["u0"]
            state["u0"] = u0
            T_u0 = (acquisition["point_ps"] + state["T_lastmid_ps"]) / 2

            # Second order correction
            if second_order_fit:
                # u1 = (state["u0"] - u0_last) / (acquisition["point_ps"] - state["T_lastmid_ps"])
                u1 = (state["u0"] - u0_last) / (T_u0 - state["T_lastu0_ps"])
                state["u1"] = u1
            
            Ta_s = np.round((fwhm_calib_ps / 10 / np.abs(udrift) * 1e-12) / Ta_s_init) * Ta_s_init
            Ta_s = np.min([Ta_s, Ta_s_max])
            state["Ta_s"] = np.max([Ta_s_init, Ta_s])
            # state["time_bin_ps"] = round(time_bin_ps_init / state["Ta_s"] * Ta_s_init / 10.0) * 10.0

            # print(f"Ta_s: {state["Ta_s"]:.2f}")
        else:
            state["Ta_s"] = Ta_s

        state["tau_ps"] = fit["center"]
        state["T_lastmid_ps"] = acquisition["point_ps"]
        state["T_lastu0_ps"] = T_u0
        u0_records[acquisition_index] = state["u0"]
        u1_records[acquisition_index] = state["u1"]

        if flag_rt == True:
            index_0rt = end_index_0
            index_2 = end_index_2

            if acquisition_index > 0:
                T_rt_thismid_ps = (acquisition_t_2[0] + acquisition_t_2[-1]) / 2
                udrift_rt = (fit_rt["center"] - tau_rt_predicted_ps) / (T_rt_thismid_ps - state["T_rt_lastmid_ps"])
                # u0 = state["u0"] + udrift
                # state["u0"] = u0
                
                Ta_rt_s = np.round((fwhm_calib_ps / 10 / np.abs(udrift_rt) * 1e-12) / Ta_s_init) * Ta_s_init
                Ta_rt_s = np.min([Ta_rt_s, Ta_rt_s_max])
                # state["Ta_rt_s"] = np.max([Ta_s_init, Ta_rt_s])
                # state["Ta_rt_s"] = Ta_rt_s
                state["Ta_rt_s"] = state["Ta_s"]
                # time_bin_rt_ps = round(time_bin_ps_init / state["Ta_rt_s"] * Ta_s_init / 10.0) * 10.0
                # state["time_bin_rt_ps"] = np.max([time_bin_rt_ps, time_bin_ps_min])
    
                # print(f"Ta_rt_s: {state["Ta_rt_s"]:.2f}")
            else:
                state["Ta_rt_s"] = Ta_rt_s

            state["tau_rt_ps"] = fit_rt["center"]
            state["T_rt_lastmid_ps"] = (acquisition_t_2[0] + acquisition_t_2[-1]) / 2

        # acquisition_results.append(acquisition)

        # Advance both channel indices and move chronologically onward.
        acquisition_index += 1
        index_0 = end_index_0
        index_1 = end_index_1
        # index_ref = end_index_ref
        window_start_ps = window_end_ps
        flag_rt = False

        # cpu_time_end = time.time()
        # print(f"Acquisition {acquisition_index} processed in {cpu_time_end - cpu_time_start:.4f} seconds.")

    # ------------------------------------------------------------
    # After scan: plot results
    # ------------------------------------------------------------

    # Step 1: remove the first six completed acquisitions from all records.
    # start_index = 6
    start_index = 1
    record_slice = slice(min(start_index, acquisition_index), acquisition_index)
    point_records = point_records[record_slice]
    tau_ps_records = tau_ps_records[record_slice]
    tau_rt_ps_records = tau_rt_ps_records[record_slice]
    epsilon_ps_records = epsilon_ps_records[record_slice]
    tau_ref_ps_records = tau_ref_ps_records[record_slice]
    tau_ref_uncertainty_ps_records = tau_ref_uncertainty_ps_records[record_slice]
    Ta_s_records = Ta_s_records[record_slice]
    Ta_rt_s_records = Ta_rt_s_records[record_slice]
    fwhm_records = fwhm_records[record_slice]
    sigma_records = sigma_records[record_slice]
    sigma_rt_records = sigma_rt_records[record_slice]
    tau_s_uncertainty_records = tau_s_uncertainty_records[record_slice]
    CC_count_records = CC_count_records[record_slice]
    CC_count_rt_records = CC_count_rt_records[record_slice]
    coincidence_records = coincidence_records[record_slice]
    u0_records = u0_records[record_slice]
    u1_records = u1_records[record_slice]
    acquisition_index = len(point_records)

    # Fig1: one way details
    fig1, axes = plt.subplots(5, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_ps_records - tau_ps_records[0], "o-")
    ax1.set_ylabel(r"$\tau$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, Ta_s_records, "o-")
    ax2.set_ylabel(r"$T_a$ (s)")
    ax2.grid(True)

    ax3 = axes[2]
    ax3.plot(point_records * 1e-12, sigma_records, "o-")
    ax3.set_ylabel(r"$\sigma$ (ps)")
    ax3.grid(True)

    ax4 = axes[3]
    ax4.plot(point_records * 1e-12, CC_count_records, "o-")
    ax4.set_ylabel(r"CC")
    ax4.grid(True)

    ax5 = axes[4]
    ax5.plot(point_records * 1e-12, tau_ref_ps_records - tau_ref_ps_records[0], "o-")
    ax5.set_ylabel(r"$\tau_{ref}$ (ps)")
    ax5.grid(True)

    # ax6 = axes[5]
    # ax6.plot(point_records * 1e-12, tau_ref_uncertainty_ps_records, "o-")
    # ax6.set_ylabel(r"$\tau_{ref}$ (ps)")
    # ax6.grid(True)

    # tmp = tau_ref_ps_records - tau_ref_ps_records[0]
    # tmp_idx = np.where(tmp > 100)[0]
    # print(tmp_idx)

    axes[-1].set_xlabel("Record Points (s)")
    fig1.tight_layout()

    # Fig2: two way details
    fig2, axes = plt.subplots(4, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_rt_ps_records - tau_rt_ps_records[0], "o-")
    ax1.set_ylabel(r"$\tau_{rt}$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, Ta_rt_s_records, "o-")
    ax2.set_ylabel(r"$T_{a_rt}$ (s)")
    ax2.grid(True)

    ax3 = axes[2]
    ax3.plot(point_records * 1e-12, sigma_rt_records, "o-")
    ax3.set_ylabel(r"$\sigma_{rt}$ (ps)")
    ax3.grid(True)

    ax4 = axes[3]
    ax4.plot(point_records * 1e-12, CC_count_rt_records, "o-")
    ax4.set_ylabel(r"CC")
    ax4.grid(True)

    axes[-1].set_xlabel("Record Points (s)")
    fig2.tight_layout()

    # Fig3: summaries
    fig3, axes = plt.subplots(2, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    tau_diff = tau_ps_records - tau_ref_ps_records
    # tau_diff = np.delete(tau_diff, tmp_idx)
    # point_records2 = np.delete(point_records, tmp_idx)
    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_diff - np.mean(tau_diff), "o-")
    # ax1.plot(point_records2 * 1e-12, tau_diff - np.mean(tau_diff), "o-")
    ax1.set_ylabel(r"$\tau$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, epsilon_ps_records, "o-")
    ax2.set_ylabel(r"$\epsilon$ (ps)")
    ax2.grid(True)

    # ax3 = axes[2]
    # ax3.plot(point_records * 1e-12, CC_count_records, "o-")
    # ax3.set_ylabel(r"CC")
    # ax3.grid(True)

    axes[-1].set_xlabel("Record Points (s)")
    fig3.tight_layout()

    print(np.std(tau_diff))
    print(np.mean(tau_diff))

    plt.show()

    return {
        "center": point_records,
        "tau": tau_ps_records,
        "tau_rt": tau_rt_ps_records,
        "tau_ref": tau_ref_ps_records,
        "tau_ref_uncertainty": tau_ref_uncertainty_ps_records,
        "tau_diff": tau_diff,
        "epsilon": epsilon_ps_records,
        "Ta": Ta_s_records,
        "Ta_rt": Ta_rt_s_records,
        "sigma": sigma_records,
        "sigma_rt": sigma_rt_records,
        "CC": CC_count_records,
        "CC_rt": CC_count_rt_records,
        "figure1": fig1,
        "figure2": fig2,
        "figure3": fig3,
        # "axes": axes,
    }


def qcs_kernal_sep(
    t1_ps,
    ch1,
    t2_ps,
    ch2,
    ch_0=1,
    ch_1=6,
    ch_2=2,
    ch_ref_0=3,
    ch_ref_1=7,

    tau_ps_init=61470,
    tau_rt_ps_init=107407,
    Ta_s_init=0.1,
    tau_span_ps_init=40e3,
    u0_init = 0.0,
    time_bin_ps_init=200,
    Ta_ref_s=0.1,

    Ta_s_max=1,
    Ta_rt_s_max=10,
    time_bin_ps_min=10,
    fit_curve="g2calib_1",
    fit_parameters={
        "height": 40,
        # "gamma": 0.7,
        # "sigma": 0.1,
        "sigma": 100,
        "offset": 0.01,
    },
    # sour_freq=100e6,
    fwhm_calib_ps=1000,
    second_order_fit=False,
    show_fit=False,
    Freq=50e6,

    T_calib = 0.0,
    T_calib_rt = 0.0,
):
    """
    Chronologically scan paired-channel data in adaptive Ta intervals.

    Leave correction, fitting, and Ta-update logic in the marked sections.
    """
    t1_ps = np.asarray(t1_ps).ravel()
    t2_ps = np.asarray(t2_ps).ravel()
    ch1 = np.asarray(ch1).ravel()
    ch2 = np.asarray(ch2).ravel()

    if len(t1_ps) == 0 or len(ch1) == 0 or len(t2_ps) == 0 or len(ch2) == 0:
        raise ValueError("t1_ps, ch1, t2_ps, and ch2 must not be empty.")
    if len(t1_ps) != len(ch1):
        raise ValueError("t1_ps and ch1 must have the same length.")
    if len(t2_ps) != len(ch2):
        raise ValueError("t2_ps and ch2 must have the same length.")

    # t_end_ps = t1_ps / Ta_s_init

    fit_curve = FIT_CURVE_ALIASES.get(str(fit_curve).strip().lower(), str(fit_curve).strip().lower())
    if fit_curve not in FIT_CURVES:
        available = ", ".join(sorted(FIT_CURVES))
        raise ValueError(f"Unknown fit_curve '{fit_curve}'. Choose one of: {available}.")
    try:
        fit_parameter_guess = {} if fit_parameters is None else dict(fit_parameters)
    except (TypeError, ValueError) as exc:
        raise TypeError("fit_parameters must be a dictionary of initial fit values.") from exc

    # Keep each channel chronological.
    t_0 = np.sort(t1_ps[ch1 == ch_0])
    t_ref_0 = np.sort(t1_ps[ch1 == ch_ref_0])
    t_1 = np.sort(t2_ps[ch2 == ch_1])
    t_ref_1 = np.sort(t2_ps[ch2 == ch_ref_1])
    t_2 = np.sort(t1_ps[ch1 == ch_2])

    if len(t_0) == 0 or len(t_1) == 0 or len(t_2) == 0:
        raise ValueError("All channels need at least one event.")

    # State carried from one adaptive interval to the next.
    state = {
        "Ta_s": float(Ta_s_init),
        "Ta_rt_s": float(Ta_s_init),
        "time_bin_ps": float(time_bin_ps_init),
        "time_bin_rt_ps": float(time_bin_ps_init),
        "tau_ps": float(tau_ps_init),
        "tau_rt_ps": float(tau_rt_ps_init),
        "epsilon_ps": float(0.0),
        "tau_span_ps": float(tau_span_ps_init),
        # "tau_span_ps": np.rint(float(1e12)/Freq) * 1.6,
        "u0": float(u0_init),
        "T_lastmid_ps": float(0.0),
        "T_lastu0_ps": float(0.0),
        "T_rt_lastmid_ps": float(0.0),
        "u0_rt": float(0.0),

        # Second_order_fit parameters
        "u1": float(0.0),
        # "u0_lastmid": float(0.0)
    }

    # Temporary: calculate u0_init from classical recorded pulses
    state["u0"] = u0_estimator(t_ref_0, t_ref_1)
    # state["u0_lastmid"] = state["u0"]
    # print(state["u0"])

    T_file_s = (np.max(t1_ps) - np.min(t1_ps)) * 1e-12
    buffer_size = int(np.floor(T_file_s / Ta_s_init))
    point_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ps_records = np.full(buffer_size, np.nan, dtype=float)
    tau_rt_ps_records = np.full(buffer_size, np.nan, dtype=float)
    epsilon_ps_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ref_ps_records = np.full(buffer_size, np.nan, dtype=float)
    tau_ref_uncertainty_ps_records = np.full(buffer_size, np.nan, dtype=float)
    Ta_s_records = np.full(buffer_size, np.nan, dtype=float)
    Ta_rt_s_records = np.full(buffer_size, np.nan, dtype=float)
    fwhm_records = np.full(buffer_size, np.nan, dtype=float)
    sigma_records = np.full(buffer_size, np.nan, dtype=float)
    sigma_rt_records = np.full(buffer_size, np.nan, dtype=float)
    tau_s_uncertainty_records = np.full(buffer_size, np.nan, dtype=float)
    CC_count_records = np.full(buffer_size, np.nan, dtype=float)
    CC_count_rt_records = np.full(buffer_size, np.nan, dtype=float)
    # S_peak_records = np.full(buffer_size, np.nan, dtype=float)
    # fwhm_uncertainty_records = np.full(buffer_size, np.nan, dtype=float)
    coincidence_records = np.full(buffer_size, np.nan, dtype=float)
    u0_records = np.full(buffer_size, np.nan, dtype=float)
    u1_records = np.full(buffer_size, np.nan, dtype=float)

    # fit_results = [None] * buffer_size
    # g2_results = [None] * buffer_size
    # errors = [None] * buffer_size

    acquisition_index = 0
    acquisition_results = []

    window_start_ps = min(t_0[0], t_1[0], t_2[0])
    recording_end_ps = max(t_0[-1], t_1[-1], t_2[-1])

    window_rt_start_ps = window_start_ps
    flag_rt = False

    index_0 = 0
    index_0rt = 0
    index_1 = 0
    index_2 = 0
    index_ref = 0

    while window_start_ps < recording_end_ps:
        # cpu_time_start = time.time()

        Ta_s = state["Ta_s"]
        Ta_rt_s = state["Ta_rt_s"]

        remaining_s = (recording_end_ps - window_start_ps) * 1e-12
        if remaining_s < Ta_s:
            # Do not fit, correct, save, or update the state for this partial tail.
            break

        if not np.isfinite(Ta_s) or Ta_s <= 0:
            raise ValueError("State Ta_s must be positive.")

        if not np.isfinite(Ta_rt_s) or Ta_rt_s <= 0:
                    raise ValueError("State Ta_rt_s must be positive.")

        # ------------------------------------------------------------
        # STEP 1:
        # Data acquisition and correction
        # ------------------------------------------------------------
        window_end_ps = window_start_ps + Ta_s * 1e12
        if int(np.rint((window_end_ps - window_rt_start_ps) / (Ta_s_init * 1e12))) * Ta_s_init >= Ta_rt_s:
            # state["Ta_rt_s"] = window_end_ps - window_rt_start_ps
            end_index_2 = np.searchsorted(t_2, window_end_ps, side="left")
            window_rt_start_ps = window_end_ps
            flag_rt = True

        # Collect only events inside this chronological interval.
        end_index_0 = np.searchsorted(t_0, window_end_ps, side="left")
        end_index_1 = np.searchsorted(t_1, window_end_ps, side="left")

        acquisition_t_0 = t_0[index_0:end_index_0]
        acquisition_t_0rt = t_0[index_0rt:end_index_0]
        acquisition_t_1 = t_1[index_1:end_index_1]
        x_ps = acquisition_t_1 - acquisition_t_1[0]
        T_ps = state["Ta_s"] * 1e12 / 2
        u0s = state["u0"] + state["u1"] *  (acquisition_t_1[0] - state["T_lastu0_ps"])
        # u0s = state["u0"] + state["u1"] *  ((window_start_ps + window_end_ps) / 2 - state["T_lastu0_ps"])
        # acquisition_t_1 = acquisition_t_1 - (acquisition_t_1 - acquisition_t_1[0]) * state["u0"] + state["u0"] * state["Ta_s"] * 1e12 / 2
        # acquisition_t_1 = acquisition_t_1 - state["u0"] * x_ps - 0.5 * state["u1"] * (x_ps ** 2) + state["u0"] * T_ps + 0.5 * state["u1"] * (T_ps ** 2)
        acquisition_t_1 = acquisition_t_1 - u0s * x_ps - 0.5 * state["u1"] * (x_ps ** 2) + u0s * T_ps + 0.5 * state["u1"] * (T_ps ** 2)
        acquisition_t_2 = t_2[index_2:end_index_2]
        # acquisition_t_2 = acquisition_t_2 - (acquisition_t_2 - acquisition_t_2[0]) * state["u0_rt"] + state["u0_rt"] * state["Ta_rt_s"] * 1e12 / 2

        acquisition = {
            "start_ps": window_start_ps,
            "end_ps": window_end_ps,
            "start_rt_ps": window_rt_start_ps,
            # "end_rt_ps": window_end_ps,
            "point_ps": (window_start_ps + window_end_ps) / 2,
            "Ta_s": Ta_s,
            "t_0": acquisition_t_0,
            "t_0rt": acquisition_t_0rt,
            "t_1": acquisition_t_1,
            "t_2": acquisition_t_2,
            # "state_before": state.copy(),
        }

        # Collect ref events inside this chronological interval.

        ref_width_ps = Ta_ref_s * 1e12
        available_start_ps = t_ref_0[0]
        available_end_ps = t_ref_0[-1]

        if ref_width_ps >= available_end_ps - available_start_ps:
            # Reference duration is shorter than the requested window.
            ref_start_ps = available_start_ps
            ref_end_ps = available_end_ps
        else:
            # Center when possible; shift inward near either boundary.
            ref_center_ps = np.clip(
                acquisition["point_ps"],
                available_start_ps + ref_width_ps / 2,
                available_end_ps - ref_width_ps / 2,
            )
            ref_start_ps = ref_center_ps - ref_width_ps / 2
            ref_end_ps = ref_center_ps + ref_width_ps / 2

        ref_start_index = np.searchsorted(t_ref_0, ref_start_ps, side="left")
        ref_end_index = np.searchsorted(t_ref_0, ref_end_ps, side="right")

        acquisition_ref_0 = t_ref_0[ref_start_index:ref_end_index]
        acquisition_ref_1 = t_ref_1[ref_start_index:ref_end_index]

        # ref_start_ps = acquisition["point_ps"] - Ta_ref_s * 1e12 / 2
        # ref_end_ps = acquisition["point_ps"] + Ta_ref_s * 1e12 / 2
        # ref_start_index = np.searchsorted(t_ref_0, ref_start_ps, side="left")
        # ref_end_index = np.searchsorted(t_ref_0, ref_end_ps, side="left")
        # acquisition_ref_0 = t_ref_0[ref_start_index:ref_end_index]
        # acquisition_ref_1 = t_ref_1[ref_start_index:ref_end_index]
        # # acquisition_ref_1 = acquisition_ref_1 - (acquisition_ref_1 - acquisition_ref_1[0]) * state["u0"]

        # ------------------------------------------------------------
        # STEP 2:
        # Create start-stop g2 processing results.
        # ------------------------------------------------------------
        tau_predicted_ps = state["tau_ps"] + (acquisition["point_ps"] - state["T_lastmid_ps"]) * state["u0"]
        t_lo = tau_predicted_ps - state["tau_span_ps"] * 0.5
        t_hi = tau_predicted_ps + state["tau_span_ps"] * 0.5
        # t_lo = tau_ps_init - state["tau_span_ps"] * 0.5
        # t_hi = tau_ps_init + state["tau_span_ps"] * 0.5
        time_bin_ps = state["time_bin_ps"]
        edges = np.arange(t_lo, t_hi + time_bin_ps, time_bin_ps)
        if len(edges) < 2:
            raise ValueError("Window too small for given bin width. Increase span or decrease bin.")
        centers = edges[:-1] + time_bin_ps / 2
        start_stop_results = np.zeros(len(centers), dtype=np.int64)

        # j_start = 0
        # for t_i in acquisition_t_0:
        #     while j_start < len(acquisition_t_1) and (acquisition_t_1[j_start] - t_i) < t_lo:
        #         j_start += 1
    
        #     j = j_start
        #     while j < len(acquisition_t_1):
        #         dt = acquisition_t_1[j] - t_i
    
        #         if dt > t_hi:
        #             break
    
        #         kbin = int(np.floor((dt - t_lo) / time_bin_ps))
        #         if 0 <= kbin < len(start_stop_results):
        #             start_stop_results[kbin] += 1
        #             # numPairs += 1
    
        #         j += 1
        
        # results = {
        #     "centers": centers.copy(),
        #     "histogram": start_stop_results.copy() / state["time_bin_ps"] * time_bin_ps_init,
        #     "histogram_raw": start_stop_results.copy(),
        #     "T_acq_s": Ta_s,
        #     "chA": ch_0,
        #     "chB": ch_1,
        # }

        results = g2_histogram_numba(
            acquisition_t_0,
            acquisition_t_1,
            edges,
            Ta_s=Ta_s,
            chA=ch_0,
            chB=ch_1,
            time_bin_ps_init=time_bin_ps_init,
        )

        if flag_rt == True:
            tau_rt_predicted_ps = state["tau_rt_ps"]
            # t_lo_rt = tau_rt_predicted_ps - state["tau_span_ps"] * 0.5
            # t_hi_rt = tau_rt_predicted_ps + state["tau_span_ps"] * 0.5
            t_lo_rt = tau_rt_ps_init - state["tau_span_ps"] * 0.5
            t_hi_rt = tau_rt_ps_init + state["tau_span_ps"] * 0.5
            time_bin_rt_ps = state["time_bin_rt_ps"]
            edges_rt = np.arange(t_lo_rt, t_hi_rt + time_bin_rt_ps, time_bin_rt_ps)
            if len(edges_rt) < 2:
                raise ValueError("Window too small for given bin width. Increase span or decrease bin.")
            centers_rt = edges_rt[:-1] + time_bin_rt_ps / 2
            start_stop_results_rt = np.zeros(len(centers_rt), dtype=np.int64)

            # j_start = 0
            # for t_i in acquisition_t_0rt:
            #     while j_start < len(acquisition_t_2) and (acquisition_t_2[j_start] - t_i) < t_lo_rt:
            #         j_start += 1
        
            #     j = j_start
            #     while j < len(acquisition_t_2):
            #         dt = acquisition_t_2[j] - t_i
        
            #         if dt > t_hi_rt:
            #             break
        
            #         kbin = int(np.floor((dt - t_lo_rt) / time_bin_rt_ps))
            #         if 0 <= kbin < len(start_stop_results_rt):
            #             start_stop_results_rt[kbin] += 1
            #             # numPairs += 1
        
            #         j += 1
            
            # results_rt = {
            #     "centers": centers_rt.copy(),
            #     "histogram": start_stop_results_rt.copy() / state["time_bin_rt_ps"] * time_bin_ps_init,
            #     "histogram_raw": start_stop_results_rt.copy(),
            #     "T_acq_s": Ta_rt_s,
            #     "chA": ch_0,
            #     "chB": ch_2,
            # }

            results_rt = g2_histogram_numba(
                acquisition_t_0rt,
                acquisition_t_2,
                edges_rt,
                Ta_s=Ta_rt_s,
                chA=ch_0,
                chB=ch_2,
                time_bin_ps_init=time_bin_ps_init,
            )

        # ------------------------------------------------------------
        # STEP 3:
        # Estimate time offset and frequency skew from g2 fit
        # ------------------------------------------------------------
        try:
            fit_parameters = fit_parameter_guess.copy()
            fit_parameters["center"] = float(
                results["centers"][np.argmax(results["histogram"])]
            )

            fit = fit_g2(
                results,
                curve=fit_curve,
                parameters=fit_parameters,
                # fit_window=state["tau_span_ps"],
                fit_window=np.rint(float(1e12)/Freq) * 1.2,
                show=show_fit,
            )

        except Exception as exc:
            raise

            print(f"Time-dependent g2 fit failed at t_stop={(acquisition["point_ps"]*1e-12):.6g} s: {exc}")
            fit = None
            exc_value = exc
        else:
            exc_value = None

        if flag_rt == True:
            try:
                fit_parameters = fit_parameter_guess.copy()
                fit_parameters["center"] = float(
                    results_rt["centers"][np.argmax(results_rt["histogram"])]
                )
    
                # print(fit_parameters["center"])
                fit_rt = fit_g2(
                    results_rt,
                    curve=fit_curve,
                    parameters=fit_parameters,
                    # fit_window=state["tau_span_ps"],
                    fit_window=np.rint(float(1e12)/Freq) * 1.2,
                    show=show_fit,
                    # show=True,
                )
    
            except Exception as exc:
                raise
    
                print(f"Time-dependent g2 fit failed at t_stop={(acquisition["point_ps"]*1e-12):.6g} s: {exc}")
                fit = None
                exc_value = exc
            else:
                exc_value = None

        # ------------------------------------------------------------
        # STEP 4:
        # Update tau, u0, and records
        # ------------------------------------------------------------

        point_records[acquisition_index] = acquisition["point_ps"]
        tau_ps_records[acquisition_index] = fit["center"]
        tau_rt_ps_records[acquisition_index] = fit_rt["center"]
        Ta_s_records[acquisition_index] = state["Ta_s"]
        Ta_rt_s_records[acquisition_index] = state["Ta_rt_s"]
        # epsilon_ps_records[acquisition_index] = fit["center"] - fit_rt["center"] / 2
        epsilon_ps_records[acquisition_index] = (fit["center"] - T_calib) - (fit_rt["center"] - T_calib_rt) / 2
        # fwhm_records[acquisition_index] = fwhm
        sigma_records[acquisition_index] = fit["parameters"]["sigma"]
        sigma_rt_records[acquisition_index] = fit_rt["parameters"]["sigma"]
        tau_s_uncertainty_records[acquisition_index] = fit["center_uncertainty"]
        # fwhm_uncertainty_records[acquisition_index] = fit["fwhm_uncertainty"]
        CC_count_records[acquisition_index] = np.sum(results["histogram_raw"])
        CC_count_rt_records[acquisition_index] = np.sum(results_rt["histogram_raw"])
        # S_peak_records[acquisition_index] = np.sqrt(np.sum(results["histogram"]) ** 2 * np.size(results["centers"]) / np.size(acquisition_t_0) / np.size(acquisition_t_1))

        # print(acquisition["point_ps"])
        tau_ref_ps, tau_ref_uncertainty_ps, N_ref_samples = qcs_ref_estimator(
            acquisition_ref_1 - acquisition_ref_0,
            acquisition_ref_0,
            t_eval_ps=acquisition["point_ps"],
        )

        tau_ref_ps_records[acquisition_index] = tau_ref_ps
        tau_ref_uncertainty_ps_records[acquisition_index] = tau_ref_uncertainty_ps / np.sqrt(N_ref_samples)
        T_u0 = 0.0

        if acquisition_index > 0:
            # udrift = (fit["center"] - tau_predicted_ps) / (acquisition["point_ps"] - state["T_lastmid_ps"])
            # u0 = state["u0"] + udrift
            # state["u0"] = u0

            # First order correction
            u0 = (fit["center"] - state["tau_ps"]) / (acquisition["point_ps"] - state["T_lastmid_ps"])
            udrift = u0 - state["u0"]
            u0_last = state["u0"]
            state["u0"] = u0
            T_u0 = (acquisition["point_ps"] + state["T_lastmid_ps"]) / 2

            # Second order correction
            if second_order_fit:
                # u1 = (state["u0"] - u0_last) / (acquisition["point_ps"] - state["T_lastmid_ps"])
                u1 = (state["u0"] - u0_last) / (T_u0 - state["T_lastu0_ps"])
                state["u1"] = u1
            
            Ta_s = np.round((fwhm_calib_ps / 10 / np.abs(udrift) * 1e-12) / Ta_s_init) * Ta_s_init
            Ta_s = np.min([Ta_s, Ta_s_max])
            state["Ta_s"] = np.max([Ta_s_init, Ta_s])
            # state["time_bin_ps"] = round(time_bin_ps_init / state["Ta_s"] * Ta_s_init / 10.0) * 10.0

            # print(f"Ta_s: {state["Ta_s"]:.2f}")
        else:
            state["Ta_s"] = Ta_s

        state["tau_ps"] = fit["center"]
        state["T_lastmid_ps"] = acquisition["point_ps"]
        state["T_lastu0_ps"] = T_u0
        u0_records[acquisition_index] = state["u0"]
        u1_records[acquisition_index] = state["u1"]

        if flag_rt == True:
            index_0rt = end_index_0
            index_2 = end_index_2

            if acquisition_index > 0:
                T_rt_thismid_ps = (acquisition_t_2[0] + acquisition_t_2[-1]) / 2
                udrift_rt = (fit_rt["center"] - tau_rt_predicted_ps) / (T_rt_thismid_ps - state["T_rt_lastmid_ps"])
                # u0 = state["u0"] + udrift
                # state["u0"] = u0
                
                Ta_rt_s = np.round((fwhm_calib_ps / 10 / np.abs(udrift_rt) * 1e-12) / Ta_s_init) * Ta_s_init
                Ta_rt_s = np.min([Ta_rt_s, Ta_rt_s_max])
                # state["Ta_rt_s"] = np.max([Ta_s_init, Ta_rt_s])
                state["Ta_rt_s"] = Ta_rt_s
                # state["Ta_rt_s"] = state["Ta_s"]
                # time_bin_rt_ps = round(time_bin_ps_init / state["Ta_rt_s"] * Ta_s_init / 10.0) * 10.0
                # state["time_bin_rt_ps"] = np.max([time_bin_rt_ps, time_bin_ps_min])
    
                # print(f"Ta_rt_s: {state["Ta_rt_s"]:.2f}")
            else:
                state["Ta_rt_s"] = Ta_rt_s

            state["tau_rt_ps"] = fit_rt["center"]
            state["T_rt_lastmid_ps"] = (acquisition_t_2[0] + acquisition_t_2[-1]) / 2

        # acquisition_results.append(acquisition)

        # Advance both channel indices and move chronologically onward.
        acquisition_index += 1
        index_0 = end_index_0
        index_1 = end_index_1
        # index_ref = end_index_ref
        window_start_ps = window_end_ps
        flag_rt = False

        # cpu_time_end = time.time()
        # print(f"Acquisition {acquisition_index} processed in {cpu_time_end - cpu_time_start:.4f} seconds.")

    # ------------------------------------------------------------
    # After scan: plot results
    # ------------------------------------------------------------

    # Step 1: remove the first six completed acquisitions from all records.
    # start_index = 6
    start_index = 1
    record_slice = slice(min(start_index, acquisition_index), acquisition_index)
    point_records = point_records[record_slice]
    tau_ps_records = tau_ps_records[record_slice]
    tau_rt_ps_records = tau_rt_ps_records[record_slice]
    epsilon_ps_records = epsilon_ps_records[record_slice]
    tau_ref_ps_records = tau_ref_ps_records[record_slice]
    tau_ref_uncertainty_ps_records = tau_ref_uncertainty_ps_records[record_slice]
    Ta_s_records = Ta_s_records[record_slice]
    Ta_rt_s_records = Ta_rt_s_records[record_slice]
    fwhm_records = fwhm_records[record_slice]
    sigma_records = sigma_records[record_slice]
    sigma_rt_records = sigma_rt_records[record_slice]
    tau_s_uncertainty_records = tau_s_uncertainty_records[record_slice]
    CC_count_records = CC_count_records[record_slice]
    CC_count_rt_records = CC_count_rt_records[record_slice]
    coincidence_records = coincidence_records[record_slice]
    u0_records = u0_records[record_slice]
    u1_records = u1_records[record_slice]
    acquisition_index = len(point_records)

    # Fig1: one way details
    fig1, axes = plt.subplots(5, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_ps_records - tau_ps_records[0], "o-")
    ax1.set_ylabel(r"$\tau$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, Ta_s_records, "o-")
    ax2.set_ylabel(r"$T_a$ (s)")
    ax2.grid(True)

    ax3 = axes[2]
    ax3.plot(point_records * 1e-12, sigma_records, "o-")
    ax3.set_ylabel(r"$\sigma$ (ps)")
    ax3.grid(True)

    ax4 = axes[3]
    ax4.plot(point_records * 1e-12, CC_count_records, "o-")
    ax4.set_ylabel(r"CC")
    ax4.grid(True)

    ax5 = axes[4]
    ax5.plot(point_records * 1e-12, tau_ref_ps_records - tau_ref_ps_records[0], "o-")
    ax5.set_ylabel(r"$\tau_{ref}$ (ps)")
    ax5.grid(True)

    # ax6 = axes[5]
    # ax6.plot(point_records * 1e-12, tau_ref_uncertainty_ps_records, "o-")
    # ax6.set_ylabel(r"$\tau_{ref}$ (ps)")
    # ax6.grid(True)

    # tmp = tau_ref_ps_records - tau_ref_ps_records[0]
    # tmp_idx = np.where(tmp > 100)[0]
    # print(tmp_idx)

    axes[-1].set_xlabel("Record Points (s)")
    fig1.tight_layout()

    # Fig2: two way details
    fig2, axes = plt.subplots(4, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_rt_ps_records - tau_rt_ps_records[0], "o-")
    ax1.set_ylabel(r"$\tau_{rt}$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, Ta_rt_s_records, "o-")
    ax2.set_ylabel(r"$T_{a_rt}$ (s)")
    ax2.grid(True)

    ax3 = axes[2]
    ax3.plot(point_records * 1e-12, sigma_rt_records, "o-")
    ax3.set_ylabel(r"$\sigma_{rt}$ (ps)")
    ax3.grid(True)

    ax4 = axes[3]
    ax4.plot(point_records * 1e-12, CC_count_rt_records, "o-")
    ax4.set_ylabel(r"CC")
    ax4.grid(True)

    axes[-1].set_xlabel("Record Points (s)")
    fig2.tight_layout()

    # Fig3: summaries
    fig3, axes = plt.subplots(3, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    tau_diff = tau_ps_records - tau_ref_ps_records
    # tau_diff = np.delete(tau_diff, tmp_idx)
    # point_records2 = np.delete(point_records, tmp_idx)
    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_diff - np.mean(tau_diff), "o-")
    # ax1.plot(point_records2 * 1e-12, tau_diff - np.mean(tau_diff), "o-")
    ax1.set_ylabel(r"$\tau$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, epsilon_ps_records, "o-")
    ax2.set_ylabel(r"$\epsilon$ (ps)")
    ax2.grid(True)

    ax3 = axes[2]
    ax3.plot(point_records * 1e-12, epsilon_ps_records - tau_ref_ps_records, "o-")
    ax3.set_ylabel(r"$\epsilon$ - $\tau_{diff}$ (ps)")
    ax3.grid(True)

    # ax3 = axes[2]
    # ax3.plot(point_records * 1e-12, CC_count_records, "o-")
    # ax3.set_ylabel(r"CC")
    # ax3.grid(True)

    axes[-1].set_xlabel("Record Points (s)")
    fig3.tight_layout()

    print(np.std(tau_diff))
    print(np.std(epsilon_ps_records - tau_ref_ps_records))
    # print(np.mean(tau_diff))

    plt.show()

    return {
        "center": point_records,
        "tau": tau_ps_records,
        "tau_rt": tau_rt_ps_records,
        "tau_ref": tau_ref_ps_records,
        "tau_ref_uncertainty": tau_ref_uncertainty_ps_records,
        "tau_diff": tau_diff,
        "epsilon": epsilon_ps_records,
        "Ta": Ta_s_records,
        "Ta_rt": Ta_rt_s_records,
        "sigma": sigma_records,
        "sigma_rt": sigma_rt_records,
        "CC": CC_count_records,
        "CC_rt": CC_count_rt_records,
        "figure1": fig1,
        "figure2": fig2,
        "figure3": fig3,
        # "axes": axes,
    }


def qcs_plot(results):
    point_records = results["center"]
    tau_ps_records = results["tau"]
    tau_rt_ps_records = results["tau_rt"]
    tau_ref_ps_records = results["tau_ref"]
    tau_ref_uncertainty_ps_records = results["tau_ref_uncertainty"]
    epsilon_ps_records = results["epsilon"]
    Ta_s_records = results["Ta"]
    Ta_rt_s_records = results["Ta_rt"]
    sigma_records = results["sigma"]
    sigma_rt_records = results["sigma_rt"]
    CC_count_records = results["CC"]
    CC_count_rt_records = results["CC_rt"]
    tau_diff = results["tau_diff"]

    # Fig1: one way details
    fig1, axes = plt.subplots(5, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_ps_records - tau_ps_records[0], "o-")
    ax1.set_ylabel(r"$\tau$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, Ta_s_records, "o-")
    ax2.set_ylabel(r"$T_a$ (s)")
    ax2.grid(True)

    ax3 = axes[2]
    ax3.plot(point_records * 1e-12, sigma_records, "o-")
    ax3.set_ylabel(r"$\sigma$ (ps)")
    ax3.grid(True)

    ax4 = axes[3]
    ax4.plot(point_records * 1e-12, CC_count_records, "o-")
    ax4.set_ylabel(r"CC")
    ax4.grid(True)

    ax5 = axes[4]
    ax5.plot(point_records * 1e-12, tau_ref_ps_records - tau_ref_ps_records[0], "o-")
    ax5.set_ylabel(r"$\tau_{ref}$ (ps)")
    ax5.grid(True)

    # ax6 = axes[5]
    # ax6.plot(point_records * 1e-12, tau_ref_uncertainty_ps_records, "o-")
    # ax6.set_ylabel(r"$\tau_{ref}$ (ps)")
    # ax6.grid(True)

    # tmp = tau_ref_ps_records - tau_ref_ps_records[0]
    # tmp_idx = np.where(tmp > 100)[0]
    # print(tmp_idx)

    axes[-1].set_xlabel("Record Points (s)")
    fig1.tight_layout()

    # Fig2: two way details
    fig2, axes = plt.subplots(4, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_rt_ps_records - tau_rt_ps_records[0], "o-")
    ax1.set_ylabel(r"$\tau_{rt}$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, Ta_rt_s_records, "o-")
    ax2.set_ylabel(r"$T_{a_rt}$ (s)")
    ax2.grid(True)

    ax3 = axes[2]
    ax3.plot(point_records * 1e-12, sigma_rt_records, "o-")
    ax3.set_ylabel(r"$\sigma_{rt}$ (ps)")
    ax3.grid(True)

    ax4 = axes[3]
    ax4.plot(point_records * 1e-12, CC_count_rt_records, "o-")
    ax4.set_ylabel(r"CC")
    ax4.grid(True)

    axes[-1].set_xlabel("Record Points (s)")
    fig2.tight_layout()

    # Fig3: summaries
    fig3, axes = plt.subplots(2, 1, sharex=True, squeeze=False)
    axes = axes[:, 0]

    tau_diff = tau_ps_records - tau_ref_ps_records
    # tau_diff = np.delete(tau_diff, tmp_idx)
    # point_records2 = np.delete(point_records, tmp_idx)
    ax1 = axes[0]
    ax1.plot(point_records * 1e-12, tau_diff - np.mean(tau_diff), "o-")
    # ax1.plot(point_records2 * 1e-12, tau_diff - np.mean(tau_diff), "o-")
    ax1.set_ylabel(r"$\tau$ (ps)")
    ax1.grid(True)

    ax2 = axes[1]
    ax2.plot(point_records * 1e-12, epsilon_ps_records, "o-")
    ax2.set_ylabel(r"$\epsilon$ (ps)")
    ax2.grid(True)

    # ax3 = axes[2]
    # ax3.plot(point_records * 1e-12, CC_count_records, "o-")
    # ax3.set_ylabel(r"CC")
    # ax3.grid(True)

    axes[-1].set_xlabel("Record Points (s)")
    fig3.tight_layout()

    # print(np.std(tau_diff))
    # print(np.mean(tau_diff))

    plt.show()