"""
fft_analysys.py
===============
Análisis FFT (Fast Fourier Transform) y detección de anomalías con Pandas.

Método:
    1. Cargar señales CSV generadas en Vibracion.py
    2. Aplicar FFT para pasar al dominio de frecuencia
    3. Detectar anomalías por tres métodos:
       a) Pico en frecuencia prohibida (frequency fault)
       b) Amplitud instantánea sobre umbral (spike fault)
       c) Energía total creciente en ventana (drift fault)
    4. Generar reporte de eventos con Pandas
    5. Visualización comparativa de espectros

Autor: Juan de Jesús Gomez López
Elaborado: Abril 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path


# ─────────────────────────────────────────────
# Parámetros de detección
# ─────────────────────────────────────────────

SAMPLE_RATE       = 1000        # Hz — debe coincidir con vibration_sim.py

# Ventana de análisis FFT (sliding window)
WINDOW_SIZE       = 512         # muestras por ventana (~0.5 segundos)
HOP_SIZE          = 256         # paso entre ventanas (50% overlap)

# Frecuencias conocidas del telar normal
FREQS_FUND       = [25.0, 50.0]   # Hz — fundamentales esperadas
FREQ_ANOMALY     = 80.0           # Hz — frecuencia que indica fallo

# Umbrales de detección
FREQ_POWER_THRESH = 0.05        # Potencia mínima para considerar un pico anómalo
SPIKE_THRESH      = 2.0         # g — amplitud máxima aceptable
DRIFT_ENERGY_WIN  = 200         # muestras — ventana para calcular energía creciente
DRIFT_THRESH      = 1.15        # factor de incremento de energía que dispara alerta


# ─────────────────────────────────────────────
# Carga de datos
# ─────────────────────────────────────────────

def load_signal(csv_path: Path) -> pd.DataFrame:
    """
    Carga un CSV de señal generado en Vibracion.py.
    Agrega columna de índice de muestra para facilitar el análisis.
    """
    df = pd.read_csv(csv_path)
    df["sample_idx"] = np.arange(len(df))
    print(f"Cargado: {csv_path.name}  ({len(df):,} muestras)")
    return df


# ─────────────────────────────────────────────
# FFT sobre señal completa
# ─────────────────────────────────────────────

def compute_fft(signal: np.ndarray,
                sample_rate: int = SAMPLE_RATE) -> tuple[np.ndarray, np.ndarray]:
    """
    Calcula la FFT de una señal y retorna frecuencias y magnitudes.

    La FFT convierte N muestras en el tiempo → N/2 componentes de frecuencia.
    Aplicamos una ventana de Hanning para reducir el efecto de borde
    (sin ventana, la FFT "ve" discontinuidades en los bordes del array).

    Returns:
        freqs  : array de frecuencias en Hz  (eje X del espectro)
        magnitudes : potencia de cada frecuencia (eje Y del espectro)
    """
    n = len(signal)

    # Ventana de Hanning — suaviza los bordes del segmento
    window     = np.hanning(n)
    windowed   = signal * window

    # FFT y nos quedamos solo con la mitad positiva (simetría de señal real)
    fft_result = np.fft.rfft(windowed)
    magnitudes = np.abs(fft_result) / n   # normalizar por número de muestras

    # Frecuencias correspondientes a cada bin de la FFT
    freqs = np.fft.rfftfreq(n, d=1.0 / sample_rate)

    return freqs, magnitudes


def sliding_fft(signal: np.ndarray,
                sample_rate: int = SAMPLE_RATE,
                window_size: int = WINDOW_SIZE,
                hop_size: int = HOP_SIZE) -> pd.DataFrame:
    """
    Aplica FFT en ventanas deslizantes sobre toda la señal.
    Esto nos permite ver cómo evoluciona el espectro en el tiempo.

    Returns:
        DataFrame con columnas:
            window_start, window_end, time_center,
            peak_freq, peak_power, anomaly_freq_power, energy
    """
    rows = []
    n    = len(signal)

    for start in range(0, n - window_size, hop_size):
        end     = start + window_size
        segment = signal[start:end]

        freqs, mags = compute_fft(segment, sample_rate)

        # Energía total de la ventana (suma de cuadrados)
        energy = float(np.sum(segment ** 2) / window_size)

        # Frecuencia con mayor potencia
        peak_idx   = np.argmax(mags)
        peak_freq  = float(freqs[peak_idx])
        peak_power = float(mags[peak_idx])

        # Potencia en la frecuencia de anomalía (80 Hz ± 2 Hz de tolerancia)
        anomaly_mask  = (freqs >= FREQ_ANOMALY - 2) & (freqs <= FREQ_ANOMALY + 2)
        anomaly_power = float(mags[anomaly_mask].max()) if anomaly_mask.any() else 0.0

        rows.append({
            "window_start":      start,
            "window_end":        end,
            "time_center":       (start + end) / 2 / sample_rate,
            "peak_freq":         peak_freq,
            "peak_power":        peak_power,
            "anomaly_freq_power": anomaly_power,
            "energy":            energy,
        })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
#  Detectores de anomalías
# ─────────────────────────────────────────────

def detect_frequency_fault(fft_df: pd.DataFrame,
                            threshold: float = FREQ_POWER_THRESH) -> pd.DataFrame:
    """
    Detecta ventanas donde aparece potencia en la frecuencia prohibida (80Hz).
    Regla: anomaly_freq_power > threshold → ALERTA
    """
    fft_df = fft_df.copy()
    fft_df["freq_fault"] = fft_df["anomaly_freq_power"] > threshold
    n_alerts = fft_df["freq_fault"].sum()
    print(f" Alertas  por frecuencia detectadas: {n_alerts} / {len(fft_df)} ventanas")
    return fft_df


def detect_spike_fault(signal_df: pd.DataFrame,
                       threshold: float = SPIKE_THRESH) -> pd.DataFrame:
    """
    Detecta muestras donde la amplitud supera el umbral físico.
    Regla: |amplitude| > threshold → ALERTA
    """
    signal_df = signal_df.copy()
    signal_df["spike_fault"] = signal_df["amplitude"].abs() > threshold
    n_alerts = signal_df["spike_fault"].sum()
    print(f" Alertas  por spike detectadas: {n_alerts} / {len(signal_df)} ventanas")
    return signal_df


def detect_drift_fault(signal_df: pd.DataFrame,
                       window: int = DRIFT_ENERGY_WIN,
                       threshold: float = DRIFT_THRESH) -> pd.DataFrame:
    """
    Detecta cuando la energía de la señal crece progresivamente.
    Usa una ventana rodante para calcular energía y compara con la
    energía de referencia (primeras muestras).

    Regla: rolling_energy / baseline_energy > threshold → ALERTA
    """
    signal_df = signal_df.copy()

    # Energía instantánea = amplitud²
    signal_df["instant_energy"] = signal_df["amplitude"] ** 2

    # Energía media en ventana rodante
    signal_df["rolling_energy"] = (
        signal_df["instant_energy"]
        .rolling(window=window, min_periods=window // 2)
        .mean()
    )

    # Energía de referencia: media de las primeras 500 muestras (estado sano)
    baseline_energy = signal_df["instant_energy"].iloc[:500].mean()

    signal_df["energy_ratio"] = signal_df["rolling_energy"] / baseline_energy
    signal_df["drift_fault"]  = signal_df["energy_ratio"] > threshold

    n_alerts = signal_df["drift_fault"].sum()
    print(f" Alertas  por drift detectadas: {n_alerts} / {len(signal_df)} ventanas")
    return signal_df


# ─────────────────────────────────────────────
# Generador de reporte de eventos
# ─────────────────────────────────────────────

def generate_event_report(signal_df: pd.DataFrame,
                           fft_df: pd.DataFrame,
                           label: str,
                           output_dir: Path) -> pd.DataFrame:
    """
    Consolida todos los detectores en un reporte de eventos por timestamp.
    Cada fila = una ventana de análisis.
    """
    # Aseguramos que existan las columnas (si el detector no se corrió)
    for col in ["spike_fault", "drift_fault"]:
        if col not in signal_df.columns:
            signal_df[col] = False

    # Agregar estadísticas de señal por ventana (para unir con FFT)
    def window_stats(df, start, end):
        seg = df[(df["sample_idx"] >= start) & (df["sample_idx"] < end)]
        return {
            "amp_max":   seg["amplitude"].abs().max(),
            "amp_mean":  seg["amplitude"].abs().mean(),
            "spike_in_window": seg["spike_fault"].any() if "spike_fault" in seg else False,
            "drift_in_window": seg["drift_fault"].any() if "drift_fault" in seg else False,
        }

    report_rows = []
    for _, row in fft_df.iterrows():
        stats = window_stats(signal_df, row["window_start"], row["window_end"])
        report_rows.append({
            "label":            label,
            "time_center_s":    round(row["time_center"], 4),
            "peak_freq_hz":     round(row["peak_freq"], 2),
            "anomaly_pwr_80hz": round(row["anomaly_freq_power"], 5),
            "window_energy":    round(row["energy"], 5),
            "amp_max":          round(stats["amp_max"], 4),
            "freq_fault":       bool(row["freq_fault"]),
            "spike_fault":      bool(stats["spike_in_window"]),
            "drift_fault":      bool(stats["drift_in_window"]),
        })

    report_df = pd.DataFrame(report_rows)

    # Columna resumen: ¿algún detector disparó alerta?
    report_df["any_fault"] = (
        report_df["freq_fault"] |
        report_df["spike_fault"] |
        report_df["drift_fault"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"report_{label}.csv"
    report_df.to_csv(out_path, index=False)
    print(f"Reporte guardado: {out_path}")
    return report_df


# ─────────────────────────────────────────────
# Visualización de espectros FFT
# ─────────────────────────────────────────────

def plot_fft_comparison(normal_df: pd.DataFrame,
                        faulty_df: pd.DataFrame,
                        fault_type: str,
                        output_dir: Path) -> None:
    """
    Compara el espectro FFT completo de señal normal vs señal con fallo.
    También muestra la evolución temporal de la energía y potencia en 80Hz.
    """
    fig = plt.figure(figsize=(16, 10), facecolor="#0f0f1a")
    fig.suptitle(
        f"Análisis FFT — Telar Monitor\nTipo de fallo: {fault_type.upper()}",
        color="white", fontsize=14, fontweight="bold"
    )

    gs  = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.35)
    PAL = {"normal": "#00d4ff", "faulty": "#ff4757",
           "alert":  "#ffa502", "bg":     "#1a1a2e"}

    # ── Panel 1: Espectro FFT señal normal (señal completa) ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PAL["bg"])
    # FFT de toda la señal normal para vista global
    t_normal = normal_df["amplitude"].values
    freqs_n, mags_n = compute_fft(t_normal)
    ax1.plot(freqs_n, mags_n, color=PAL["normal"], linewidth=0.8)
    ax1.set_title("Espectro NORMAL", color="white", fontsize=10)
    ax1.set_xlabel("Frecuencia (Hz)", color="#aaaaaa", fontsize=8)
    ax1.set_ylabel("Magnitud", color="#aaaaaa", fontsize=8)
    ax1.axvline(25, color="#00ff88", linewidth=1, linestyle="--", alpha=0.7, label="25Hz")
    ax1.axvline(50, color="#88ff00", linewidth=1, linestyle="--", alpha=0.7, label="50Hz")
    ax1.set_xlim(0, 150)
    ax1.legend(fontsize=7, facecolor=PAL["bg"], labelcolor="white")
    ax1.tick_params(colors="#aaaaaa")
    for sp in ax1.spines.values(): sp.set_color("#333355")
    ax1.grid(True, color="#222244", linewidth=0.4)

    # ── Panel 2: Espectro FFT señal con fallo ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PAL["bg"])
    t_faulty = faulty_df["amplitude"].values
    freqs_f, mags_f = compute_fft(t_faulty)
    ax2.plot(freqs_f, mags_f, color=PAL["faulty"], linewidth=0.8)
    ax2.set_title(f"Espectro CON FALLO ({fault_type})", color="white", fontsize=10)
    ax2.set_xlabel("Frecuencia (Hz)", color="#aaaaaa", fontsize=8)
    ax2.set_ylabel("Magnitud", color="#aaaaaa", fontsize=8)
    ax2.axvline(25, color="#00ff88", linewidth=1, linestyle="--", alpha=0.7, label="25Hz")
    ax2.axvline(50, color="#88ff00", linewidth=1, linestyle="--", alpha=0.7, label="50Hz")
    ax2.axvline(80, color=PAL["alert"], linewidth=1.5, linestyle="--", alpha=0.9, label="80Hz ⚠")
    ax2.set_xlim(0, 150)
    ax2.legend(fontsize=7, facecolor=PAL["bg"], labelcolor="white")
    ax2.tick_params(colors="#aaaaaa")
    for sp in ax2.spines.values(): sp.set_color("#333355")
    ax2.grid(True, color="#222244", linewidth=0.4)

    # ── Panel 3: Potencia en 80Hz a lo largo del tiempo ──
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PAL["bg"])
    # Sliding FFT ya viene calculado en normal_df y faulty_df (son fft_df aquí)
    ax3.set_title("Potencia en 80Hz — Evolución temporal", color="white", fontsize=10)
    ax3.set_xlabel("Tiempo (s)", color="#aaaaaa", fontsize=8)
    ax3.set_ylabel("Potencia @ 80Hz", color="#aaaaaa", fontsize=8)
    ax3.axhline(FREQ_POWER_THRESH, color=PAL["alert"],
                linewidth=1, linestyle="--", label=f"Umbral ({FREQ_POWER_THRESH})")
    ax3.tick_params(colors="#aaaaaa")
    for sp in ax3.spines.values(): sp.set_color("#333355")
    ax3.grid(True, color="#222244", linewidth=0.4)
    ax3.legend(fontsize=7, facecolor=PAL["bg"], labelcolor="white")
    # (Se llena desde run_analysis pasando sliding_fft_dfs)
    ax3._needs_sliding = True
    ax3._pal = PAL

    # ── Panel 4: Energía de ventana deslizante ──
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PAL["bg"])
    ax4.set_title("Energía de ventana — Evolución temporal", color="white", fontsize=10)
    ax4.set_xlabel("Tiempo (s)", color="#aaaaaa", fontsize=8)
    ax4.set_ylabel("Energía media", color="#aaaaaa", fontsize=8)
    ax4.tick_params(colors="#aaaaaa")
    for sp in ax4.spines.values(): sp.set_color("#333355")
    ax4.grid(True, color="#222244", linewidth=0.4)
    ax4._needs_sliding = True
    ax4._pal = PAL

    return fig, (ax3, ax4)


def fill_temporal_panels(ax3, ax4,
                          sliding_normal: pd.DataFrame,
                          sliding_faulty: pd.DataFrame,
                          PAL: dict) -> None:
    """Rellena los paneles temporales con los datos de sliding FFT."""
    # Panel 3 — potencia en 80Hz
    ax3.plot(sliding_normal["time_center"], sliding_normal["anomaly_freq_power"],
             color=PAL["normal"], linewidth=1.2, label="Normal", alpha=0.8)
    ax3.plot(sliding_faulty["time_center"], sliding_faulty["anomaly_freq_power"],
             color=PAL["faulty"], linewidth=1.2, label="Con fallo", alpha=0.8)
    ax3.legend(fontsize=7, facecolor=PAL["bg"], labelcolor="white")

    # Panel 4 — energía
    ax4.plot(sliding_normal["time_center"], sliding_normal["energy"],
             color=PAL["normal"], linewidth=1.2, label="Normal", alpha=0.8)
    ax4.plot(sliding_faulty["time_center"], sliding_faulty["energy"],
             color=PAL["faulty"], linewidth=1.2, label="Con fallo", alpha=0.8)
    ax4.legend(fontsize=7, facecolor=PAL["bg"], labelcolor="white")


# ─────────────────────────────────────────────
# Metodo principal de análisis
# ─────────────────────────────────────────────

def run_analysis(data_dir:   Path = Path("data/vibration"),
                 output_dir: Path = Path("data/analysis")) -> pd.DataFrame:
    """
    Metodo completo:
        1. Carga todas las señales CSV
        2. Aplica FFT deslizante
        3. Corre los tres detectores
        4. Genera reportes y figuras
        5. Retorna tabla resumen consolidada
    """
    print("\n" + "═" * 58)
    print("  TELAR MONITOR — Análisis FFT y Detección de Anomalías")
    print("═" * 58)

    output_dir.mkdir(parents=True, exist_ok=True)
    fault_types = ["frequency", "spike", "drift"]
    all_reports = []

    # Cargar señal normal una sola vez (referencia)
    print("\n Señal normal:")
    normal_sig_df = load_signal(data_dir / "normal.csv")
    normal_fft_df = sliding_fft(normal_sig_df["amplitude"].values)
    normal_fft_df = detect_frequency_fault(normal_fft_df)

    for fault in fault_types:
        print(f"\n{'─'*58}")
        print(f"  Procesando: {fault.upper()}")
        print(f"{'─'*58}")

        # 1. Cargar señal
        sig_df = load_signal(data_dir / f"faulty_{fault}.csv")

        # 2. FFT deslizante
        fft_df = sliding_fft(sig_df["amplitude"].values)

        # 3. Detectores
        fft_df  = detect_frequency_fault(fft_df)
        sig_df  = detect_spike_fault(sig_df)
        sig_df  = detect_drift_fault(sig_df)

        # 4. Reporte consolidado
        report = generate_event_report(sig_df, fft_df, label=fault,
                                       output_dir=output_dir)
        all_reports.append(report)

        # 5. Visualización
        fig, (ax3, ax4) = plot_fft_comparison(
            normal_sig_df, sig_df, fault, output_dir
        )
        fill_temporal_panels(ax3, ax4, normal_fft_df, fft_df,
                             PAL={"normal": "#00d4ff", "faulty": "#ff4757",
                                  "alert":  "#ffa502", "bg":    "#1a1a2e"})
        fig_path = output_dir / f"fft_analysis_{fault}.png"
        plt.savefig(fig_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"Figura: {fig_path}")

    # ── Resumen consolidado ──
    full_report = pd.concat(all_reports, ignore_index=True)

    print("\n" + "─" * 58)
    print("  RESUMEN DE DETECCIÓN POR TIPO DE FALLO")
    print("─" * 58)
    summary = (
        full_report.groupby("label")[["freq_fault", "spike_fault",
                                      "drift_fault", "any_fault"]]
        .sum()
        .rename(columns={
            "freq_fault":  "alertas_freq",
            "spike_fault": "alertas_spike",
            "drift_fault": "alertas_drift",
            "any_fault":   "alertas_total"
        })
    )
    total_windows = full_report.groupby("label").size().rename("ventanas_total")
    summary = pd.concat([summary, total_windows], axis=1)
    summary["detección_%"] = (
        summary["alertas_total"] / summary["ventanas_total"] * 100
    ).round(1)
    print(summary.to_string())

    summary_path = output_dir / "detection_summary.csv"
    summary.to_csv(summary_path)
    print(f"\n Resumen guardado: {summary_path}")
    print("═" * 58 + "\n")

    return full_report


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    report = run_analysis(
        data_dir=Path("data/vibration"),
        output_dir=Path("data/analysis")
    )