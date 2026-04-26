"""
Vibracion_Sim.py
================
Simulación de señales de vibración de un telar industrial.

Modelo:
    señal(t) = componente_fundamental + armónicos + ruido + anomalía

Autor: Juan de Jesús Gomez López
Elaborado: Abril 2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ─────────────────────────────────────────────
# Parámetros de la máquina
# ─────────────────────────────────────────────

# Parámetros temporales
SAMPLE_RATE    = 1000   # Hz — muestras por segundo (resolución de la señal)
DURATION       = 10.0   # segundos por segmento de medición
N_SAMPLES      = int(SAMPLE_RATE * DURATION)

# Frecuencias del telar (Hz)
FREQ_FUND      = 25.0   # Frecuencia fundamental del motor principal
FREQ_ARMONIC  = 50.0   # Primer armónico (el doble es común en motores)
FREQ_ANOMALY   = 80.0   # Frecuencia que aparece cuando hay un defecto

# Amplitudes con unidades de g (aceleración) 
# Representando la intensidad de la aceleración máxima que experimenta el telar al oscilar
AMP_FUND       = 1.0    # Amplitud de la componente fundamental
AMP_ARMONIC   = 0.4    # Los armónicos son más débiles que la fundamental
AMP_AMBIENTE      = 0.05   # Ruido ambiental (siempre presente)
AMP_ANOMALY    = 0.6    # Intensidad de la anomalía cuando ocurre


# ─────────────────────────────────────────────
# Generador de señal NORMAL
# ─────────────────────────────────────────────

def generate_normal_signal(duration: float = DURATION,
                            sample_rate: int = SAMPLE_RATE,
                            seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """
    Genera una señal de vibración de un telar en operación normal.

    Returns:
        t      : array de tiempo en segundos
        signal : array de amplitud (aceleración en g)
    """
    rng = np.random.default_rng(seed)
    n   = int(sample_rate * duration)
    t   = np.linspace(0, duration, n, endpoint=False)

    # Componente fundamental — Vibracion principal del telar
    fundamental = AMP_FUND * np.sin(2 * np.pi * FREQ_FUND * t)

    # Armónico — vibración secundaria normal en cualquier motor
    harmonic = AMP_ARMONIC * np.sin(2 * np.pi * FREQ_ARMONIC * t)

    # Ruido gaussiano — siempre presente en sensores reales
    noise = AMP_AMBIENTE * rng.standard_normal(n)

    signal = fundamental + harmonic + noise
    return t, signal


# ─────────────────────────────────────────────
# Inyector de ANOMALÍAS
# ─────────────────────────────────────────────

def inject_anomaly(t: np.ndarray,
                   signal: np.ndarray,
                   start_pct: float = 0.5,
                   duration_pct: float = 0.3,
                   anomaly_type: str = "frequency") -> tuple[np.ndarray, np.ndarray]:
    """
    Inyecta una anomalía en una señal normal existente.

    Argumentos:
        t             : array de tiempo
        signal        : señal base (se modifica una copia)
        start_pct     : en qué fracción del tiempo empieza la anomalía (0 a 1)
        duration_pct  : qué fracción del tiempo dura la anomalía (0 a 1)
        anomaly_type  : "frequency" | "spike" | "drift"
            - frequency : aparece una nueva frecuencia (fallo mecánico)
            - spike     : golpes intermitentes (hilo roto que golpea)
            - drift     : la amplitud sube gradualmente (desgaste)

    Returns:
        signal_faulty : señal con anomalía inyectada
        anomaly_mask  : array booleano — True donde hay anomalía
    """
    signal_faulty = signal.copy()
    n = len(t)

    # Calcular los índices donde vive la anomalía
    idx_start = int(n * start_pct)
    idx_end   = int(n * (start_pct + duration_pct))
    idx_end   = min(idx_end, n)  # No salirse del array

    # Máscara booleana — útil para visualizar y para Pandas
    anomaly_mask = np.zeros(n, dtype=bool)
    anomaly_mask[idx_start:idx_end] = True

    t_window = t[idx_start:idx_end]

    if anomaly_type == "frequency":
        # Una nueva frecuencia aparece — típico de un rodamiento desgastado
        anomaly_wave = AMP_ANOMALY * np.sin(2 * np.pi * FREQ_ANOMALY * t_window)
        signal_faulty[idx_start:idx_end] += anomaly_wave

    elif anomaly_type == "spike":
        # Golpes periódicos — como un hilo roto que golpea la lanzadera
        spike_period = int(SAMPLE_RATE * 0.1)  # cada 100ms
        for i in range(idx_start, idx_end, spike_period):
            end = min(i + 5, n)
            signal_faulty[i:end] += AMP_ANOMALY * 3  # pico de amplitud alta

    elif anomaly_type == "drift":
        # La amplitud crece — desgaste progresivo
        ramp = np.linspace(0, AMP_ANOMALY, len(t_window))
        signal_faulty[idx_start:idx_end] += ramp

    return signal_faulty, anomaly_mask


# ─────────────────────────────────────────────
# Guardado con Pandas
# ─────────────────────────────────────────────

def save_to_csv(t: np.ndarray,
                signal: np.ndarray,
                anomaly_mask: np.ndarray,
                label: str,
                output_dir: Path) -> pd.DataFrame:
    """
    Empaqueta la señal en un DataFrame y la guarda en CSV.

    Columnas:
        timestamp    : tiempo en segundos
        amplitude    : valor de la vibración (g)
        is_anomaly   : 0 = normal, 1 = anómalo
        label        : nombre del experimento
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "timestamp":  t,
        "amplitude":  signal,
        "is_anomaly": anomaly_mask.astype(int),
        "label":      label
    })

    filepath = output_dir / f"{label}.csv"
    df.to_csv(filepath, index=False)
    print(f"Guardado: {filepath}  ({len(df):,} filas)")
    return df


# ─────────────────────────────────────────────
# Visualización con Matplotlib
# ─────────────────────────────────────────────

def plot_signals(t: np.ndarray,
                 normal: np.ndarray,
                 faulty: np.ndarray,
                 mask: np.ndarray,
                 anomaly_type: str,
                 output_dir: Path) -> None:
    """
    Genera una figura comparativa de señal normal vs señal con anomalía.
    Resalta visualmente la zona donde ocurre el fallo.
    """
    fig = plt.figure(figsize=(14, 8), facecolor="#0f0f1a")
    fig.suptitle(
        f"Sistema de Monitoreo de Telar — Señal de Vibración\nAnomalia: [{anomaly_type.upper()}]",
        color="white", fontsize=14, fontweight="bold", y=0.98
    )

    gs = gridspec.GridSpec(2, 1, hspace=0.45)
    colors = {"normal": "#00d4ff", "faulty": "#ff4757", "mask": "#f5b743"}

    for idx, (signal, title, color) in enumerate([
        (normal, "Operación NORMAL",           colors["normal"]),
        (faulty, f"Operación con FALLO ({anomaly_type})", colors["faulty"]),
    ]):
        ax = fig.add_subplot(gs[idx])
        ax.set_facecolor("#1a1a2e")

        ax.plot(t, signal, color=color, linewidth=0.7, alpha=0.9, label=title)

        # Sombrear la zona de anomalía también en la señal normal (referencia)
        if idx == 1:
            ax.fill_between(t, signal.min(), signal.max(),
                            where=mask,
                            color=colors["mask"], alpha=0.15,
                            label="Zona de anomalía")
            # Líneas verticales de inicio/fin
            starts = np.where(np.diff(mask.astype(int)) == 1)[0]
            ends   = np.where(np.diff(mask.astype(int)) == -1)[0]
            for s in starts:
                ax.axvline(t[s], color=colors["mask"], linewidth=1.5,
                           linestyle="--", alpha=0.8)
            for e in ends:
                ax.axvline(t[e], color=colors["mask"], linewidth=1.5,
                           linestyle="--", alpha=0.8)

        ax.set_xlabel("Tiempo (s)", color="#aaaaaa", fontsize=9)
        ax.set_ylabel("Amplitud (g)", color="#aaaaaa", fontsize=9)
        ax.set_title(title, color="white", fontsize=11, pad=8)
        ax.tick_params(colors="#aaaaaa")
        for spine in ax.spines.values():
            spine.set_color("#333355")
        ax.legend(loc="upper right", fontsize=8,
                  facecolor="#1a1a2e", labelcolor="white")
        ax.grid(True, color="#222244", linewidth=0.5, alpha=0.7)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"vibration_comparison_{anomaly_type}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.show()


# ─────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────

def run_simulation(output_base: Path = Path("data/vibration")) -> dict[str, pd.DataFrame]:
    """
    Ejecuta la simulación completa para los tres tipos de anomalía.
    Guarda CSVs y figuras. Retorna un dict con todos los DataFrames.
    """
    print("\n" + "═" * 55)
    print("  TELAR MONITOR — Simulación de Vibraciones")
    print("═" * 55)

    anomaly_types = ["frequency", "spike", "drift"]
    results = {}

    # Señal base normal (se reutiliza en todos los experimentos)
    print("\n 1: Generando señal normal base...")
    t, normal_signal = generate_normal_signal()

    # Guardar señal normal
    df_normal = save_to_csv(
        t, normal_signal,
        np.zeros(len(t), dtype=bool),
        label="normal",
        output_dir=output_base
    )
    results["normal"] = df_normal

    # Por cada tipo de anomalía: inyectar, guardar, visualizar
    for i, atype in enumerate(anomaly_types, start=2):
        print(f"\n{i+1}: Procesando anomalía: {atype.upper()}...")

        faulty_signal, mask = inject_anomaly(
            t, normal_signal,
            start_pct=0.45,
            duration_pct=0.35,
            anomaly_type=atype
        )

        df_faulty = save_to_csv(
            t, faulty_signal, mask,
            label=f"faulty_{atype}",
            output_dir=output_base
        )
        results[f"faulty_{atype}"] = df_faulty

        plot_signals(t, normal_signal, faulty_signal, mask, atype, output_base)

    # Resumen estadístico con Pandas
    print("\n" + "─" * 55)
    print("  RESUMEN ESTADÍSTICO")
    print("─" * 55)
    summary_rows = []
    for name, df in results.items():
        summary_rows.append({
            "experimento":     name,
            "muestras":        len(df),
            "amp_media":       df["amplitude"].mean().round(4),
            "amp_std":         df["amplitude"].std().round(4),
            "amp_max":         df["amplitude"].max().round(4),
            "pct_anomalia":    (df["is_anomaly"].sum() / len(df) * 100).round(1)
        })

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_path = output_base / "summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n Resumen guardado: {summary_path}")
    print("═" * 55 + "\n")

    return results


# ─────────────────────────────────────────────
# PUNTO DE ENTRADA
# ─────────────────────────────────────────────

if __name__ == "__main__":
    results = run_simulation(output_base=Path("data/vibration"))