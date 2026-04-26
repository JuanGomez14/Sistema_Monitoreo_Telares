#Telar Monitor — Sistema de Monitoreo Industrial

> Sistema de control de calidad y mantenimiento predictivo para telares industriales,  
> combinando análisis de señales de vibración con visión artificial.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green)
![NumPy](https://img.shields.io/badge/NumPy-1.24+-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.x-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Descripción

**Telar Monitor** correlaciona dos fuentes de datos en tiempo real:

| Fuente | Qué detecta | Herramienta |
|---|---|---|
| Señal de vibración | Fallos mecánicos (frecuencia, golpes, desgaste) | NumPy + FFT |
| Imágenes de la tela | Defectos visuales (huecos, cortes, contaminación) | OpenCV |
| Motor de correlación | Fallos confirmados por ambas fuentes | Pandas |

La premisa central: **una máquina que falla mecánicamente también produce tela defectuosa**.  
Cuando ambas fuentes coinciden, la alerta es de alta confianza y requiere acción inmediata.

---

## Arquitectura del sistema

```
[Simulador de vibración]          [Imágenes MVTec AD]
         │                                 │
         ▼                                 ▼
   [FFT + Pandas]                 [OpenCV Pipeline]
   • Freq. anómala                • Detección de bordes
   • Picos de amplitud            • Análisis de parches
   • Desgaste gradual             • Score de defecto
         │                                 │
         └──────────────┬──────────────────┘
                        ▼
           [Motor de Correlación]
           • Línea de tiempo unificada
           • 4 niveles de alerta
                        │
                        ▼
           [Dashboard Matplotlib]
```

---

## Instalación y uso

### Requisitos

```bash
pip install numpy pandas matplotlib opencv-python
```

### Uso básico — solo vibración y FFT

```bash
python main.py --skip-vision
```

### Uso completo — con dataset MVTec

```bash
python main.py --mvtec "ruta/a/mvtec"
```

### Solo correlación — si ya corriste las fases anteriores

```bash
python main.py --only-correlation
```

---

## Estructura del proyecto

```
telar_monitor/
├── Vibracion_Sim.py        # Simulación de vibraciones  
├── FFT_analysys.py         # Análisis FFT
├── Vision_analysis.py      # Visión artificial
└── correlacion.py           # Correlación y alertas
├── Main.py                 # Punto de entrada unificado
├── data/                   # Generado al ejecutar 
│   ├── vibration/
│   ├── analysis/
│   ├── vision_output/
│   └── dashboard/
└── README.md
```

---

## Resultados obtenidos

### Detección de vibraciones

| Tipo de fallo | Método | Tasa de detección |
|---|---|---|
| Frecuencia anómala | FFT — pico en 80Hz | 42.1% de ventanas |
| Golpes (spike) | Umbral de amplitud | 42.1% de ventanas |
| Desgaste (drift) | Energía rodante | 23.7% de ventanas |

> El drift es el más difícil de detectar con métodos clásicos — requiere ML para mejores resultados.

### Detección visual 

| Categoría | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Grid** | 79.2% | 79.2% | 100% | **88.4%** |
| **Carpet** | 41.1% | 78.9% | 40.0% | 53.1% |

**Nota sobre carpet:** La textura orgánica densa de carpet presenta un desafío conocido como *domain shift* — las métricas globales (Canny, Sobel) no capturan defectos sutiles. Se implementó análisis por parches locales como solución parcial. Para superar ~65% F1 se requiere un modelo de deep learning (PatchCore, AutoEncoder Convolucional).

### Correlación

| Combinación | Alerta Alta | Normal |
|---|---|---|
| frequency + grid | 40% del tiempo | 23% |
| spike + grid | 40% del tiempo | 23% |
| drift + grid | 22% del tiempo | 31% |

---

## Conceptos técnicos implementados

- **FFT (Fast Fourier Transform)** — transformación de señal temporal a dominio de frecuencia
- **Sliding window** — análisis espectral evolutivo con ventana deslizante
- **Ventana de Hanning** — prevención de spectral leakage en la FFT
- **Canny Edge Detection** — detección de bordes en imágenes de tela
- **Sobel Gradient** — gradiente de intensidad para texturas regulares
- **Análisis de parches (patch-based)** — base de algoritmos como PatchCore y PaDiM
- **Umbral adaptativo** — calibración automática por tipo de material
- **Correlación multimodal** — fusión de señal vibratoria e imagen

---

## Dataset

Este proyecto utiliza el **MVTec AD Dataset** para las categorías `carpet` y `grid`.

- **Descarga:** https://www.mvtec.com/company/research/datasets/mvtec-ad
- **Licencia:** CC BY-NC-SA 4.0 (uso académico y de investigación)
- **Referencia:** Paul Bergmann et al., "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", CVPR 2019

> El dataset **no está incluido en este repositorio**. Descárgalo y pasa la ruta con `--mvtec`, coloca la ruta en el código modificando (\Datos_Imagenes_Tela)

---

---

## Autor

**Juan de Jesús Gómez López**  
Universidad Autónoma Metropolitana Unidad Lerma

Ingeniería en Sistemas Mecatrónicos

---

## 📄 Licencia

MIT License — libre para uso académico y personal.
