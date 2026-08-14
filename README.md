# patronesGraficos-va

Sistema de reconocimiento de patrones en gráficos mediante visión artificial

Fase 1
────────
Capturar una ventana.
Al finalizar esta fase, el proyecto deberá ser capaz de:

- localizar una ventana por su título;
- obtener su posición y tamaño;
- capturar únicamente esa ventana;
- definir opcionalmente una ROI (Región de Interés);
- mostrar la captura en tiempo real con OpenCV.

Criterio para dar por terminada la Fase 1

Consideraremos esta fase finalizada cuando podamos ejecutar el programa y ocurra lo siguiente:

✅ encuentra automáticamente la ventana indicada;
✅ funciona aunque la ventana cambie de posición;
✅ captura únicamente esa ventana;
✅ puede limitar la captura a una ROI si se configura;
✅ muestra los frames en tiempo real;
✅ el código queda organizado y documentado.

Empecemos por el Paso 1
_Desde la raíz del proyecto:_

- pip install -r requirements.txt

Fase 2
────────
Procesar la imagen.

Fase 3
────────
Detectar objetos simples.

ROI
 ↓
HSV
 ↓
máscara multicolor
 ↓
findContours()
 ↓
filtro por área
 ↓
bounding box
 ↓
ancho típico dinámico
 ↓
candidatos por ancho relativo
 ↓
posición X
 ↓
separación X típica

Fase 4
────────
Registrar resultados.

Fase 5
────────
Construir patrones.

Fase 6
────────
Analizar comportamiento.

Fase 7
────────
Experimentar con IA.
