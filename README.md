# Football Analytics CV - Detector Manual Clasico

Este proyecto implementa analisis de futbol con vision por computador clasica en C++ y OpenCV.
La deteccion de jugadores es manual y explicable: no usa YOLO, Faster R-CNN, Mask R-CNN,
modelos preentrenados ni detectores deep learning.

El objetivo del sistema es transformar posiciones detectadas manualmente en informacion tactica
util mediante homografia, tracking clasico, metricas espaciales y reglas estrategicas explicables.

Flujo principal:

```text
Frame
  -> mascara de cancha por HSV
  -> sustraccion de fondo
  -> contornos
  -> filtros manuales
  -> HOG+SVM opcional
  -> detecciones finales
  -> ByteTrack / DeepSORT
  -> clasificacion de equipo por color
  -> homografia a cancha 2D 105x68 m
  -> metricas tacticas
  -> visualizacion
  -> recomendaciones estrategicas
```

## Estructura

- `src/main.cpp`: pipeline principal.
- `include/detection/` y `src/detection/`: detector clasico manual.
- `dataset/positives/`: recortes etiquetados de jugadores para HOG+SVM.
- `dataset/negatives/`: cesped, lineas, sombras, balon, publico, publicidad y falsos positivos.
- `include/tracking/` y `src/tracking/`: ByteTrack y DeepSORT.
- `include/reid/` y `src/reid/`: ReID simple por histograma/color.
- `include/analysis/` y `src/analysis/`: analisis tactico basico.
- `include/analysis/AdvancedTactics.hpp` y `src/analysis/AdvancedTactics.cpp`: homografia,
  minimapa, clasificacion por color, exportacion CSV, heatmaps, formaciones, lineas de pase,
  control de espacios y recomendaciones por reglas.

## Arquitectura tactica explicable

Los modulos tacticos agregados son:

- `FieldHomography`: permite seleccionar manualmente 4 puntos de la cancha en la imagen,
  relacionarlos con una cancha real de 105x68 metros, calcular `cv::findHomography`, proyectar
  jugadores con `cv::perspectiveTransform` y guardar/cargar la matriz en YAML.
- `TacticalMinimap`: dibuja una vista superior 2D de la cancha con jugadores por equipo, IDs,
  trayectorias recientes, balon si esta disponible, lineas de pase, mapas de calor y zonas de control.
- `TeamColorClassifier`: estima el equipo desde el color dominante del torso en HSV, entrena dos
  clusters con K-means y estabiliza la decision con voto temporal por `track_id`.
- `TrackingDataExporter`: exporta `frame,timestamp,player_id,team_id,x_image,y_image,x_field,y_field,speed,vx,vy`.
- `HeatmapAnalyzer`: acumula presencia por equipo y jugador sobre coordenadas reales del campo,
  suaviza con `GaussianBlur` y lo visualiza sobre el minimapa.
- `FormationAnalyzer`: calcula ancho, profundidad, compactacion por convex hull, lineas defensiva,
  media y ofensiva, y detecta equipos largos, estrechos o desordenados.
- `PassingLaneAnalyzer`: evalua lineas de pase entre companeros; una linea se considera bloqueada
  si un rival esta cerca del segmento de pase.
- `SpaceControlAnalyzer`: implementa una version simple de Voronoi por equipo sobre una grilla,
  usando distancia al jugador mas cercano y velocidad estimada.
- `StrategyAdvisor`: genera recomendaciones tacticas con reglas simples y defendibles, por ejemplo
  reducir distancia entre lineas, atacar verticalmente, aprovechar superioridades por banda,
  progresar por lineas de pase abiertas o presionar cuando el rival tiene pocas opciones.

## Detector manual

El detector `ClassicDetector` combina:

- Mascara de cancha en HSV con rangos de verde amplios y estrictos.
- Indice de verde `ExG` para tolerar mejor cambios de iluminacion.
- Apertura, cierre, dilatacion, erosion y relleno mediante contorno principal + convex hull.
- Estabilizacion temporal de la mascara de cancha para evitar parpadeos.
- Sustraccion de fondo MOG2 o Canny opcional.
- Contornos y filtros por area, alto/ancho, ancho relativo, relleno, movimiento y perspectiva.
- Filtro por posicion dentro de la cancha y por apoyo del borde inferior en la mascara.
- Porcentaje minimo de pixeles no verdes para descartar cesped vacio.
- Rechazo de lineas blancas, sombras, objetos muy pequenos y regiones demasiado anchas.
- Separacion de blobs anchos por proyeccion vertical; si no hay valle claro, divide por ancho esperado.
- Supresion de cajas duplicadas antes del tracking.

El SVM opcional no detecta por ventana deslizante. Solo valida candidatos ya generados por el
detector manual.

Las alertas del sistema son visuales dentro de la interfaz y del video exportado. No se usan
pitidos ni sintesis de voz.

## Compilacion

```bash
mkdir build
cd build
cmake ..
make
```

En Windows con CMake multi-config:

```powershell
cmake -S . -B build_vcpkg
cmake --build build_vcpkg --config Release
ctest --test-dir build_vcpkg -C Release --output-on-failure
```

## Ejecucion

Coloca un video llamado `input.mp4` en la raiz y ejecuta:

```bash
./football_analytics
```

Tambien puedes configurar el pipeline:

```bash
./football_analytics --input input.mp4 --output output.avi --detector classic --tracker bytetrack
./football_analytics --input input.mp4 --output output.avi --detector classic --tracker deepsort
```

Para transmisiones con jugadores pequenos o poco movimiento, usa el perfil sensible. Sigue usando
deteccion clasica manual, sin librerias externas de deteccion de jugadores:

```bash
./football_analytics --input input.mp4 --output output.avi --profile sensitive
```

Calibrar homografia manualmente desde el primer frame:

```powershell
.\build_470\Release\football_analytics.exe `
  --input input.mp4 `
  --calibrate-homography 1 `
  --homography-file dataset\homography.yml
```

Durante la calibracion selecciona 4 esquinas de la cancha en este orden:
superior izquierda, superior derecha, inferior derecha, inferior izquierda. Esos puntos se mapean
a coordenadas reales `0,0`, `105,0`, `105,68`, `0,68`.

Ejecutar el analisis tactico completo con minimapa, lineas de pase, control de espacios,
heatmaps, recomendaciones y CSV:

```powershell
.\build_470\Release\football_analytics.exe `
  --input input.mp4 `
  --hog-svm 1 `
  --svm-model dataset\hog_svm.yml `
  --homography-file dataset\homography.yml `
  --export-csv 1 `
  --csv dataset\tracking_export.csv
```

Parametros disponibles:

- `--help`
- `--input <path>`
- `--output <path>`
- `--profile <default|sensitive>`
- `--detector classic`
- `--tracker bytetrack|deepsort`
- `--min-area <int>`
- `--max-area <int>`
- `--min-area-far <int>`
- `--min-area-near <int>`
- `--max-area-far <int>`
- `--max-area-near <int>`
- `--blur <int>`
- `--morph <int>`
- `--equalize <0|1>`
- `--use-canny <0|1>`
- `--canny1 <int>`
- `--canny2 <int>`
- `--bg-history <int>`
- `--bg-var <float>`
- `--field-erode <int>`
- `--color-candidates <0|1>`
- `--min-non-green <float>`
- `--min-non-green-far <float>`
- `--min-non-green-near <float>`
- `--min-h-rel <float>`
- `--max-h-rel <float>`
- `--min-field-overlap <float>`
- `--min-bottom-field-support <float>`
- `--min-ground-green-support <float>`
- `--require-bottom-on-field <0|1>`
- `--horizon-rel <float>`
- `--min-h-rel-near <float>`
- `--max-h-rel-near <float>`
- `--min-h-rel-far <float>`
- `--max-h-rel-far <float>`
- `--confirm-frames <int>`
- `--max-candidate-miss <int>`
- `--confirm-iou <float>`
- `--min-motion-far <float>`
- `--min-motion-near <float>`
- `--min-fill-far <float>`
- `--min-fill-near <float>`
- `--max-width-rel-far <float>`
- `--max-width-rel-near <float>`
- `--max-wide-aspect-far <float>`
- `--max-wide-aspect-near <float>`
- `--split-wide-aspect <float>`
- `--split-force-aspect <float>`
- `--split-valley-ratio <float>`
- `--split-min-component-area-ratio <float>`
- `--split-max-parts <int>`
- `--drop-unknown <0|1>`
- `--team-auto <0|1>`
- `--team-warmup <int>`
- `--team-min-samples <int>`
- `--verbose <0|1>` muestra log por frame
- `--hog-svm <0|1>` activa el validador HOG+SVM opcional
- `--svm-model <path>` ruta del modelo SVM entrenado desde cero
- `--svm-positives <dir>` carpeta de recortes positivos
- `--svm-negatives <dir>` carpeta de recortes negativos
- `--train-svm <0|1>` entrena el SVM antes de procesar video
- `--train-svm-only <0|1>` solo entrena y sale
- `--save-hard-negatives <0|1>` guarda candidatos rechazados por SVM
- `--hard-negative-dir <dir>` destino para negativos dificiles
- `--tactical <0|1>` activa o desactiva la capa tactica avanzada
- `--calibrate-homography <0|1>` permite seleccionar manualmente puntos de cancha
- `--homography-file <path>` archivo YAML de homografia
- `--export-csv <0|1>` exporta datos tacticos por jugador/frame
- `--csv <path>` ruta del CSV exportado
- `--no-ui <0|1>` ejecuta sin ventana interactiva, util para pruebas y batch
- `--max-frames <int>` procesa solo N frames; `0` procesa todo

## HOG + SVM opcional

Estructura esperada del dataset local:

```text
dataset/
  positives/
    jugador_001.png
    jugador_002.png
  negatives/
    cesped_001.png
    linea_001.png
    sombra_001.png
    balon_001.png
    publico_001.png
    hard/
```

## Evaluacion objetiva del detector

El proyecto incluye herramientas para medir si un cambio mejora o empeora la deteccion clasica.
No usan librerias externas de deteccion de jugadores.

Exportar frames para etiquetar:

```powershell
.\build_vcpkg\Release\export_eval_frames.exe `
  --input input.mp4 `
  --output-dir dataset\eval_frames `
  --start 1 `
  --end 250 `
  --step 25
```

Etiqueta manualmente cajas de jugadores sobre esos frames y guarda un CSV en coordenadas `1280x720`.
Formato aceptado:

```csv
frame,x,y,w,h
1,430,210,24,58
```

Tambien puedes incluir ID:

```csv
frame,id,x,y,w,h
1,7,430,210,24,58
```

Evaluar el detector:

```powershell
.\build_vcpkg\Release\evaluate_detector.exe `
  --input input.mp4 `
  --labels dataset\eval_labels.csv `
  --profile sensitive `
  --iou 0.5 `
  --predictions dataset\eval_predictions.csv
```

El reporte imprime `precision`, `recall`, `f1`, `tp`, `fp`, `fn` y `mean_iou`.

Entrenar desde cero:

```bash
./football_analytics --train-svm-only 1 --svm-positives dataset/positives --svm-negatives dataset/negatives --svm-model dataset/hog_svm.yml
```

Usarlo como filtro opcional:

```bash
./football_analytics --input input.mp4 --hog-svm 1 --svm-model dataset/hog_svm.yml
```

Hard negative mining:

```bash
./football_analytics --input input.mp4 --hog-svm 1 --svm-model dataset/hog_svm.yml --save-hard-negatives 1
```

Luego revisa `dataset/negatives/hard/`, conserva los falsos positivos reales y reentrena el SVM.

SoccerNet Tracking, SportsMOT o frames propios se pueden usar solo para obtener imagenes y
anotaciones de apoyo. En este proyecto no se cargan como detectores ni aportan modelos
preentrenados.

## Crear dataset desde SoccerNet/SportsMOT

La herramienta `dataset_cropper` acepta anotaciones tipo MOTChallenge, usadas normalmente por
SportsMOT y por secuencias de tracking con formato similar:

```text
frame,id,x,y,w,h,conf,class,visibility
```

Ejemplo de uso en Windows:

```powershell
.\build_470\Release\dataset_cropper.exe `
  --images "ruta\a\secuencia\img1" `
  --annotations "ruta\a\secuencia\gt\gt.txt" `
  --output dataset `
  --person-class -1 `
  --frame-step 2 `
  --negatives-per-frame 4
```

Esto genera recortes en:

```text
dataset/positives/
dataset/negatives/
```

Despues se entrena el SVM desde cero:

```powershell
.\build_470\Release\football_analytics.exe --train-svm-only 1 --svm-positives dataset\positives --svm-negatives dataset\negatives --svm-model dataset\hog_svm.yml
```

Y se activa como validador opcional:

```powershell
.\build_470\Release\football_analytics.exe --input input.mp4 --hog-svm 1 --svm-model dataset\hog_svm.yml
```

## Requisitos

- OpenCV 4.x
- CMake 3.10+
- C++17

## Notas

- El detector usa sustraccion de fondo + mascara de area de cancha derivada del verde.
- Usa `--min-non-green` para filtrar falsos positivos sobre el cesped.
- Ajusta `--min-h-rel-*` y `--max-h-rel-*` si hay jugadores muy pequenos o muy cerca de camara.
- Ajusta los parametros `*-far` y `*-near` para tratar distinto jugadores lejanos y cercanos.
- Ajusta `--split-*` si los jugadores pegados se separan demasiado o muy poco.
- `--min-field-overlap` y `--require-bottom-on-field` ayudan a eliminar detecciones fuera de la cancha.
- `--min-bottom-field-support` y `--min-ground-green-support` reducen falsos positivos en gradas,
  vallas y publicidad exigiendo soporte de cancha/cesped cerca de los pies.
- `--horizon-rel` define donde empieza la zona lejana para modelar perspectiva.
- `--confirm-frames` y `--max-candidate-miss` agregan confirmacion temporal.
- El ReID actual es un embedding simple basado en histograma HSV.

## Proximos pasos

- Seguir la ruta de trabajo en `docs/PLANEACION_PROYECTO.md`.
- Usar `dataset_cropper` para recortar ejemplos positivos/negativos desde datasets de tracking.
- Ajustar rangos HSV por estadio e iluminacion.
- Analisis tactico avanzado: lineas, presion y cobertura.
