# Planeacion del proyecto

Objetivo: lograr una deteccion de jugadores lo mas parecida posible en apariencia a un detector moderno,
manteniendo una arquitectura manual, explicable y sin modelos preentrenados de deteccion.

## Fase 1 - Base manual robusta

Estado: completada.

Entregables:

- Mascara de cancha por HSV combinada con indice de verde `ExG`.
- Limpieza morfologica de la cancha con apertura, cierre, dilatacion y erosion.
- Estabilizacion temporal de la mascara de cancha para evitar parpadeos.
- Sustraccion de fondo restringida a la cancha.
- Rechazo temprano de lineas blancas y sombras sobre cesped.
- Supresion de cajas duplicadas con NMS antes del tracking.

Criterio de aceptacion:

- La mascara cubre la cancha en la mayoria de frames sin tragarse graderia/publicidad.
- Los candidatos iniciales aparecen principalmente sobre jugadores.
- El tracker recibe menos cajas duplicadas o cajas falsas sobre lineas.

## Fase 2 - Filtros finos de jugador

Estado: completada.

Entregables:

- Ajuste por perspectiva usando posicion vertical del jugador.
- Filtros por area, relacion alto/ancho, ancho relativo y relleno de contorno.
- Filtros por porcentaje de pixeles no verdes, blancos y sombra.
- Ajustes separados para jugadores lejanos, medios y cercanos.
- Parametros `*-far` y `*-near` para calibrar el detector sin recompilar.

Criterio de aceptacion:

- Menos falsos positivos en cesped vacio, lineas y sombras.
- Menos perdida de jugadores pequenos en zona lejana.

## Fase 3 - Separacion de jugadores pegados

Estado: completada.

Entregables:

- Deteccion de blobs anormalmente anchos.
- Division por proyeccion vertical.
- Refinamiento por componentes conectados cuando la proyeccion no alcance.
- Validacion geometrica de cada subcaja.
- Parametros `--split-*` para controlar agresividad de separacion.

Criterio de aceptacion:

- En jugadas con grupos, el sistema separa al menos parte de los jugadores pegados.
- El tracker mantiene IDs mas estables despues de cruces.

## Fase 4 - HOG + SVM desde cero

Estado: en progreso.

Entregables:

- Dataset local `positives/` y `negatives/`.
- Herramienta `dataset_cropper` para convertir anotaciones MOT en recortes.
- Extraccion HOG a 64x128.
- Entrenamiento SVM lineal con OpenCV.
- Validador opcional sobre candidatos manuales.
- Hard negative mining con falsos positivos del propio sistema.

Criterio de aceptacion:

- El SVM reduce falsos positivos sin eliminar demasiados jugadores reales.
- El entrenamiento es reproducible y no usa modelos preentrenados.

## Fase 5 - Evaluacion y presentacion

Entregables:

- Videos comparativos antes/despues.
- Tabla de falsos positivos por minuto.
- Conteo promedio de jugadores detectados por frame.
- Casos documentados: sombras, lineas, jugadores pegados, jugadores lejanos.

Criterio de aceptacion:

- El proyecto demuestra mejoras medibles y explicables.
- La defensa tecnica puede justificar cada paso sin recurrir a deep learning.
