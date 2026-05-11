# Tactical Vision

## Tecnologias, librerias utilizadas y correcciones del sistema

**Proyecto:** Tactical Vision  
**Area:** Procesamiento de imagenes aplicado al analisis tactico de futbol  
**Stack principal:** C++, OpenCV, YOLO ONNX, homografia, tracking y metricas colectivas  
**Objetivo:** convertir video comun de futbol en informacion tactica accionable y exportable.

---

## Resumen ejecutivo

Tactical Vision es un sistema de analisis tactico de futbol construido sobre un pipeline hibrido. El sistema usa YOLO como apoyo para generar candidatos visuales iniciales, pero no depende exclusivamente de este modelo para producir la salida tactica final.

Despues de la deteccion inicial, el sistema aplica filtros de contexto de cancha, tracking temporal, clasificacion de equipos por color, homografia a una cancha de 105x68 metros, render visual y exportacion de metricas en CSV.

La idea central es mantener una solucion accesible, explicable y util para futbol formativo, amateur o universitario, donde normalmente hay video, pero no hay sistemas profesionales de medicion como EPTS.

---

## Tabla de contenido

1. [Arquitectura general](#1-arquitectura-general)
2. [Tecnologias y librerias utilizadas](#2-tecnologias-y-librerias-utilizadas)
3. [Pipeline tecnico](#3-pipeline-tecnico)
4. [Errores encontrados y correcciones](#4-errores-encontrados-y-correcciones)
5. [Matriz de problemas y soluciones](#5-matriz-de-problemas-y-soluciones)
6. [Diferencia entre deteccion visual y motor tactico](#6-diferencia-entre-deteccion-visual-y-motor-tactico)
7. [Conclusiones tecnicas](#7-conclusiones-tecnicas)

---

## 1. Arquitectura general

El sistema se divide en dos capas principales:

| Capa | Funcion | Resultado |
| --- | --- | --- |
| Deteccion visual | Encontrar jugadores, balon y elementos relevantes en el video. | Candidatos visuales por frame. |
| Motor tactico | Interpretar posiciones, equipos, zonas y comportamiento colectivo. | Metricas tacticas, minimapa, panel y CSV. |

Esta separacion permite explicar mejor el proyecto: YOLO ayuda a detectar, pero la interpretacion tactica se construye con etapas adicionales y verificables.

---

## 2. Tecnologias y librerias utilizadas

| Tecnologia / libreria | Uso en el proyecto | Por que se utilizo |
| --- | --- | --- |
| **C++** | Lenguaje principal del sistema. | Permite procesar video frame por frame con buen rendimiento y control sobre memoria, estructuras y flujo. |
| **CMake** | Configuracion de compilacion. | Organiza el proyecto, facilita builds en Windows y permite separar ejecutables, librerias internas y pruebas. |
| **OpenCV** | Procesamiento de imagenes, video, matrices, dibujo, morfologia, contornos, homografia y DNN. | Es la base del pipeline de vision por computador y permite combinar tecnicas clasicas con inferencia de modelos. |
| **OpenCV DNN / YOLO ONNX** | Deteccion inicial de jugadores y balon. | YOLO ayuda a proponer candidatos visuales en video real, especialmente cuando la escena es compleja. |
| **NMS** | Eliminacion de cajas duplicadas. | Reduce detecciones repetidas sobre el mismo jugador antes del tracking. |
| **HSV** | Segmentacion por color de cancha, camisetas y posibles arbitros. | Es mas estable que BGR ante cambios de iluminacion porque separa tono, saturacion y brillo. |
| **Morfologia matematica** | Apertura, cierre, dilatacion y erosion sobre mascaras. | Limpia ruido, une regiones utiles de cancha y reduce falsos positivos pequenos. |
| **Canny** | Deteccion de bordes para lineas, contornos y fallback del balon. | Ayuda a localizar objetos pequenos o bordes definidos cuando la deteccion principal falla. |
| **Contornos** | Extraccion y filtrado de regiones candidatas. | Permite validar objetos por area, forma, circularidad, posicion y proporcion. |
| **KMeans** | Clasificacion de equipo por color dominante de camiseta. | Agrupa colores del torso para separar equipos sin etiquetar manualmente cada jugador. |
| **Tracking por centroides** | Seguimiento temporal entre frames. | Mantiene identidad aproximada, reduce parpadeos y permite analizar continuidad y saltos. |
| **Optical Flow Lucas-Kanade** | Estimacion de movimiento de camara. | Ayuda a compensar desplazamientos globales cuando la camara se mueve. |
| **Homografia** | Proyeccion de la imagen a una cancha 2D de 105x68 m. | Convierte posiciones de imagen en coordenadas tacticas comparables sobre campo real. |
| **CSV** | Exportacion de metricas por frame. | Permite analizar datos despues de la demo y tener evidencia exportable. |
| **HTML/CSS/JavaScript** | Presentacion visual del proyecto en `index.html`. | Muestra problema, solucion, demo, pipeline, videos y tecnologias de manera clara. |

---

## 3. Pipeline tecnico

```text
Video de entrada
  -> resize del frame
  -> deteccion inicial con YOLO ONNX
  -> NMS para reducir duplicados
  -> filtros de contexto de cancha
  -> rechazo de falsos positivos
  -> tracking temporal
  -> clasificacion de equipo por color
  -> deteccion/fallback de balon
  -> homografia a cancha 105x68 m
  -> calculo de metricas colectivas
  -> render de video + minimapa + panel
  -> exportacion CSV por frame
```

### Metricas generadas

| Metrica | Descripcion | Valor tactico |
| --- | --- | --- |
| Conteo por equipo | Numero de jugadores detectados por frame. | Ayuda a verificar cobertura visual y estabilidad. |
| Ocupacion por tercios | Distribucion del equipo en defensa, medio y ataque. | Permite interpretar altura y progresion. |
| Ocupacion por carriles | Uso de carril izquierdo, central y derecho. | Ayuda a entender amplitud y zonas de ataque. |
| Bloque LOW/MID/HIGH | Altura colectiva del equipo. | Resume si el equipo esta bajo, medio o alto. |
| Ancho y largo | Dimensiones ocupadas por el equipo. | Mide amplitud, profundidad y compactacion. |
| Posesion estimada | Equipo probablemente mas cercano o relacionado con el balon. | Resume dominio momentaneo del juego. |
| Ataque/defensa probable | Estado tactico inferido desde posiciones y balon. | Ayuda a resumir la fase colectiva. |

---

## 4. Errores encontrados y correcciones

### 4.1 Publico reconocido como jugador

**Problema:**  
En algunos frames, personas del publico, banca o zonas externas al terreno eran detectadas como jugadores.

**Causa probable:**  
El detector encontraba siluetas humanas o regiones con colores parecidos a uniformes, aunque estuvieran fuera del campo.

**Correcciones aplicadas o propuestas:**

- Restringir detecciones al contexto de cancha usando mascara verde HSV.
- Validar que la parte inferior de la caja tenga contacto con la cancha.
- Rechazar detecciones ubicadas claramente fuera del plano de juego.
- Aplicar filtros geometricos: area, relacion alto/ancho y posicion vertical.
- Usar limites de cancha u homografia para descartar actores externos.

**Resultado esperado:**  
Menos falsos positivos sobre publico, banca, publicidad y zonas externas al terreno.

---

### 4.2 La pelota no se reconoce de forma estable

**Problema:**  
La pelota es pequena, se mueve rapido y puede quedar borrosa, oculta o mezclada con lineas blancas.

**Causa probable:**  
El balon ocupa pocos pixeles, cambia de posicion rapidamente y puede confundirse con marcas, botas o ruido.

**Correcciones aplicadas o propuestas:**

- Usar YOLO como primera opcion cuando exista un candidato plausible.
- Agregar fallback clasico con Canny, contornos, circularidad y tamano esperado.
- Restringir la busqueda del balon a la zona de cancha.
- Exigir estabilidad temporal antes de aceptar puntos aislados.
- Dar prioridad a candidatos cercanos a jugadores cuando haya varios puntos blancos.

**Resultado esperado:**  
Mayor estabilidad en la deteccion del balon y menos confusiones con lineas, brillos o ruido.

---

### 4.3 Falsos positivos dentro del campo

**Problema:**  
Lineas, sombras, manchas del cesped o partes de la cancha podian detectarse como jugadores.

**Causa probable:**  
Algunas regiones del campo cumplen condiciones de color, borde o movimiento parecidas a un jugador.

**Correcciones aplicadas o propuestas:**

- Aplicar mascara de cancha para procesar solo areas utiles.
- Filtrar regiones demasiado verdes, demasiado blancas o con forma incompatible.
- Rechazar cajas con relacion alto/ancho poco realista.
- Usar NMS para evitar multiples cajas sobre una misma region.
- Ajustar umbrales por perspectiva para jugadores lejanos y cercanos.

**Resultado esperado:**  
Menos detecciones sobre lineas, cesped vacio, sombras y ruido visual.

---

### 4.4 Arbitro confundido con jugador

**Problema:**  
El arbitro podia entrar en los conteos de jugadores o afectar la clasificacion por equipo.

**Causa probable:**  
El arbitro comparte forma humana con los jugadores, pero no pertenece a ningun equipo.

**Correcciones aplicadas o propuestas:**

- Usar filtro cromatico en HSV para detectar colores caracteristicos del arbitro.
- Separar al arbitro antes de calcular conteos por equipo.
- Evitar que el color del arbitro contamine el KMeans de camisetas.

**Resultado esperado:**  
Conteos por equipo mas confiables y menor ruido en metricas tacticas.

---

### 4.5 Cambios de camara y saltos en tracking

**Problema:**  
Cuando la camara se mueve, algunos tracks pueden saltar o perder continuidad.

**Causa probable:**  
El movimiento global de camara cambia la posicion aparente de todos los jugadores.

**Correcciones aplicadas o propuestas:**

- Estimar movimiento de camara con Optical Flow Lucas-Kanade.
- Evaluar saltos imposibles en coordenadas de campo.
- Ignorar tracks con continuidad insuficiente antes de calcular ocupacion tactica.
- Usar velocidad fisica maxima como filtro de consistencia.

**Resultado esperado:**  
Tracking mas estable y metricas menos sensibles al movimiento de camara.

---

### 4.6 Homografia poco confiable

**Problema:**  
Si la camara no muestra suficiente cancha o la calibracion es incorrecta, las posiciones proyectadas pueden ser imprecisas.

**Causa probable:**  
La homografia depende de puntos de referencia bien ubicados y de un encuadre compatible con el plano de cancha.

**Correcciones aplicadas o propuestas:**

- Validar que los puntos proyectados esten dentro de la cancha 105x68 m.
- Medir error de reproyeccion cuando haya puntos de referencia.
- Etiquetar la calidad espacial como HIGH, MEDIUM, LOW o INVALID.
- No publicar metricas espaciales finas cuando la calidad sea baja.

**Resultado esperado:**  
Metricas tacticas mas honestas y explicables, especialmente en videos con encuadre dificil.

---

## 5. Matriz de problemas y soluciones

| Error observado | Impacto | Correccion principal | Estado recomendado |
| --- | --- | --- | --- |
| Publico detectado como jugador | Aumenta falsos positivos y altera conteos. | Mascara de cancha, validacion de contacto con campo y limites por homografia. | Mantener y ajustar por video. |
| Balon no detectado | Afecta posesion y fase de juego. | YOLO + fallback Canny/contornos/circularidad. | Mejorar con estabilidad temporal. |
| Falsos positivos en campo | Ensucia tracking y metricas. | Filtros por color, forma, area y perspectiva. | Seguir calibrando umbrales. |
| Arbitro como jugador | Distorsiona conteos por equipo. | Filtro HSV y exclusion antes de KMeans. | Validar segun color de uniforme. |
| Saltos de tracking | Rompe trayectorias y velocidades. | Optical flow, continuidad minima y velocidad maxima. | Integrar como confianza de tracking. |
| Homografia inestable | Proyecta mal posiciones tacticas. | Error de reproyeccion y estados de confianza. | Publicar metricas solo con calidad suficiente. |

---

## 6. Diferencia entre deteccion visual y motor tactico

Una decision clave del proyecto es separar la deteccion visual del motor tactico.

| Componente | Pregunta que responde |
| --- | --- |
| Deteccion visual | Donde parecen estar los jugadores y el balon. |
| Motor tactico | Que significa esa distribucion para el equipo. |

Por eso YOLO no es la dependencia total del sistema. YOLO propone candidatos, pero las metricas finales se apoyan en:

- Filtros de contexto de cancha.
- Tracking temporal.
- Clasificacion de equipos por color.
- Homografia.
- Validacion de confianza espacial.
- Reglas colectivas interpretables.

Frase recomendada para la sustentacion:

> YOLO se usa como generador de candidatos visuales, pero la salida tactica final depende de etapas adicionales: filtrado contextual, tracking, clasificacion de equipo, proyeccion geometrica y reglas colectivas.

---

## 7. Conclusiones tecnicas

Tactical Vision no busca reemplazar sistemas EPTS profesionales. Su valor esta en convertir video comun en informacion tactica accesible, reproducible y util para contextos formativos, amateur o universitarios.

Las tecnologias se eligieron porque permiten equilibrar rendimiento, explicabilidad y facilidad de validacion. La combinacion de OpenCV, YOLO, reglas visuales, tracking y homografia permite construir una solucion hibrida donde la IA apoya, pero la logica tactica sigue siendo comprensible y defendible.

