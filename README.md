# SubLingo 1.2

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/AI-Gemini%202.0-orange.svg" alt="AI Powered">
</p>

## 📋 Descripción

**SubLingo** es una herramienta avanzada para traducir subtítulos de forma inteligente utilizando Google Gemini AI. Diseñada especialmente para manejar archivos de subtítulos en formato ASS y SRT con características avanzadas como detección automática de idioma, traducción contextual y soporte para formato "doble origen".

## ✨ Características Principales

### 🎯 Traducción Inteligente
- **Contexto Narrativo**: Permite ingresar una descripción de la obra para mejorar la precisión y estilo de la traducción
- **Detección Automática de Idioma**: Analiza el contenido y detecta automáticamente el idioma origen
- **Soporte Doble Origen**: Maneja diálogos con formato `{texto_original} texto_traducido` para mayor precisión

### 🛠️ Funciones Avanzadas
- **Procesamiento por Lotes**: Traduce múltiples archivos de subtítulos de una vez
- **Validación y Corrección**: Verifica y corrige automáticamente los tiempos de sincronización
- **Filtrado Inteligente**: Omite líneas de créditos y texto no relevante
- **Manejo de Errores**: Sistema robusto de reintentos con back-off exponencial

### 🎨 Interfaz Visual Rica
- **Barras de Progreso**: Seguimiento visual del progreso de archivos y líneas
- **Paneles Informativos**: Muestra resultados detallados de cada archivo procesado
- **Temporizador de Back-off**: Visualización elegante de los tiempos de espera
- **Colores y Formato**: Interfaz colorida y fácil de seguir

## 🔧 Requisitos del Sistema

### Dependencias de Python
```bash
pip install pysubs2 rich langdetect google-generativeai python-dotenv
```

### Versiones Requeridas
- **Python**: 3.8 o superior
- **Sistema Operativo**: Windows, macOS, Linux

## 🚀 Instalación

1. **Clona o descarga** el archivo `SubLingo_1.2.py`

2. **Instala las dependencias**:
   ```bash
   pip install pysubs2 rich langdetect google-generativeai python-dotenv
   ```

3. **Configura la API Key de Gemini**:
   - Crea un archivo `.env` en la misma carpeta del script
   - Agrega tu API Key:
     ```
     GEMINI_API_KEY=tu_api_key_aqui
     ```

## 📖 Uso

### Ejecución Básica
```bash
python SubLingo_1.2.py
```

### Flujo de Trabajo

1. **Contexto Narrativo**: 
   - El programa te pedirá describir la obra (serie, película, etc.)
   - Esto mejora la calidad y estilo de la traducción

2. **Detección de Archivos**:
   - Busca automáticamente archivos `.ass` y `.srt` en la carpeta actual y subcarpetas
   - Muestra una lista numerada de todos los archivos encontrados

3. **Configuración de Idiomas**:
   - Detecta automáticamente el idioma origen del primer archivo
   - Permite confirmar o corregir la detección
   - Solicita el idioma destino (por defecto: español latinoamericano)

4. **Procesamiento**:
   - Traduce todos los archivos con barras de progreso en tiempo real
   - Valida y corrige automáticamente los tiempos de sincronización
   - Genera un informe detallado para cada archivo

## 📁 Estructura de Salida

Los archivos traducidos se guardan en:
```
./output_traducidos/
├── archivo1_traducido.es-419.ass
├── subcarpeta/
│   └── archivo2_traducido.es-419.srt
└── ...
```

El código de idioma (`es-419` en el ejemplo) se agrega al nombre de cada
archivo para que herramientas como **MKVToolNix** detecten automáticamente
el idioma.

La estructura de carpetas original se mantiene intacta.

## ⚙️ Configuración Avanzada

### Variables de Configuración
```python
# Modelo de AI utilizado
DEFAULT_MODEL_NAME = "gemini-2.0-flash-lite"

# Tamaño de lote para procesamiento
BATCH_SIZE = 15

# Configuración de reintentos
MAX_RETRIES = 5
INITIAL_BACKOFF_SECONDS = 5

# Tolerancia de tiempo para sincronización
TIME_TOLERANCE_MS = 5
```

### Formato Doble Origen
El programa reconoce y optimiza diálogos con este formato:
```
{Texto en idioma original} Traducción intermedia en inglés
```

Ejemplo:
```
{こんにちは世界} Hello world
```

Este formato permite una traducción más precisa al tener dos referencias.

## 🔍 Funciones Especiales

### Filtrado Inteligente
Automáticamente omite líneas que contienen:
- Créditos y agradecimientos
- URLs y enlaces
- Información de episodios
- Texto de fansubs

### Validación y Corrección
- **Tiempos de Sincronización**: Corrige automáticamente desfases temporales
- **Líneas Faltantes**: Identifica diálogos que no se tradujeron correctamente
- **Informe Detallado**: Muestra estadísticas de correcciones por archivo

## 📊 Ejemplo de Salida

```
┌─ Resultado ─────────────────────────┐
│ episodio_01.ass                     │
│ Tiempos corregidos : 3              │
│ Líneas sin traducir : 0             │
└─────────────────────────────────────┘

Proceso terminado.
Archivos traducidos en ./output_traducidos
Tiempo de ejecución: 2 min 34 s
```

## 🚨 Solución de Problemas

### Error: "Falta instalar google-generativeai"
```bash
pip install google-generativeai
```

### Error: "Se requiere API-KEY de Gemini"
- Verifica que el archivo `.env` existe
- Confirma que la variable `GEMINI_API_KEY` está configurada correctamente

### Error: "No se encontraron archivos de subtítulos"
- Verifica que los archivos tengan extensión `.ass` o `.srt`
- Asegúrate de ejecutar el script en la carpeta correcta

### Cuota de API Agotada
El programa maneja automáticamente los límites de cuota con:
- Sistema de reintentos con back-off exponencial
- Visualización del tiempo de espera
- Hasta 5 intentos por lote

## 🔒 Privacidad y Seguridad

- ✅ Las API Keys se cargan desde variables de entorno
- ✅ No se almacenan credenciales en el código
- ✅ Los archivos originales nunca se modifican
- ✅ Procesamiento local con API externa solo para traducción

## 📝 Notas Técnicas

### Rendimiento
- Procesamiento asíncrono para mayor eficiencia
- Batches de 15 líneas para optimizar las llamadas a la API
- Espera de 3 segundos entre lotes para evitar límites de velocidad

### Compatibilidad
- Soporta archivos ASS y SRT
- Codificación UTF-8 por defecto
- Compatible con estructura de carpetas anidadas

## 🤝 Contribuciones

Si encuentras bugs o tienes sugerencias de mejora, por favor:
1. Describe el problema detalladamente
2. Incluye el mensaje de error completo
3. Especifica la versión de Python y dependencias utilizadas

## 📄 Licencia

Este proyecto está bajo la Licencia MIT. Puedes usarlo libremente para proyectos personales y comerciales.

---

<p align="center">
  <strong>Desarrollado con ❤️ para la comunidad de subtítulos</strong>
</p>
