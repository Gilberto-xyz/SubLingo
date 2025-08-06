import subprocess
import os

def listar_idiomas_en_directorio(directory):
    video_extensions = ['.mkv']
    files = os.listdir(directory)
    idiomas_encontrados = set()
    archivos_info = []

    for file_name in files:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()
        if ext in video_extensions:
            video_path = os.path.join(directory, file_name)
            cmd_probe = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 's',
                '-show_entries', 'stream=index,codec_type:stream_tags=language',
                '-of', 'json',
                video_path
            ]
            try:
                result = subprocess.run(cmd_probe, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                import json
                streams_info = json.loads(result.stdout.decode())
                if 'streams' not in streams_info:
                    continue
                for stream in streams_info['streams']:
                    if stream.get('codec_type') != 'subtitle':
                        continue
                    language = stream.get('tags', {}).get('language', 'und')
                    idiomas_encontrados.add(language)
                    archivos_info.append((file_name, stream.get('index'), language))
            except Exception:
                continue
    return sorted(idiomas_encontrados), archivos_info

def extraer_subtitulos(directory, archivos_info, idiomas_seleccionados):
    output_dir = os.path.join(directory, "subtitulos_extraidos")
    os.makedirs(output_dir, exist_ok=True)  # Crea la carpeta si no existe
    for file_name, index, language in archivos_info:
        if language not in idiomas_seleccionados:
            continue
        video_path = os.path.join(directory, file_name)
        base_name = os.path.splitext(file_name)[0]
        output_filename = f"{base_name}_{language}_sub{index}.srt"
        output_path = os.path.join(output_dir, output_filename)  # Guardar en la carpeta
        cmd_extract = [
            'ffmpeg',
            '-y',
            '-i', video_path,
            '-map', f'0:{index}',
            '-c:s', 'srt',
            output_path
        ]
        extract_result = subprocess.run(cmd_extract, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if extract_result.returncode == 0:
            print(f"Subtítulo {index} ({language}) extraído a '{output_filename}'.")
        else:
            print(f"Error al extraer subtítulo {index} ({language}).")

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    idiomas, archivos_info = listar_idiomas_en_directorio(current_directory)
    if not idiomas:
        print("No se encontraron subtítulos en los archivos del directorio.")
        exit()
    print("Idiomas encontrados:")
    for i, idioma in enumerate(idiomas):
        print(f"{i+1}. {idioma}")
    seleccion = input("Introduce los números de los idiomas a extraer separados por coma (ej: 1,3): ")
    indices = [int(x.strip())-1 for x in seleccion.split(",") if x.strip().isdigit()]
    idiomas_seleccionados = [idiomas[i] for i in indices if 0 <= i < len(idiomas)]
    print(f"Extrayendo subtítulos para: {', '.join(idiomas_seleccionados)}")
    extraer_subtitulos(current_directory, archivos_info, idiomas_seleccionados)
