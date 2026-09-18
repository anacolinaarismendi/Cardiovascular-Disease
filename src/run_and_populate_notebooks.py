"""
Ejecutor y poblador de outputs de los Notebooks 01, 02 y 03.
Captura stdout, gráficos de matplotlib (como PNG base64) y DataFrames.
"""

import os
os.environ['MPLCONFIGDIR'] = '/tmp'
import json
import io
import base64
import contextlib
import matplotlib.pyplot as plt
import pandas as pd

def run_notebook(nb_path):
    print(f"\n==========================================")
    print(f"Ejecutando y poblando: {nb_path}")
    print(f"==========================================")

    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Namespace aislado por notebook
    ns = {
        '__name__': '__main__',
        'display': lambda x: print(x)
    }

    # Cambiar temporalmente el directorio al de Notebooks
    orig_cwd = os.getcwd()
    nb_dir = os.path.dirname(os.path.abspath(nb_path))
    os.chdir(nb_dir)

    exec_count = 1

    try:
        for i, cell in enumerate(nb['cells']):
            if cell['cell_type'] == 'code':
                code_text = "".join(cell['source'])
                stdout_buf = io.StringIO()
                cell_outputs = []

                # Limpiar figuras previas
                plt.close('all')

                try:
                    with contextlib.redirect_stdout(stdout_buf):
                        exec(code_text, ns)
                except Exception as e:
                    print(f"  [!] Error en celda {i+1}: {e}")
                    stdout_str = stdout_buf.getvalue()
                    if stdout_str:
                        cell_outputs.append({
                            "name": "stdout",
                            "output_type": "stream",
                            "text": [line + "\n" for line in stdout_str.split("\n") if line]
                        })
                    cell_outputs.append({
                        "name": "stderr",
                        "output_type": "stream",
                        "text": [f"Error: {e}\n"]
                    })
                    cell['execution_count'] = exec_count
                    cell['outputs'] = cell_outputs
                    exec_count += 1
                    continue

                stdout_str = stdout_buf.getvalue()
                if stdout_str:
                    lines = [l + "\n" for l in stdout_str.splitlines()]
                    cell_outputs.append({
                        "name": "stdout",
                        "output_type": "stream",
                        "text": lines
                    })

                # Capturar figuras de matplotlib si existen
                fig_nums = plt.get_fignums()
                for fig_num in fig_nums:
                    fig = plt.figure(fig_num)
                    img_buf = io.BytesIO()
                    fig.savefig(img_buf, format='png', bbox_inches='tight', dpi=120)
                    img_buf.seek(0)
                    b64_data = base64.b64encode(img_buf.read()).decode('utf-8')
                    cell_outputs.append({
                        "data": {
                            "image/png": b64_data,
                            "text/plain": [f"<Figure size {fig.get_size_inches()[0]*100}x{fig.get_size_inches()[1]*100}>"]
                        },
                        "metadata": {},
                        "output_type": "display_data"
                    })
                    plt.close(fig)

                cell['execution_count'] = exec_count
                cell['outputs'] = cell_outputs
                exec_count += 1
                print(f"  ✓ Celda {i+1} ejecutada con éxito.")

    finally:
        os.chdir(orig_cwd)

    # Guardar notebook actualizado
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print(f"✓ Notebook {nb_path} guardado con todos los outputs.")

if __name__ == '__main__':
    notebooks = [
        'Notebooks/01_exploracion.ipynb',
        'Notebooks/02_preprocesamiento.ipynb',
        'Notebooks/03_eda.ipynb'
    ]
    for nb in notebooks:
        if os.path.exists(nb):
            run_notebook(nb)
