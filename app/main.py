from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
import numpy as np
from typing import List
import matplotlib
import matplotlib.pyplot as plt
import googlenet 
import json

matplotlib.use('Agg')  # Usar backend no interactivo
app = FastAPI()

# Definir el modelo para el vector
class VectorF(BaseModel):
    vector: List[float]
    
@app.post("/googlenet")
def calculo(samples: int, test_samples: int, n_neighbors: int):
    output_file = 'googlenet.png'
    
    # Crear una instancia de GoogleNet
    model = googlenet.GoogleNet()

    # Datos de ejemplo (imagen de 28x28, similar a MNIST)
    input_image = np.random.rand(28, 28)

    # Realizar el forward pass
    output = model.forward(input_image)

    # Graficar la salida
    plt.plot(output, marker='o')
    plt.title("Salida de GoogleNet inspirado en LeNet")
    plt.xlabel("Clase")
    plt.ylabel("Valor de salida")
    plt.grid(True)
    #plt.show()

    plt.savefig(output_file)
    plt.close()
    
    j1 = {
        "Grafica": output_file
    }
    jj = json.dumps(str(j1))

    return jj

@app.get("/googlenet-graph")
def getGraph(output_file: str):
    return FileResponse(output_file, media_type="image/png", filename=output_file)
