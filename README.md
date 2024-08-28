# indoor-positioning-system

Este proyecto es un sistema que determina tu ubicacion aproximada dentro de un area delimitada escaneando las potencias de señal recibida de las redes de Wifi disponibles y realizando calculos con estos resultados

## Obtencion del training data

En caso desees implementar este sistema en un area diferente, puedes usar el siguiente script de python para rrealizar el escaneo de las redes wifi y guardarlos en formato CSV: [get-rssi](https://github.com/diegoroca/get-rssi)

## Instalacion

1. Clona este repositorio:

```bash
git clone https://github.com/diegoroca/indoor-positioning-system.git
cd get-rssi
```

2. Crea un entorno virtual (virtual enviroment):

```bash
python -m venv venv
source venv/bin/activate  # En Windows usa `venv\Scripts\activate`
```

3. Installa las dependencias:

```bash
pip install -r requirements.txt
```

## Uso

Ejecuta el programa main.py con permisos de administrador

```bash
sudo python main.py # En Windows usa python get-rssi.py en powershell o cmd abierto como administrador
```
