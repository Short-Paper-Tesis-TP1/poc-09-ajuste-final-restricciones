# PoC 09: ajuste final de restricciones y combinaciones de variables

Parte del proyecto de tesis "Plataforma web para la detección temprana del riesgo de sobreendeudamiento en mujeres emprendedoras" (UPC, Taller de Proyecto I, 2026-20).

## Objetivo único
Comparar cuántas variables llevan restricción monotónica y qué combinaciones de candidatas convienen, antes de congelar el diccionario:
- **A1:** 5 restricciones.
- **A2:** solo las 4 variables con respaldo de la Res. SBS N.° 6941-2008.
- **A3:** sin restricciones.
- **B1 a B5:** una candidata a la vez sobre A2.
- **C1 a C6:** combinaciones de candidatas.

## Datos
Home Credit Default Risk (Kaggle, 2018): `application_train`, `bureau` y `bureau_balance`. Los CSV no se incluyen en el repositorio porque las reglas de la competencia no permiten redistribuirlos. Se descargan de https://www.kaggle.com/competitions/home-credit-default-risk/data.

Población: mujeres (`CODE_GENDER == "F"`). El subgrupo emprendedoras son las clientas con `NAME_INCOME_TYPE == "Commercial associate"` u `ORGANIZATION_TYPE == "Self-employed"`, y se reporta por separado.

## Método
Validación cruzada estratificada de 5 particiones con semilla 42, con la misma configuración de LGBMClassifier que la PoC 08. Las diferencias se miden frente a A2.

## Cómo correrla
```
pip install -r requirements.txt
python poc-ajuste-final.py "C:\ruta\home-credit-default-risk"
```

## Resultado
| Configuración | Variables | Restringidas | AUC-ROC todas | vs A2 | AUC-ROC emprendedoras | vs A2 | Preguntas nuevas |
|---|---|---|---|---|---|---|---|
| A1 BASE, 5 restricciones (actual) | 8 | 5 | 0.6422 | -0.0061 | 0.6291 | -0.0090 | 0 |
| A2 BASE, solo 4 restricciones SBS | 8 | 4 | 0.6483 | +0.0000 | 0.6381 | +0.0000 | 0 |
| A3 BASE, sin restricciones | 8 | 0 | 0.6562 | +0.0079 | 0.6471 | +0.0090 | 0 |
| B1 SBS + años trabajo/negocio | 9 | 4 | 0.6580 | +0.0097 | 0.6486 | +0.0105 | 1 |
| B2 SBS + años historial crediticio | 9 | 4 | 0.6607 | +0.0124 | 0.6479 | +0.0098 | 1 |
| B3 SBS + rechazos | 9 | 4 | 0.6577 | +0.0093 | 0.6462 | +0.0081 | 1 |
| B4 SBS + plazo (sin pregunta nueva) | 9 | 4 | 0.6952 | +0.0469 | 0.6895 | +0.0514 | 0 |
| B5 SBS + solicitudes último año | 9 | 4 | 0.6537 | +0.0054 | 0.6431 | +0.0050 | 1 |
| C1 SBS + trabajo + historial | 10 | 4 | 0.6676 | +0.0192 | 0.6563 | +0.0182 | 2 |
| C2 SBS + trabajo + historial + plazo | 11 | 4 | 0.7050 | +0.0567 | 0.6965 | +0.0584 | 2 |
| C3 SBS + trabajo + historial + rechazos | 11 | 4 | 0.6760 | +0.0277 | 0.6620 | +0.0239 | 3 |
| C4 SBS + las cuatro | 12 | 4 | 0.7090 | +0.0607 | 0.6986 | +0.0605 | 3 |
| C5 SBS + trabajo + plazo + solicitudes | 11 | 4 | 0.7011 | +0.0528 | 0.6938 | +0.0557 | 2 |
| C6 SBS + trabajo + plazo | 10 | 4 | 0.6994 | +0.0510 | 0.6928 | +0.0547 | 1 |

vs A2 = diferencia de AUC-ROC promedio (validación cruzada de 5 particiones) frente a A2.

## Decisión de diseño
- La base es A2: restricción monotónica solo en las 4 variables con respaldo de la SBS. Restringir la razón monto/ingreso reducía la discriminación.
- Entra B1 (años de trabajo o negocio), lo que da 9 variables.
- El plazo se verificó aparte, en la PoC 10.

## Archivos
- `poc-ajuste-final.py`: script.
- `resultado_poc_ajuste_final.json`: salida estructurada.
- `salida_ajuste_final.txt`: salida completa de la consola.
