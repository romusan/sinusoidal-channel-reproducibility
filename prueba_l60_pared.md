# Refinamiento junto al absorbedor para la onda de 60 mm

Se comparan el canal plano y el ondulado de A = 1,50 mm a Re₀ = 100 y 800 con laterales sólidos o periódicos. La cuarta malla divide en dos cada una de las cuatro primeras capas de aire junto a la pared inferior de la malla común fina. Mantiene las coordenadas axiales y transversales y las 14.580 celdas de la placa. En la onda, la altura de la primera celda pasa de 0,73–0,89 mm a 0,36–0,44 mm. Los archivos de malla de momento son idénticos entre Reynolds para cada geometría y contorno. Se verificó que la velocidad impuesta en la entrada se multiplica por ocho.

## Coeficiente pareado y contraste de Reynolds

El cambio porcentual compara el ondulado con el plano. El contraste resta el cambio a Re₀ = 100 del cambio a Re₀ = 800 y se expresa en puntos porcentuales. Cada par de cifras muestra el stencil base y el ampliado.

| Malla | Celdas de aire | Δh a 100 sólidos (%) | Δh a 100 periódicos (%) | Δh a 800 sólidos (%) | Δh a 800 periódicos (%) | Contraste sólidos (pp) | Contraste periódicos (pp) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Gruesa | 7.776 | +0,465 / +0,391 | +0,439 / +0,203 | −1,587 / −3,737 | −2,750 / −3,668 | −2,052 / −4,128 | −3,189 / −3,872 |
| Media | 26.244 | +0,411 / +0,330 | +0,435 / +0,353 | −1,789 / −1,783 | −1,804 / −1,819 | −2,200 / −2,112 | −2,239 / −2,172 |
| Fina | 91.854 | +0,360 / +0,374 | +0,349 / +0,376 | −2,423 / −2,052 | −2,401 / −2,035 | −2,783 / −2,426 | −2,750 / −2,411 |
| Cerca del absorbedor | 118.098 | +0,335 / +0,369 | +0,341 / +0,378 | −2,535 / −2,236 | −2,517 / −2,247 | −2,870 / −2,605 | −2,858 / −2,626 |

La sensibilidad descriptiva es 1,25 veces el rango entre los cuatro resultados con el stencil base más el mayor cambio al sustituirlo por el ampliado en una misma malla. La sensibilidad del contraste es 3,098 pp con laterales sólidos y 1,870 pp con periodicidad. Esta medida resume la dispersión observada. Las mallas son anisótropas y la cuarta refina solo la dirección normal junto a la pared, por lo que la medida carece de cobertura de incertidumbre.

## Cambios del coeficiente absoluto

| Reynolds | Canal | Rango de cambio entre malla fina y cuarta (%) |
| ---: | --- | ---: |
| 100 | Plano | +0,639 a +1,153 |
| 100 | Ondulado | +0,614 a +1,150 |
| 800 | Plano | −1,027 a +0,039 |
| 800 | Ondulado | −1,215 a −0,080 |

Los campos de momento cumplen log₁₀(RMS de presión) ≤ −11. El mayor desequilibrio relativo de energía es 2.01e-11. La tolerancia de las cotas nodales de temperatura es 10⁻⁶ K. Los dos stencils usan los mismos campos de velocidad y presión.

[Resultados por caso](resultados_l60_pared.csv) · [Comparaciones pareadas](comparacion_l60_pared.csv) · [Contrastes](contraste_reynolds_l60_pared.csv) · [Sensibilidad](sensibilidad_l60_pared.csv) · [Variación de h absoluto](refinamiento_h_l60_pared.csv)

Para reproducir, desde esta carpeta con el entorno descrito en README.md, ejecutar:

~~~powershell
python refinar_l60_pared.py --phase momentum --workers 2
python refinar_l60_pared.py --phase thermal --workers 2
python comparar_l60_pared.py
~~~
