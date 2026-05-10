# Memoria TFM en LaTeX

Proyecto LaTeX basado en la estructura de la plantilla de referencia indicada por el autor.

Compilacion recomendada desde esta carpeta:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

El contenido principal esta en `chapters/`; los anexos tecnicos estan en `appendices/`; los prefacios estan en `prefaces/`.

Antes de entrega final conviene revisar `variables.sty` para sustituir los tutores pendientes y adaptar cualquier dato institucional definitivo.
