#!/bin/bash

DPI_START=96
DPI_EXPORT=200

STRING_START="svg-dpi=\"$DPI_START\""
STRING_EXPORT="svg-dpi=\"$DPI_EXPORT\""

echo "s/$STRING_EXPORT/$STRING_START/g"

i="figures/Figure1"
echo "$i.svg"


cd figures

for i in "Figure1" "Figure2" "Figure3" "Figure4" "Figure5" "Extended Data Figure 1" "Extended Data Figure 2" "Extended Data Figure 3" "Extended Data Figure 4"
do
    sed -i -e "s/$STRING_START/$STRING_EXPORT/g" "$i.svg"
    inkscape --export-type="png" --export-dpi=$DPI_EXPORT "$i.svg"
    mv -f "$i.png" "pngs/$i.png"
    sed -i -e "s/$STRING_EXPORT/$STRING_START/g" "$i.svg"
done

cd ..

# sed -i -e 's/"96"/"600"/g' Figure1.svg
# inkscape --export-type="pdf" --export-dpi=600 Figure1.svg
# mv -f Figure1.pdf pdfs/Figure1.pdf
# sed -i -e 's/"600"/"96"/g' Figure1.svg
