#!/bin/bash

# Parcours de tous les fichiers sans extension dans le répertoire donné
for file in ./*; do
    # Vérification que le fichier est un fichier exécutable et n'a pas l'extension .sh
    if [ -x "$file" ] && [[ "$file" != *.sh ]] && [[ "$file" != *.cu ]] && [[ "$file" != *.cpp ]] && [[ "$file" != *.jpg ]]; then
        # Exécution du fichier
        echo "Exécution du fichier $file"
        ./$file in.jpg out.jpg
        # ./$file in2.jpg out2.jpg
    fi
done