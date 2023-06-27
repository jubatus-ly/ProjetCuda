#!/bin/bash

# Parcours de tous les fichiers sans extension dans le r�pertoire donn�
for file in ./*; do
    # V�rification que le fichier est un fichier ex�cutable et n'a pas l'extension .sh
    if [ -x "$file" ] && [[ "$file" != *.sh ]] && [[ "$file" != *.cu ]] && [[ "$file" != *.cpp ]] && [[ "$file" != *.jpg ]]; then
        # Ex�cution du fichier
        echo "Ex�cution du fichier $file"
        ./$file in.jpg out.jpg
        # ./$file in2.jpg out2.jpg
    fi
done