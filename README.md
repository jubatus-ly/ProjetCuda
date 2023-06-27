# ProjetCuda2023
# Membres du groupe :
Erwan CARNEIRO 
Yanis LAGAB 
Maxime LEMOULT

Pour compiler les différentes implémentations :
Aller dans le dossier du filtre et faire
> make

Pour executer une seule version avec une image :
> ./sobel_shared nom_de_image_en_entree.jpg nom_de_image_en_sortie.jpg

Pour executer tout les versions :
> ./execut_all.sh

Commande compilation fichier Cu :
> nvcc -o exemple exemple.cu -std=c++11 -lopencv_core -lopencv_imgcodecs

Commande compilation fichier Cpp :
> g++ -o exemple exemple.cpp $(pkg-config --libs --cflags opencv)

Accès à la machine CUDA
login: cuda
mot de passe: F0b1sH3+4=7